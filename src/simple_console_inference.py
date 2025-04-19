#!/usr/bin/env python3
"""
Simple Console-based Speech Emotion Recognition.
This version is streamlined to work reliably with simple models.
"""

import torch
import numpy as np
import pyaudio
import argparse
import time
import queue
import threading
import logging
import os
import sys
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = ['😐', '😊', '😢', '😠', '😨', '🤢', '😲']

class SimpleModel(torch.nn.Module):
    """A very simple model that works reliably"""
    def __init__(self, num_emotions=7):
        super().__init__()
        # Feature extraction
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4, stride=4),
            torch.nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4, stride=4)
        )
        
        # Global pooling and classification
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, num_emotions)
        )
        
    def forward(self, x):
        # Handle input shape
        if x.dim() == 2:  # [batch, time]
            x = x.unsqueeze(1)  # [batch, channel, time]
        
        # Process through convolutional layers
        features = self.conv(x)
        
        # Classification
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        
        # Add a dummy VAD probability of 0.7 (default to active)
        vad = torch.ones((x.size(0), 1), device=x.device) * 0.7
        
        return probs, vad

class SimpleConsoleRecognizer:
    def __init__(self, model_path, device='cpu', sample_rate=16000, 
                 chunk_size=1024, buffer_seconds=3, device_index=None):
        """Initialize the simple emotion recognizer"""
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(self.sample_rate * self.buffer_seconds)
        self.device_index = device_index
        
        # Initialize audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.audio_queue = queue.Queue()
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Initialize emotion history for smoothing
        self.emotion_history = deque(maxlen=3)  # Shorter history for responsiveness
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        
        # Setup audio
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Display the available audio devices
        self._list_audio_devices()
        
        # Processing flags
        self.is_running = False
        self.processing_thread = None
        
        # Display variables
        self.last_emotion = None
        self.update_interval = 0.3  # More frequent updates
        self.last_update_time = 0
    
    def _list_audio_devices(self):
        """List all available audio devices"""
        logger.info("Available audio devices:")
        for i in range(self.pyaudio_instance.get_device_count()):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            name = device_info.get('name', 'Unknown')
            channels = device_info.get('maxInputChannels', 0)
            if channels > 0:
                logger.info(f"  Device {i}: {name} (Input channels: {channels})")
                if self.device_index is None and channels > 0:
                    self.device_index = i
                    logger.info(f"  Selected as default input device")
    
    def _load_model(self, model_path):
        """Load model with fallback options"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Try to load the model
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                logger.info(f"Loaded checkpoint of type: {type(checkpoint)}")
                
                # Create a simple model
                model = SimpleModel(num_emotions=len(EMOTION_LABELS))
                
                # If the checkpoint is a state dict, try to load it
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Try to load weights that match
                model_dict = model.state_dict()
                if isinstance(state_dict, dict):
                    # Filter state_dict to match model's keys and shapes
                    compatible_weights = {}
                    for k, v in model_dict.items():
                        if k in state_dict and v.shape == state_dict[k].shape:
                            compatible_weights[k] = state_dict[k]
                    
                    # Update model's state dict with compatible weights
                    if compatible_weights:
                        logger.info(f"Loaded {len(compatible_weights)} compatible weights")
                        model_dict.update(compatible_weights)
                        model.load_state_dict(model_dict)
                    else:
                        logger.warning("No compatible weights found, using random initialization")
                
                model.to(self.device)
                model.eval()
                return model
                
            except Exception as e:
                logger.error(f"Error loading model from checkpoint: {e}")
                # Fallback to a fresh model
                model = SimpleModel(num_emotions=len(EMOTION_LABELS))
                model.to(self.device)
                model.eval()
                logger.info("Using randomly initialized model as fallback")
                return model
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Final fallback
            model = SimpleModel(num_emotions=len(EMOTION_LABELS))
            model.to(self.device)
            model.eval()
            logger.info("Using randomly initialized model")
            return model
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio to get audio data"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio data from the queue"""
        logger.info("Audio processing thread started")
        
        while self.is_running:
            try:
                # Get audio data from queue with timeout
                audio_data = self.audio_queue.get(timeout=1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when buffer has enough data
                if len(self.audio_buffer) >= self.buffer_size * 0.75:
                    try:
                        # Convert buffer to numpy array
                        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
                        
                        # Skip processing if audio is too quiet
                        audio_energy = np.mean(np.abs(audio_array))
                        if audio_energy < 0.01:  # Very quiet
                            continue
                        
                        # Perform inference
                        with torch.no_grad():
                            # Convert to tensor
                            waveform = torch.from_numpy(audio_array).float().to(self.device)
                            
                            # Add batch dimension if needed
                            if waveform.dim() == 1:
                                waveform = waveform.unsqueeze(0)
                            
                            # Model inference
                            outputs = self.model(waveform)
                            
                            # Extract probabilities
                            if isinstance(outputs, tuple) and len(outputs) >= 1:
                                emotion_probs = outputs[0]
                                vad_probs = outputs[1] if len(outputs) > 1 else torch.tensor([[0.7]], device=self.device)
                            else:
                                emotion_probs = outputs
                                vad_probs = torch.tensor([[0.7]], device=self.device)
                            
                            # Convert to numpy
                            emotion_probs_np = emotion_probs.squeeze().cpu().numpy()
                            vad_value = vad_probs.item() if hasattr(vad_probs, 'item') else 0.7
                            
                            # Only update if voice activity detected
                            if vad_value > 0.5:
                                self.emotion_history.append(emotion_probs_np)
                                self.emotion_probs = np.mean(self.emotion_history, axis=0)
                                
                                # Update display periodically
                                current_time = time.time()
                                if current_time - self.last_update_time >= self.update_interval:
                                    self.update_console_display()
                                    self.last_update_time = current_time
                    
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
            
            except queue.Empty:
                # No data available, just continue
                pass
            except Exception as e:
                logger.error(f"Error in audio queue: {e}")
    
    def update_console_display(self):
        """Update the console display with current emotion probabilities"""
        try:
            # Get top emotion
            top_idx = np.argmax(self.emotion_probs)
            top_emotion = EMOTION_LABELS[top_idx]
            confidence = self.emotion_probs[top_idx]
            
            # Skip if confidence is very low or emotion hasn't changed
            if confidence < 0.15 or (self.last_emotion == top_emotion and confidence < 0.6):
                return
            
            # Update last emotion
            self.last_emotion = top_emotion
            
            # Clear line and print current emotion
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            emoji = EMOTION_EMOJIS[top_idx]
            
            # Create a simple bar for the top emotion
            bar_length = 20
            bar = '|' * bar_length
            
            # Print the top emotion with emphasis
            sys.stdout.write(f"\r{emoji} {top_emotion.upper()}: {confidence:.2f} [{bar}] {emoji}\n")
            
            # Print other emotions
            for i, (label, prob) in enumerate(zip(EMOTION_LABELS, self.emotion_probs)):
                if i != top_idx:
                    # Create proportional bar
                    bar_value = int(prob * bar_length)
                    bar = '|' * bar_value + ' ' * (bar_length - bar_value)
                    sys.stdout.write(f"  {label}: {prob:.2f} [{bar}]\n")
            
            # Move cursor back up for next update
            sys.stdout.write(f"\033[{len(EMOTION_LABELS)}A")
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Error updating display: {e}")
    
    def start(self):
        """Start real-time emotion recognition"""
        self.is_running = True
        
        # Start audio stream
        try:
            logger.info(f"Opening audio stream with device {self.device_index}")
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            logger.info("Audio stream opened successfully")
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self.is_running = False
            return
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time emotion recognition started")
        
        # Make space for emotion display
        print("\n" * len(EMOTION_LABELS))
        
        try:
            # Keep the main thread running
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()
    
    def stop(self):
        """Stop real-time emotion recognition"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
        self.pyaudio_instance.terminate()
        
        # Add newlines to clear display area
        print("\n" * (len(EMOTION_LABELS) + 1))
        logger.info("Real-time emotion recognition stopped")

def main():
    parser = argparse.ArgumentParser(description="Simple Console-based Speech Emotion Recognition")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=int, default=3, help="Audio buffer length in seconds")
    parser.add_argument("--device_index", type=int, help="Audio device index (default: auto-detect)")
    parser.add_argument("--log_file", type=str, help="Log file path (default: console only)")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print header
    print("""
╔════════════════════════════════════════════╗
║     Real-time Speech Emotion Recognition    ║
║            Simple Console Edition           ║
╚════════════════════════════════════════════╝
    
Listening for audio input... Speak to detect emotions.
Press Ctrl+C to exit.
""")
    
    # Initialize recognizer
    recognizer = SimpleConsoleRecognizer(
        model_path=args.model,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds,
        device_index=args.device_index
    )
    
    # Start recognition
    recognizer.start()

if __name__ == "__main__":
    main() 