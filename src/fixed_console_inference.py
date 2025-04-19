#!/usr/bin/env python3
"""
Fixed Console Inference for Speech Emotion Recognition
Uses the fixed SimpleEmotionRecognitionModel to ensure compatibility with saved weights.
"""

import torch

def safe_index(value, idx=0, default=0.0):
    '''Safely index a value that might be a scalar or array.'''
    if isinstance(value, (float, int, bool)):
        return value
    try:
        return value[idx]
    except (IndexError, TypeError):
        return default

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

# Import the fixed model
from fixed_simple_model import SimpleEmotionRecognitionModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels and emojis
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = ['😐', '😊', '😢', '😠', '😨', '🤢', '😲']

class FixedConsoleRecognizer:
    def __init__(self, model_path, device='cpu', sample_rate=16000, 
                 chunk_size=1024, buffer_seconds=3, device_index=None):
        """Initialize the console emotion recognizer"""
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
        self.emotion_history = deque(maxlen=3)
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        
        # Setup audio
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Display available audio devices
        self._list_audio_devices()
        
        # Processing flags
        self.is_running = False
        self.processing_thread = None
        
        # Display variables
        self.last_emotion = None
        self.update_interval = 0.3
        self.last_update_time = 0
    
    def _list_audio_devices(self):
        """List available audio input devices"""
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
        """Load the model with proper error handling"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create model instance
            model = SimpleEmotionRecognitionModel(num_emotions=len(EMOTION_LABELS))
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded model from state_dict in checkpoint")
            else:
                try:
                    model.load_state_dict(checkpoint)
                    logger.info("Loaded model directly from checkpoint")
                except Exception as e:
                    logger.warning(f"Error loading model state: {e}")
                    logger.warning("Using model with random weights")
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback - create a new model
            model = SimpleEmotionRecognitionModel(num_emotions=len(EMOTION_LABELS))
            model.to(self.device)
            model.eval()
            logger.info("Using new model with random weights")
            return model
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio data from the queue"""
        logger.info("Audio processing thread started")
        
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when buffer has enough data
                if len(self.audio_buffer) >= self.buffer_size * 0.75:
                    # Convert buffer to numpy array
                    audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
                    
                    # Skip if audio is too quiet
                    audio_energy = np.mean(np.abs(audio_array))
                    if audio_energy < 0.01:
                        continue
                    
                    # Perform inference
                    with torch.no_grad():
                        # Convert to tensor
                        waveform = torch.from_numpy(audio_array).float().to(self.device)
                        
                        # Add batch dimension if needed
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        
                        # Run inference
                        try:
                            emotion_probs, vad_probs = self.model(waveform)
                            
                            # Convert to numpy
                            emotion_probs_np = emotion_probs.cpu().numpy().squeeze()
                            vad_value = vad_probs.cpu().numpy().item()
                            
                            # Only update if voice activity detected
                            if vad_value > 0.5:
                                self.emotion_history.append(emotion_probs_np)
                                self.emotion_probs = np.mean(self.emotion_history, axis=0)
                                
                                # Update display
                                current_time = time.time()
                                if current_time - self.last_update_time >= self.update_interval:
                                    self.update_console_display()
                                    self.last_update_time = current_time
                        
                        except Exception as e:
                            logger.error(f"Error during inference: {e}")
            
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
    
    def update_console_display(self):
        """Update the console display with emotion probabilities"""
        try:
            # Get top emotion
            top_idx = np.argmax(self.emotion_probs)
            top_emotion = EMOTION_LABELS[top_idx]
            confidence = self.emotion_probs[top_idx]
            
            # Skip if confidence is low or emotion hasn't changed
            if confidence < 0.20 or (self.last_emotion == top_emotion and confidence < 0.6):
                return
            
            # Update last emotion
            self.last_emotion = top_emotion
            
            # Clear line
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            emoji = EMOTION_EMOJIS[top_idx]
            
            # Create bar for top emotion
            bar_length = 20
            bar = '|' * bar_length
            
            # Print top emotion with emphasis
            sys.stdout.write(f"\r{emoji} {top_emotion.upper()}: {confidence:.2f} [{bar}] {emoji}\n")
            
            # Print other emotions
            for i, (label, prob) in enumerate(zip(EMOTION_LABELS, self.emotion_probs)):
                if i != top_idx:
                    bar_value = int(prob * bar_length)
                    bar = '|' * bar_value + ' ' * (bar_length - bar_value)
                    sys.stdout.write(f"  {label}: {prob:.2f} [{bar}]\n")
            
            # Move cursor back up
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
            # Keep main thread running
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.stop()
    
    def stop(self):
        """Stop emotion recognition"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
        self.pyaudio_instance.terminate()
        
        print("\n" * (len(EMOTION_LABELS) + 1))
        logger.info("Real-time emotion recognition stopped")

def main():
    parser = argparse.ArgumentParser(description="Fixed Console-based Speech Emotion Recognition")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
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
║             Fixed Console Edition           ║
╚════════════════════════════════════════════╝
    
Listening for audio input... Speak to detect emotions.
Press Ctrl+C to exit.
""")
    
    # Initialize recognizer
    recognizer = FixedConsoleRecognizer(
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