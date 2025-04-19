#!/usr/bin/env python3
"""
Console-based Real-time Speech Emotion Recognition inference script.
This version runs in terminal environments without requiring GUI components.
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
from collections import deque
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_EMOJIS = ['😐', '😊', '😢', '😠', '😨', '🤢', '😲']

# Simple model fallback if transformers fails
class SimpleEmotionRecognitionModel(torch.nn.Module):
    """Simple speech emotion recognition model that handles tensor dimensions properly"""
    def __init__(self, num_emotions=7):
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=4),
            
            torch.nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=4),
            
            torch.nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)  # Global pooling to handle variable length
        )
        
        # Fully connected layers for classification
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_emotions)
        )
        
        # Voice activity detection branch
        self.vad = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        # Handle different input shapes
        if x.dim() == 2:  # [batch_size, sequence_length]
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, sequence_length]
        elif x.dim() == 3 and x.shape[1] != 1:  # Wrong channel dimension
            x = x.transpose(1, 2)  # Transpose to [batch_size, 1, sequence_length]
        elif x.dim() == 4:  # Extra dimension
            x = x.squeeze(2)  # Remove extra dimension
            
        # Apply convolutional layers
        features = self.conv_layers(x)
        features = features.squeeze(-1)  # Remove last dimension after global pooling
        
        # Apply classifiers
        emotion_logits = self.classifier(features)
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        
        # Voice activity detection
        vad_probs = self.vad(features)
        
        return emotion_probs, vad_probs


class ConsoleEmotionRecognizer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 sample_rate=16000, chunk_size=1024, buffer_seconds=3):
        """
        Initialize the console-based emotion recognizer.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            sample_rate: Audio sample rate
            chunk_size: PyAudio chunk size
            buffer_seconds: Length of audio buffer in seconds
        """
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.buffer_size = self.sample_rate * self.buffer_seconds
        
        # Initialize audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.audio_queue = queue.Queue()
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        
        # Initialize emotion history for smoothing
        self.emotion_history = deque(maxlen=5)
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        
        # Setup audio
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Processing flag
        self.is_running = False
        self.processing_thread = None
        
        # Last displayed emotion
        self.last_emotion = None
        self.update_interval = 0.5  # Update display every 0.5 seconds
        self.last_update_time = 0
    
    def _load_model(self, model_path):
        """Load the trained model with error handling and fallbacks"""
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's a state_dict or full model
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                num_emotions = checkpoint.get('num_emotions', 7)
            else:
                state_dict = checkpoint
                num_emotions = 7  # Default
            
            # Try to load custom models
            try:
                # Try importing custom models
                from model import SpeechEmotionRecognitionModel
                model = SpeechEmotionRecognitionModel(num_emotions=num_emotions)
                model.load_state_dict(state_dict)
                logger.info("Loaded SpeechEmotionRecognitionModel")
            except Exception as e:
                logger.warning(f"Error loading custom model: {e}")
                try:
                    from model_enhanced import EnhancedSpeechEmotionRecognitionModel
                    model = EnhancedSpeechEmotionRecognitionModel(num_emotions=num_emotions)
                    model.load_state_dict(state_dict)
                    logger.info("Loaded EnhancedSpeechEmotionRecognitionModel")
                except Exception as e:
                    logger.warning(f"Error loading enhanced model: {e}")
                    # Fallback to simple model
                    model = SimpleEmotionRecognitionModel(num_emotions=num_emotions)
                    
                    # Only load compatible keys
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    
                    logger.info("Using SimpleEmotionRecognitionModel fallback")
            
            # Move model to device and set to eval mode
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Final fallback - create a new simple model
            model = SimpleEmotionRecognitionModel(num_emotions=7)
            model.to(self.device)
            model.eval()
            logger.info("Using new SimpleEmotionRecognitionModel as fallback")
            return model
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio data from the queue and perform emotion recognition"""
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when buffer is full
                if len(self.audio_buffer) >= self.buffer_size * 0.75:
                    try:
                        # Convert buffer to numpy array
                        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
                        
                        # Perform inference
                        with torch.no_grad():
                            # Ensure proper tensor shape
                            waveform = torch.from_numpy(audio_array).float().to(self.device)
                            
                            # Handle dimensions
                            if waveform.dim() == 1:
                                waveform = waveform.unsqueeze(0)  # Add batch dimension [batch_size, seq_len]
                            
                            # Model inference with error handling
                            outputs = self.model(waveform)
                            
                            # Extract emotion probabilities
                            if isinstance(outputs, tuple) and len(outputs) >= 1:
                                emotion_probs = outputs if isinstance(outputs, (float, int, bool)) else outputs[0]
                                vad_probs = outputs[1] if len(outputs) > 1 else torch.tensor([0.7], device=self.device)
                            elif isinstance(outputs, dict):
                                emotion_probs = outputs.get('emotion_probs', outputs.get('emotion_logits'))
                                vad_probs = outputs.get('vad_probs', torch.tensor([0.7], device=self.device))
                            else:
                                emotion_probs = outputs
                                vad_probs = torch.tensor([0.7], device=self.device)
                            
                            # Ensure emotion_probs is a tensor with proper shape
                            if not isinstance(emotion_probs, torch.Tensor):
                                raise ValueError("Model output is not a tensor")
                            
                            # Handle single dimension outputs
                            if emotion_probs.dim() == 1:
                                emotion_probs = emotion_probs.unsqueeze(0)
                                
                            # Convert to numpy safely
                            emotion_probs_np = emotion_probs.squeeze().cpu().detach().numpy()
                            
                            # Ensure proper dimensionality after squeezing
                            if emotion_probs_np.ndim == 0:  # Handle scalar case
                                emotion_probs_np = np.array([1.0] + [0.0] * (len(EMOTION_LABELS)-1))
                            elif len(emotion_probs_np) != len(EMOTION_LABELS):
                                # Pad or truncate as needed
                                if len(emotion_probs_np) < len(EMOTION_LABELS):
                                    emotion_probs_np = np.pad(emotion_probs_np, 
                                                          (0, len(EMOTION_LABELS) - len(emotion_probs_np)), 
                                                          'constant')
                                else:
                                    emotion_probs_np = emotion_probs_np[:len(EMOTION_LABELS)]
                            
                            # Handle vad_probs safely
                            if isinstance(vad_probs, torch.Tensor):
                                vad_value = vad_probs.item() if vad_probs.numel() == 1 else 0.7
                            else:
                                vad_value = 0.7
                            
                            # Only update emotion if voice activity is detected
                            if vad_value > 0.5:
                                self.emotion_history.append(emotion_probs_np)
                                # Smooth predictions with a moving average
                                self.emotion_probs = np.mean(self.emotion_history, axis=0)
                                
                                # Update console display at intervals
                                current_time = time.time()
                                if current_time - self.last_update_time >= self.update_interval:
                                    self.update_console_display()
                                    self.last_update_time = current_time
                    
                    except Exception as e:
                        logger.error(f"Error in audio processing: {str(e)}")
                        # Don't break the loop, just log the error and continue
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio queue processing: {str(e)}")
    
    def update_console_display(self):
        """Update the console display with current emotion probabilities"""
        try:
            # Get predicted emotion and confidence
            predicted_idx = np.argmax(self.emotion_probs)
            predicted_emotion = EMOTION_LABELS[predicted_idx]
            confidence = self.emotion_probs[predicted_idx]
            
            # Skip if not changed significantly
            if self.last_emotion == predicted_emotion and confidence < 0.6:
                return
            
            self.last_emotion = predicted_emotion
            
            # Clear line
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            
            # Create bar chart
            emoji = EMOTION_EMOJIS[predicted_idx]
            bar_length = 20
            
            for i, (label, prob) in enumerate(zip(EMOTION_LABELS, self.emotion_probs)):
                bar = int(prob * bar_length)
                if i == predicted_idx:
                    # Highlight predicted emotion
                    sys.stdout.write(f"\r{emoji} {label.upper()}: {prob:.2f} [{bar_length * '|'}] {emoji}\n")
                else:
                    # Standard bar for other emotions
                    bar_str = bar * '|' + (bar_length - bar) * ' '
                    sys.stdout.write(f"  {label}: {prob:.2f} [{bar_str}]\n")
            
            # Move cursor back up for the next update
            sys.stdout.write(f"\033[{len(EMOTION_LABELS)}A")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error updating console display: {str(e)}")
    
    def start(self):
        """Start real-time emotion recognition"""
        self.is_running = True
        
        # Start audio stream
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
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
        
        # Move cursor down past the emotion display
        print("\n" * (len(EMOTION_LABELS) + 1))
        logger.info("Real-time emotion recognition stopped")


def main():
    parser = argparse.ArgumentParser(description="Console-based Real-time Speech Emotion Recognition")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=int, default=3, help="Length of audio buffer in seconds")
    
    args = parser.parse_args()
    
    # Initialize emotion recognizer
    recognizer = ConsoleEmotionRecognizer(
        model_path=args.model,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds
    )
    
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("""
╔════════════════════════════════════════════╗
║     Real-time Speech Emotion Recognition    ║
║              Console Edition                ║
╚════════════════════════════════════════════╝
    
Listening for audio input... Speak to detect emotions.
Press Ctrl+C to exit.
""")
    
    # Start recognition
    recognizer.start()

if __name__ == "__main__":
    main() 