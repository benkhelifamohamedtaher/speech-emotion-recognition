#!/usr/bin/env python3
"""
Ensemble Speech Emotion Recognition inference script.
This script combines predictions from multiple models for more accurate emotion recognition.
"""

import torch
import numpy as np
import pyaudio
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import threading
from pathlib import Path
import logging
from collections import deque
import warnings
import glob
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

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


class EnsembleEmotionRecognizer:
    def __init__(self, models_dir, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 sample_rate=16000, chunk_size=1024, buffer_seconds=3):
        """
        Initialize the ensemble emotion recognizer.
        
        Args:
            models_dir: Directory containing trained model checkpoints
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
        
        # Load models
        logger.info(f"Loading models from {models_dir}")
        self.models = self._load_models(models_dir)
        
        if not self.models:
            logger.error("No models loaded. Using fallback model.")
            self.models = [SimpleEmotionRecognitionModel(num_emotions=7).to(self.device)]
        
        logger.info(f"Loaded {len(self.models)} models for ensemble prediction")
        
        # Initialize emotion history for smoothing
        self.emotion_history = deque(maxlen=10)
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        
        # Setup audio
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Visualization
        self.fig = None
        self.ax = None
        self.bars = None
        
        # Processing flag
        self.is_running = False
        self.processing_thread = None
    
    def _load_model(self, model_path):
        """Load a single model with error handling and fallbacks"""
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check if it's a state_dict or full model
            if 'model_state_dict' in checkpoint:
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
                logger.info(f"Loaded SpeechEmotionRecognitionModel from {model_path}")
            except Exception as e:
                logger.warning(f"Error loading standard model: {e}")
                try:
                    from model_enhanced import EnhancedSpeechEmotionRecognitionModel
                    model = EnhancedSpeechEmotionRecognitionModel(num_emotions=num_emotions)
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded EnhancedSpeechEmotionRecognitionModel from {model_path}")
                except Exception as e:
                    logger.warning(f"Error loading enhanced model: {e}")
                    # Fallback to simple model
                    model = SimpleEmotionRecognitionModel(num_emotions=num_emotions)
                    
                    # Only load compatible keys
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in state_dict.items() 
                                      if k in model_dict and model_dict[k].shape == v.shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    
                    logger.info(f"Using SimpleEmotionRecognitionModel fallback for {model_path}")
            
            # Move model to device and set to eval mode
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None
    
    def _load_models(self, models_dir):
        """Load all models from the directory"""
        models = []
        
        # Find best models in subdirectories
        for subdir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, subdir, 'best_model.pt')
            if os.path.exists(model_path):
                model = self._load_model(model_path)
                if model is not None:
                    models.append(model)
        
        # If no models found, look for direct pt files in the directory
        if not models:
            for model_file in glob.glob(os.path.join(models_dir, '*.pt')):
                model = self._load_model(model_file)
                if model is not None:
                    models.append(model)
        
        return models
    
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
                    # Convert buffer to numpy array
                    audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
                    
                    # Perform inference with all models
                    ensemble_probs = []
                    
                    with torch.no_grad():
                        waveform = torch.from_numpy(audio_array).float().to(self.device)
                        
                        # Handle dimensions
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)  # Add batch dimension [batch_size, seq_len]
                        
                        # Voice activity detection from first model (for simplicity)
                        vad_prob = 0.7  # Default value
                        
                        # Get predictions from all models
                        for model in self.models:
                            try:
                                # Model inference
                                outputs = model(waveform)
                                
                                # Extract emotion probabilities
                                if isinstance(outputs, tuple):
                                    emotion_probs, model_vad = outputs
                                    vad_prob = model_vad.squeeze().cpu().numpy()  # Use latest VAD
                                elif isinstance(outputs, dict):
                                    emotion_probs = outputs.get('emotion_probs', outputs.get('emotion_logits'))
                                    if 'vad_probs' in outputs:
                                        vad_prob = outputs['vad_probs'].squeeze().cpu().numpy()
                                else:
                                    emotion_probs = outputs
                                
                                # Add to ensemble
                                ensemble_probs.append(emotion_probs.squeeze().cpu().numpy())
                                
                            except RuntimeError as e:
                                logger.error(f"Inference error for model: {e}")
                                # Skip this model
                                continue
                    
                    # Average predictions from all models that worked
                    if ensemble_probs:
                        # Only update emotion if voice activity is detected
                        if vad_prob > 0.5:
                            # Average ensemble predictions
                            avg_probs = np.mean(ensemble_probs, axis=0)
                            self.emotion_history.append(avg_probs)
                            
                            # Smooth predictions with a moving average
                            self.emotion_probs = np.mean(self.emotion_history, axis=0)
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
    
    def start(self, visualize=True):
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
        
        logger.info("Real-time ensemble emotion recognition started")
        
        # Start visualization if requested
        if visualize:
            self.visualize()
    
    def stop(self):
        """Stop real-time emotion recognition"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
        self.pyaudio_instance.terminate()
        logger.info("Real-time emotion recognition stopped")
    
    def visualize(self):
        """Visualize emotion probabilities in real-time"""
        plt.style.use('ggplot')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bars = self.ax.bar(EMOTION_LABELS, self.emotion_probs)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Real-time Ensemble Emotion Recognition')
        self.ax.set_ylabel('Probability')
        
        def update_plot(frame):
            for bar, prob in zip(self.bars, self.emotion_probs):
                bar.set_height(prob)
            
            # Highlight the predicted emotion
            predicted_emotion = np.argmax(self.emotion_probs)
            for i, bar in enumerate(self.bars):
                if i == predicted_emotion:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            return self.bars
        
        # Create animation
        ani = FuncAnimation(self.fig, update_plot, interval=100, blit=True)
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Ensemble Speech Emotion Recognition")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--no_visual", action="store_true", help="Disable visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=int, default=3, help="Length of audio buffer in seconds")
    
    args = parser.parse_args()
    
    # Initialize ensemble recognizer
    recognizer = EnsembleEmotionRecognizer(
        models_dir=args.models_dir,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds
    )
    
    try:
        # Start recognition
        recognizer.start(visualize=not args.no_visual)
        
        # If no visualization, keep running
        if args.no_visual:
            print("Press Ctrl+C to stop...")
            while recognizer.is_running:
                time.sleep(1)
                
                # Print predicted emotion
                predicted_emotion = EMOTION_LABELS[np.argmax(recognizer.emotion_probs)]
                confidence = recognizer.emotion_probs[np.argmax(recognizer.emotion_probs)]
                print(f"\rCurrent emotion: {predicted_emotion} (confidence: {confidence:.2f})", end="")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recognizer.stop()

if __name__ == "__main__":
    main() 