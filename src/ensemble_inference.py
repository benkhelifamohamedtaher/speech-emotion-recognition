#!/usr/bin/env python3
"""
Ensemble Inference for Speech Emotion Recognition
This script uses multiple trained models in ensemble to improve emotion recognition accuracy.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyaudio
import time
import threading
import queue
import argparse
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import logging

# Import model implementations
from model_fixed import FixedSpeechEmotionRecognitionModel, EnhancedFixedSpeechEmotionRecognitionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_COLORS = ['#808080', '#FFD700', '#1E90FF', '#FF4500', '#800080', '#006400', '#FF69B4']

class EnsembleEmotionRecognizer:
    """Ensemble model for real-time emotion recognition from speech"""
    def __init__(self, model_paths, weights=None, device='cpu', sample_rate=16000, chunk_size=1024, buffer_seconds=3):
        """
        Initialize the ensemble recognizer
        Args:
            model_paths: List of paths to trained model checkpoints
            weights: Weights for each model in the ensemble (None for equal weighting)
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
        self.models = []
        for path in model_paths:
            try:
                checkpoint = torch.load(path, map_location=device)
                
                # Determine model type based on state dict keys or checkpoint metadata
                if 'multihead_attn.in_proj_weight' in checkpoint['model_state_dict']:
                    # This is likely an enhanced model
                    model = EnhancedFixedSpeechEmotionRecognitionModel(
                        num_emotions=len(EMOTION_LABELS)
                    )
                    logger.info(f"Loading enhanced model from {path}")
                else:
                    # This is likely a basic model
                    model = FixedSpeechEmotionRecognitionModel(
                        num_emotions=len(EMOTION_LABELS)
                    )
                    logger.info(f"Loading base model from {path}")
                
                # Load weights with error handling
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e:
                    logger.warning(f"Error loading exact state dict: {e}. Trying with strict=False...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
                model.to(device)
                model.eval()
                self.models.append(model)
                
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
        
        # Set model weights (equal by default)
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
            
        logger.info(f"Loaded {len(self.models)} models with weights {self.weights}")
        
        if not self.models:
            raise ValueError("No models could be loaded. Ensemble cannot proceed.")
        
        # Initialize emotion history for smoothing
        self.emotion_history = deque(maxlen=10)
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        
        # Audio stream
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Processing thread
        self.is_running = False
        self.processing_thread = None
        
        # Visualization
        self.fig = None
        self.ax = None
        self.bars = None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio data from the queue and perform ensemble inference"""
        logger.info("Audio processing started. Listening...")
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when buffer is full enough
                if len(self.audio_buffer) >= self.buffer_size * 0.75:
                    # Convert buffer to numpy array
                    audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
                    
                    # Preprocess audio
                    # - Center audio
                    audio_array = audio_array - audio_array.mean()
                    # - Normalize
                    if audio_array.std() > 0:
                        audio_array = audio_array / audio_array.std()
                    
                    # Convert to tensor
                    waveform = torch.from_numpy(audio_array).float().to(self.device)
                    waveform = waveform.unsqueeze(0)  # Add batch dimension
                    
                    # Make predictions with each model
                    ensemble_probs = np.zeros(len(EMOTION_LABELS))
                    voice_activity = 0.0
                    
                    for i, model in enumerate(self.models):
                        with torch.no_grad():
                            try:
                                # Forward pass
                                outputs = model(waveform)
                                
                                # Extract emotion probabilities and VAD
                                if isinstance(outputs, dict):
                                    emotion_probs = outputs["emotion_probs"].squeeze().cpu().numpy()
                                    vad_prob = outputs["vad_probs"].squeeze().cpu().numpy()
                                elif isinstance(outputs, tuple):
                                    emotion_probs, vad_prob = outputs
                                    emotion_probs = emotion_probs.squeeze().cpu().numpy()
                                    vad_prob = vad_prob.squeeze().cpu().numpy()
                                else:
                                    emotion_probs = outputs.squeeze().cpu().numpy()
                                    vad_prob = np.array([0.5])  # Default VAD if not available
                                
                                # Add to ensemble with weight
                                ensemble_probs += emotion_probs * self.weights[i]
                                voice_activity += float(vad_prob) * self.weights[i]
                                
                            except Exception as e:
                                logger.error(f"Error in model {i} inference: {e}")
                                # Skip this model
                                continue
                    
                    # Only update emotion if voice activity is detected
                    if voice_activity > 0.5:
                        self.emotion_history.append(ensemble_probs)
                        # Smooth predictions with exponential moving average
                        if len(self.emotion_history) > 1:
                            alpha = 0.3  # Smoothing factor
                            self.emotion_probs = alpha * ensemble_probs + (1 - alpha) * self.emotion_probs
                        else:
                            self.emotion_probs = ensemble_probs
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
    
    def start(self, visualize=True):
        """Start real-time emotion recognition with ensemble"""
        if not self.models:
            logger.error("No models available. Cannot start inference.")
            return
        
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
        
        logger.info("Ensemble emotion recognition started")
        
        # Start visualization if requested
        if visualize:
            self.visualize()
        else:
            try:
                # Keep the main thread running
                while self.is_running:
                    time.sleep(0.1)
                    
                    # Print predicted emotion
                    predicted_emotion = EMOTION_LABELS[np.argmax(self.emotion_probs)]
                    confidence = self.emotion_probs[np.argmax(self.emotion_probs)]
                    print(f"\rCurrent emotion: {predicted_emotion} (confidence: {confidence:.2f})", end="")
                    
            except KeyboardInterrupt:
                print("\nStopping...")
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
        logger.info("Ensemble emotion recognition stopped")
    
    def visualize(self):
        """Visualize emotion probabilities in real-time"""
        plt.style.use('ggplot')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.bars = self.ax.bar(EMOTION_LABELS, self.emotion_probs, color=EMOTION_COLORS)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Ensemble Emotion Recognition')
        self.ax.set_ylabel('Probability')
        
        def update_plot(frame):
            # Update bar heights
            for bar, prob in zip(self.bars, self.emotion_probs):
                bar.set_height(prob)
            
            # Highlight the predicted emotion
            predicted_idx = np.argmax(self.emotion_probs)
            for i, bar in enumerate(self.bars):
                bar.set_alpha(1.0 if i == predicted_idx else 0.7)
            
            # Add text label for the dominant emotion
            self.ax.set_title(f'Detected Emotion: {EMOTION_LABELS[predicted_idx].capitalize()}')
            
            return self.bars
        
        # Create animation
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(self.fig, update_plot, interval=100, blit=True)
        plt.tight_layout()
        
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close()
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Ensemble Speech Emotion Recognition")
    parser.add_argument("--model_dirs", type=str, nargs='+', required=True, 
                        help="Directories containing trained models (will use best_model.pt from each)")
    parser.add_argument("--weights", type=float, nargs='+', 
                        help="Weights for each model in the ensemble (default: equal weighting)")
    parser.add_argument("--no_visual", action="store_true", help="Disable visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=int, default=3, help="Length of audio buffer in seconds")
    
    args = parser.parse_args()
    
    # Collect model paths
    model_paths = []
    for directory in args.model_dirs:
        model_path = os.path.join(directory, 'best_model.pt')
        if os.path.exists(model_path):
            model_paths.append(model_path)
        else:
            logger.warning(f"No best_model.pt found in {directory}")
    
    if not model_paths:
        logger.error("No models found. Cannot proceed.")
        return
    
    logger.info(f"Using models: {model_paths}")
    
    # Initialize ensemble recognizer
    recognizer = EnsembleEmotionRecognizer(
        model_paths=model_paths,
        weights=args.weights,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds
    )
    
    try:
        # Start recognition
        recognizer.start(visualize=not args.no_visual)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main() 