#!/usr/bin/env python3
"""
Real-time Speech Emotion Recognition inference script.
This module provides functions for real-time speech emotion recognition.
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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import threading
from pathlib import Path
import yaml
import logging
from collections import deque

# Custom imports
from model import SpeechEmotionRecognitionModel
from model_enhanced import EnhancedSpeechEmotionRecognitionModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

class RealTimeEmotionRecognizer:
    def __init__(self, model_path, config_path=None, use_enhanced_model=False, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 sample_rate=16000, chunk_size=1024, buffer_seconds=3):
        """
        Initialize the real-time emotion recognizer.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to the configuration file
            use_enhanced_model: Whether to use the enhanced model
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
        
        # Load configuration if provided
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'model': {'model_name': 'facebook/wav2vec2-base-960h'}}
        
        # Initialize model
        logger.info(f"Loading model from {model_path} (using {'enhanced' if use_enhanced_model else 'standard'} model)")
        if use_enhanced_model:
            self.model = EnhancedSpeechEmotionRecognitionModel(
                model_name=self.config['model'].get('model_name', 'facebook/wav2vec2-base-960h')
            )
        else:
            self.model = SpeechEmotionRecognitionModel(
                model_name=self.config['model'].get('model_name', 'facebook/wav2vec2-base-960h')
            )
        
        # Load model checkpoint
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        
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
                    
                    # Perform inference
                    with torch.no_grad():
                        waveform = torch.from_numpy(audio_array).float().to(self.device)
                        waveform = waveform.unsqueeze(0)  # Add batch dimension
                        
                        # Model inference
                        emotion_probs, vad_probs = self.model(waveform)
                        
                        # Convert to numpy
                        emotion_probs = emotion_probs.squeeze().cpu().numpy()
                        vad_probs = vad_probs.squeeze().cpu().numpy()
                        
                        # Only update emotion if voice activity is detected
                        if vad_probs > 0.5:
                            self.emotion_history.append(emotion_probs)
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
        
        logger.info("Real-time emotion recognition started")
        
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
        self.ax.set_title('Real-time Emotion Recognition')
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
    parser = argparse.ArgumentParser(description="Real-time Speech Emotion Recognition")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced model")
    parser.add_argument("--no_visual", action="store_true", help="Disable visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=int, default=3, help="Length of audio buffer in seconds")
    
    args = parser.parse_args()
    
    # Initialize emotion recognizer
    recognizer = RealTimeEmotionRecognizer(
        model_path=args.model_path,
        config_path=args.config,
        use_enhanced_model=args.enhanced,
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
                print(f"\rCurrent emotion: {predicted_emotion}", end="")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main() 