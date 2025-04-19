#!/usr/bin/env python3
"""
Real-Time Speech Emotion Recognition Using Optimal Model
"""

import os
import sys
import torch
import numpy as np
import argparse
import pyaudio
import threading
import time
import logging
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import our custom model
from optimal_model import OptimalSpeechEmotionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define emotion mappings for full and simplified emotion sets
FULL_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SIMPLIFIED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

# Map emotions to emojis
EMOJI_MAP = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😄',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgust': '🤢',
    'surprised': '😲'
}

class EmotionRecognizer:
    """Real-time speech emotion recognition"""
    def __init__(
        self,
        model_path,
        sample_rate=16000,
        chunk_size=1024,
        buffer_seconds=3.0,
        simplified=False,
        device="cpu",
        visualize=True
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.buffer_frames = int(sample_rate * buffer_seconds)
        self.device = device
        self.visualize = visualize
        self.simplified = simplified
        
        # Set emotion labels based on emotion type
        self.emotion_labels = SIMPLIFIED_EMOTIONS if simplified else FULL_EMOTIONS
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = OptimalSpeechEmotionModel.from_pretrained(model_path, map_location=device)
        self.model.to(device)
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Initialize audio buffer
        self.audio_buffer = np.zeros(self.buffer_frames, dtype=np.float32)
        
        # Initialize emotion probabilities
        self.emotion_probs = np.zeros(len(self.emotion_labels))
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        
        # Flag to control recognition loop
        self.running = False
        
        # Queue for audio data
        self.audio_queue = Queue()
        
        # Initialize visualization
        if visualize:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.bars = None
            self.prediction_text = None
    
    def start(self):
        """Start emotion recognition"""
        self.running = True
        
        # Start audio streaming in a separate thread
        self.audio_thread = threading.Thread(target=self.audio_stream)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start recognition in a separate thread
        self.recognition_thread = threading.Thread(target=self.recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
        # Start visualization if enabled
        if self.visualize:
            self.setup_visualization()
            plt.show()
        else:
            # Wait for Ctrl+C
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop()
    
    def setup_visualization(self):
        """Setup visualization plot"""
        # Create horizontal bar chart
        self.bars = self.ax.barh(
            self.emotion_labels,
            np.zeros(len(self.emotion_labels)),
            color=plt.cm.viridis(np.linspace(0, 1, len(self.emotion_labels)))
        )
        
        # Add emoji indicators
        for i, emotion in enumerate(self.emotion_labels):
            emoji = EMOJI_MAP.get(emotion, "")
            self.ax.text(1.01, i, emoji, fontsize=20, va='center')
        
        # Add labels
        self.ax.set_xlim(0, 1)
        self.ax.set_title("Real-time Speech Emotion Recognition", fontsize=16)
        self.ax.set_xlabel("Probability", fontsize=12)
        
        # Add prediction text
        self.prediction_text = self.ax.text(
            0.5, -0.1,
            "Speak to detect emotion...",
            fontsize=14,
            ha='center',
            transform=self.ax.transAxes
        )
        
        # Setup animation
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=100,
            blit=False
        )
    
    def update_plot(self, frame):
        """Update visualization with current probabilities"""
        # Update bar heights
        for bar, prob in zip(self.bars, self.emotion_probs):
            bar.set_width(prob)
        
        # Update prediction text
        if self.current_confidence > 0.3:
            emoji = EMOJI_MAP.get(self.current_emotion, "")
            self.prediction_text.set_text(
                f"Detected: {self.current_emotion} ({self.current_confidence:.2f}) {emoji}"
            )
        else:
            self.prediction_text.set_text("Speak to detect emotion...")
        
        return self.bars + [self.prediction_text]
    
    def audio_stream(self):
        """Stream audio from microphone"""
        try:
            audio = pyaudio.PyAudio()
            
            # Open stream
            stream = audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Audio streaming started")
            
            # Stream audio
            while self.running:
                try:
                    # Read audio chunk
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    
                    # Put chunk in queue
                    self.audio_queue.put(audio_chunk)
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
            
            # Clean up
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
            self.running = False
    
    def recognition_loop(self):
        """Main recognition loop"""
        logger.info("Recognition started")
        
        try:
            while self.running:
                # Get audio chunk from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                    
                    # Update buffer (shift left and add new chunk)
                    shift = len(audio_chunk)
                    self.audio_buffer = np.roll(self.audio_buffer, -shift)
                    self.audio_buffer[-shift:] = audio_chunk
                    
                    # Skip if audio level is too low (likely silence)
                    audio_level = np.abs(self.audio_buffer).mean()
                    if audio_level < 0.01:
                        continue
                    
                    # Convert to tensor
                    waveform = torch.from_numpy(self.audio_buffer).float().to(self.device)
                    
                    # Ensure correct dimensions (add batch dimension if needed)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)  # [1, buffer_frames]
                    
                    # Add channel dimension if needed
                    if waveform.dim() == 2:
                        waveform = waveform.unsqueeze(1)  # [1, 1, buffer_frames]
                    
                    # Normalize audio
                    if waveform.abs().max() > 0:
                        waveform = waveform / waveform.abs().max()
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model(waveform)
                        probs = outputs['emotion_probs'][0].cpu().numpy()
                    
                    # Update probabilities
                    self.emotion_probs = probs
                    
                    # Get predicted emotion
                    emotion_id = np.argmax(probs)
                    self.current_emotion = self.emotion_labels[emotion_id]
                    self.current_confidence = probs[emotion_id]
                    
                    # Print prediction if confidence is high enough
                    if self.current_confidence > 0.3 and not self.visualize:
                        emoji = EMOJI_MAP.get(self.current_emotion, "")
                        print(f"\rEmotion: {self.current_emotion} (confidence: {self.current_confidence:.2f}) {emoji}", end="")
                        sys.stdout.flush()
                    
                except TimeoutError:
                    continue
                
        except Exception as e:
            logger.error(f"Error in recognition loop: {e}")
        
        logger.info("Recognition stopped")
    
    def stop(self):
        """Stop emotion recognition"""
        logger.info("Stopping...")
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1)
        
        if hasattr(self, 'recognition_thread'):
            self.recognition_thread.join(timeout=1)
        
        logger.info("Audio processing stopped")

def main():
    parser = argparse.ArgumentParser(description="Real-time Speech Emotion Recognition")
    parser.add_argument("--model", type=str, default="models/optimal/full/best_model.pt", help="Path to model file")
    parser.add_argument("--simplified", action="store_true", help="Use simplified emotions (4 classes)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=float, default=3.0, help="Audio buffer length in seconds")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file {args.model} not found")
        sys.exit(1)
    
    # Print info
    print("\n" + "="*50)
    print(" Real-time Speech Emotion Recognition ")
    if args.simplified:
        print(" Using simplified emotions (neutral, happy, sad, angry)")
    else:
        print(" Using full emotions (8 categories)")
    print("="*50 + "\n")
    
    # Create and start recognizer
    recognizer = EmotionRecognizer(
        model_path=args.model,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds,
        simplified=args.simplified,
        device=args.device,
        visualize=not args.no_visualize
    )
    
    try:
        print("Initializing audio... Please wait.")
        recognizer.start()
        if not args.no_visualize:
            print("Visualization window opened. Close the window to exit.")
        else:
            print("Press Ctrl+C to exit")
            while recognizer.running:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recognizer.stop()
        print("Done")

if __name__ == "__main__":
    main() 