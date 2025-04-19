#!/usr/bin/env python3
"""
Interactive Speech Emotion Recognition Demo
This script provides a simple terminal-based real-time speech emotion recognition demo.
It uses a pre-trained model to recognize emotions from microphone input.
"""

import os
import sys
import time
import signal
import argparse
import numpy as np
import torch
import pyaudio
import threading
from collections import deque
from pathlib import Path

# Add the src directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import SpeechEmotionRecognitionModel
from src.model_enhanced import EnhancedSpeechEmotionRecognitionModel

# Define emotion labels
EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SIMPLIFIED_EMOTION_LABELS = ["neutral", "happy", "sad", "angry"]

class InteractiveEmotionRecognizer:
    def __init__(self, model_path, config_file=None, use_enhanced_model=False, 
                 device="cpu", sample_rate=16000, chunk_size=1024, buffer_seconds=3, 
                 simplified_emotions=False):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(sample_rate * buffer_seconds)
        self.running = False
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Fill buffer with zeros initially
        self.audio_buffer.extend(np.zeros(self.buffer_size))
        
        # Load the model
        if use_enhanced_model:
            self.model = EnhancedSpeechEmotionRecognitionModel(
                num_emotions=4 if simplified_emotions else 8
            )
        else:
            self.model = SpeechEmotionRecognitionModel(
                num_emotions=4 if simplified_emotions else 8
            )
        
        # Load model weights
        if model_path:
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)
        
        self.model.to(device)
        self.model.eval()
        
        # Set emotion labels
        self.emotion_labels = SIMPLIFIED_EMOTION_LABELS if simplified_emotions else EMOTION_LABELS
        
        # Audio setup
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.processing_thread = None
        
        # Last prediction results
        self.last_emotion = None
        self.last_confidence = 0.0
        self.display_threshold = 0.3  # Minimum confidence to display emotion
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        last_print_time = 0
        print_interval = 0.5  # Print every 0.5 seconds
        
        while self.running:
            try:
                # Get audio data from buffer
                audio_data = np.array(list(self.audio_buffer))
                
                # Skip if we don't have enough data
                if len(audio_data) < self.buffer_size:
                    time.sleep(0.1)
                    continue
                
                # Convert to tensor
                waveform = torch.FloatTensor(audio_data).to(self.device)
                
                # Reshape if needed
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # Add batch dimension
                
                # Process through model
                with torch.no_grad():
                    # For the enhanced model, input shape might be different
                    if isinstance(self.model, EnhancedSpeechEmotionRecognitionModel):
                        emotion_probs, _ = self.model(waveform)
                    else:
                        emotion_probs, _ = self.model(waveform)
                    
                    if emotion_probs.dim() > 1:
                        emotion_probs = emotion_probs.squeeze(0)
                    
                    # Get the prediction
                    emotion_idx = torch.argmax(emotion_probs).item()
                    confidence = emotion_probs[emotion_idx].item()
                    
                    # Update last prediction
                    self.last_emotion = self.emotion_labels[emotion_idx]
                    self.last_confidence = confidence
                    
                    # Print results at regular intervals
                    current_time = time.time()
                    if current_time - last_print_time > print_interval and confidence > self.display_threshold:
                        print(f"\rDetected emotion: {self.last_emotion.upper()} (confidence: {confidence:.2f})", end="")
                        last_print_time = current_time
                
                # Sleep to avoid using too much CPU
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError processing audio: {e}")
                time.sleep(1)
    
    def start(self):
        self.running = True
        
        # Start audio stream
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        print("\nListening... Speak to detect emotions (Ctrl+C to exit)")
        print("Detected emotions will appear below:")
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        self.pa.terminate()
        print("\nStopped emotion recognition")

def signal_handler(sig, frame):
    print("\nExiting...")
    if recognizer:
        recognizer.stop()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Speech Emotion Recognition Demo")
    parser.add_argument("--model", type=str, default="models/fixed_simple/best_model.pt", 
                        help="Path to the trained model")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to model configuration file")
    parser.add_argument("--enhanced", action="store_true", 
                        help="Use enhanced model architecture")
    parser.add_argument("--simplified", action="store_true", 
                        help="Use simplified emotion set (4 emotions instead of 8)")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--sample-rate", type=int, default=16000, 
                        help="Audio sample rate")
    parser.add_argument("--buffer-seconds", type=float, default=3.0, 
                        help="Seconds of audio to buffer for analysis")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Available models:")
        for model_dir in os.listdir("models"):
            model_path = os.path.join("models", model_dir, "best_model.pt")
            if os.path.exists(model_path):
                print(f"  {model_path}")
        sys.exit(1)
    
    # Use CUDA if available and requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"
    
    # Initialize and start the recognizer
    recognizer = InteractiveEmotionRecognizer(
        model_path=args.model,
        config_file=args.config,
        use_enhanced_model=args.enhanced,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds,
        simplified_emotions=args.simplified
    )
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start recognition
    recognizer.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        recognizer.stop() 