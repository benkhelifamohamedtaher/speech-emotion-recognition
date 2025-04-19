#!/usr/bin/env python3
"""
Real-time RAVDESS emotion recognition console interface.
This script properly handles tensor dimension issues and uses RAVDESS emotion categories.
"""

import os

def safe_index(value, idx=0, default=0.0):
    '''Safely index a value that might be a scalar or array.'''
    if isinstance(value, (float, int, bool)):
        return value
    try:
        return value[idx]
    except (IndexError, TypeError):
        return default

import sys
import time
import queue
import threading
import logging
import argparse
import numpy as np
import torch
import pyaudio
from collections import deque

# Import the RAVDESS model
from ravdess_model import RAVDESSRecognizer, RAVDESS_EMOTIONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emoji representations for each emotion
EMOTION_EMOJIS = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgust': '🤢',
    'surprised': '😲',
    'error': '❓',
    'uncertain': '❓'
}


class RealTimeEmotionRecognizer:
    """Real-time emotion recognition with audio input and console visualization"""
    def __init__(self, model_path, device='cpu', sample_rate=16000, 
                 chunk_size=1024, buffer_seconds=3, device_index=None,
                 use_simplified_emotions=False):
        """Initialize the real-time recognizer"""
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_seconds = buffer_seconds
        self.buffer_size = int(self.sample_rate * self.buffer_seconds)
        self.device_index = device_index
        self.use_simplified_emotions = use_simplified_emotions
        
        # Create audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.audio_queue = queue.Queue()
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.recognizer = RAVDESSRecognizer(
            model_path=model_path,
            device=device,
            use_simplified_emotions=use_simplified_emotions
        )
        
        # Setup audio
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Show available audio devices
        self._list_audio_devices()
        
        # Processing flags
        self.is_running = False
        self.processing_thread = None
        
        # Display variables
        self.last_emotion = None
        self.update_interval = 0.3  # More frequent updates
        self.last_update_time = 0
    
    def _list_audio_devices(self):
        """List all available audio input devices"""
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
                        
                        # Perform inference - robust handling in recognizer
                        result = self.recognizer.predict(audio_array)
                        
                        # Update display periodically
                        current_time = time.time()
                        if current_time - self.last_update_time >= self.update_interval:
                            emotion = result['emotion']
                            confidence = result['confidence']
                            vad = result['vad']
                            
                            # Only update if voice activity detected and confidence is sufficient
                            if vad > 0.5 and (confidence > 0.25 or emotion == 'uncertain'):
                                self.update_console_display(emotion, confidence, result['probs'])
                                self.last_update_time = current_time
                    
                    except Exception as e:
                        logger.error(f"Error processing audio: {e}")
            
            except queue.Empty:
                # No data available, just continue
                pass
            except Exception as e:
                logger.error(f"Error in audio queue: {e}")
    
    def update_console_display(self, emotion, confidence, all_probs):
        """Update the console display with current emotion probabilities"""
        try:
            # Skip if emotion hasn't changed and confidence is low
            if self.last_emotion == emotion and confidence < 0.6 and emotion != 'uncertain':
                return
            
            # Update last emotion
            self.last_emotion = emotion
            
            # Clear line and print current emotion
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            emoji = EMOTION_EMOJIS.get(emotion, '❓')
            
            # Create a simple bar for the top emotion
            bar_length = 20
            bar = '|' * int(bar_length * confidence) + ' ' * int(bar_length * (1 - confidence))
            
            # Print the top emotion with emphasis
            sys.stdout.write(f"\r{emoji} {emotion.upper()}: {confidence:.2f} [{bar}] {emoji}\n")
            
            # Get emotions based on simplified or full set
            display_emotions = ["neutral", "happy", "sad", "angry"] if self.use_simplified_emotions else RAVDESS_EMOTIONS
            
            # Print other emotions in sorted order
            sorted_indices = np.argsort(-all_probs)  # Sort by descending probability
            for idx in sorted_indices:
                prob = all_probs[idx]
                label = display_emotions[idx] if idx < len(display_emotions) else f"emotion_{idx}"
                if label != emotion:  # Skip the top emotion we already displayed
                    bar_value = int(prob * bar_length)
                    bar = '|' * bar_value + ' ' * (bar_length - bar_value)
                    sys.stdout.write(f"  {label}: {prob:.2f} [{bar}]\n")
            
            # Move cursor back up for next update
            num_emotions = len(display_emotions)
            sys.stdout.write(f"\033[{num_emotions}A")
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
        if self.use_simplified_emotions:
            print("\n" * 4)  # 4 emotions
        else:
            print("\n" * 8)  # 8 emotions
        
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
        num_emotions = 4 if self.use_simplified_emotions else 8
        print("\n" * (num_emotions + 1))
        logger.info("Real-time emotion recognition stopped")


def main():
    parser = argparse.ArgumentParser(description="Real-time RAVDESS Speech Emotion Recognition")
    parser.add_argument("--model", type=str, default="models/ravdess/full/best_model.pt",
                      help="Path to model file")
    parser.add_argument("--simplified", action="store_true", 
                      help="Use simplified emotions (neutral, happy, sad, angry)")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to run on (cpu or cuda)")
    parser.add_argument("--sample_rate", type=int, default=16000,
                      help="Audio sample rate")
    parser.add_argument("--buffer_seconds", type=int, default=3,
                      help="Audio buffer length in seconds")
    parser.add_argument("--device_index", type=int, 
                      help="Audio device index (default: auto-detect)")
    parser.add_argument("--log_file", type=str,
                      help="Log file path (default: console only)")
    
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
║            RAVDESS Edition                  ║
╚════════════════════════════════════════════╝
    
Listening for audio input... Speak to detect emotions.
Press Ctrl+C to exit.
""")
    
    # Print emotion set being used
    if args.simplified:
        print("Using simplified emotions: neutral, happy, sad, angry")
    else:
        print("Using full RAVDESS emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised")
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.warning(f"Model file not found: {args.model}")
        logger.warning("Will use randomly initialized model (poor performance expected)")
    
    # Initialize recognizer
    recognizer = RealTimeEmotionRecognizer(
        model_path=args.model,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds,
        device_index=args.device_index,
        use_simplified_emotions=args.simplified
    )
    
    # Start recognition
    recognizer.start()


if __name__ == "__main__":
    main() 