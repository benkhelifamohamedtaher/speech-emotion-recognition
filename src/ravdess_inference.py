#!/usr/bin/env python3
"""
Real-time RAVDESS Emotion Recognition Inference
Captures audio from microphone and predicts emotions in real-time with visualization
"""

import os
import sys
import json
import time
import argparse
import threading
import queue
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
import torch.nn.functional as F
import pyaudio
import librosa

# Import our model
from ravdess_model import AdvancedSpeechEmotionRecognizer
from ravdess_dataset import RAVDESSDataset


class RealTimeEmotionRecognizer:
    """
    Real-time emotion recognition from microphone input
    """
    def __init__(self, model_path, device='cpu', sample_rate=16000,
                 chunk_size=1024, buffer_seconds=3.0, 
                 update_interval=0.1, energy_threshold=0.01):
        """
        Initialize the real-time emotion recognizer
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
            sample_rate: Audio sample rate
            chunk_size: Number of audio samples per chunk
            buffer_seconds: Size of audio buffer in seconds
            update_interval: How often to update predictions (in seconds)
            energy_threshold: Minimum energy level to process audio
        """
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_samples = int(buffer_seconds * sample_rate)
        self.update_interval = update_interval
        self.energy_threshold = energy_threshold
        
        # Load the model
        self.model, self.config = self._load_model(model_path)
        
        # Get emotion mapping
        self.emotion_mapping = RAVDESSDataset.get_emotion_mapping()
        self.emotion_list = [self.emotion_mapping[i] for i in range(len(self.emotion_mapping))]
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize audio buffer
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        
        # Initialize variables for real-time processing
        self.running = False
        self.stream = None
        self.thread = None
        self.audio_queue = queue.Queue()
        
        # Initialize emotion probabilities
        self.emotion_probs = np.zeros(len(self.emotion_list), dtype=np.float32)
        self.emotion_lock = threading.Lock()
        
        # Voice activity detection
        self.vad_history = np.zeros(10, dtype=np.float32)
        self.vad_index = 0
        self.is_speech = False
        
        # Current top emotion
        self.current_emotion = "neutral"
        self.emotion_hold_frames = 0
        self.emotion_min_hold = 3  # Minimum frames to hold an emotion
    
    def _load_model(self, model_path):
        """
        Load the trained model from checkpoint
        """
        print(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get model arguments
        args = checkpoint['args'] if 'args' in checkpoint else {}
        
        # Get config file if it exists
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Combine config and args
        combined_config = {**config, **args}
        
        # Get number of classes
        num_classes = combined_config.get('num_classes', 8)
        
        # Get other model parameters
        wav2vec_model = combined_config.get('wav2vec_model', 'facebook/wav2vec2-base')
        context_layers = combined_config.get('context_layers', 2)
        attention_heads = combined_config.get('attention_heads', 4)
        dropout_rate = combined_config.get('dropout_rate', 0.3)
        use_gender_branch = combined_config.get('use_gender_branch', True)
        use_spectrogram_branch = combined_config.get('use_spectrogram_branch', True)
        
        # Create model
        model = AdvancedSpeechEmotionRecognizer(
            num_emotions=num_classes,
            wav2vec_model_name=wav2vec_model,
            context_layers=context_layers,
            attention_heads=attention_heads,
            dropout_rate=dropout_rate,
            use_gender_branch=use_gender_branch,
            use_spectrogram_branch=use_spectrogram_branch
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model'])
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        return model, combined_config
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for PyAudio stream
        """
        if self.running:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            
            # Normalize audio data
            audio_data = audio_data / 32768.0
            
            # Add to queue
            self.audio_queue.put(audio_data)
        
        # Return empty data and continue flag
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """
        Process audio data in a separate thread
        """
        last_update_time = time.time()
        
        while self.running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Update buffer
                new_len = len(audio_data)
                self.audio_buffer = np.roll(self.audio_buffer, -new_len)
                self.audio_buffer[-new_len:] = audio_data
                
                # Check if enough time has passed since last update
                current_time = time.time()
                if current_time - last_update_time >= self.update_interval:
                    # Calculate energy
                    energy = np.mean(np.abs(self.audio_buffer))
                    
                    # Update VAD history
                    self.vad_history[self.vad_index] = energy
                    self.vad_index = (self.vad_index + 1) % len(self.vad_history)
                    
                    # Determine if speech is present
                    avg_energy = np.mean(self.vad_history)
                    self.is_speech = avg_energy > self.energy_threshold
                    
                    # Only process if speech is detected
                    if self.is_speech:
                        # Preprocess audio buffer
                        processed_audio = self._preprocess_audio(self.audio_buffer)
                        
                        # Predict emotion
                        with torch.no_grad():
                            waveform = torch.from_numpy(processed_audio).to(self.device)
                            outputs = self.model(waveform)
                            emotion_logits = outputs['emotion_logits']
                            emotion_probs = F.softmax(emotion_logits, dim=1)[0].cpu().numpy()
                            emotion_pred = np.argmax(emotion_probs)
                        
                        # Update emotion probabilities with lock
                        with self.emotion_lock:
                            # Smooth probabilities (70% previous, 30% new)
                            alpha = 0.3
                            self.emotion_probs = (1 - alpha) * self.emotion_probs + alpha * emotion_probs
                            
                            # Get current top emotion
                            top_emotion_idx = np.argmax(self.emotion_probs)
                            top_emotion = self.emotion_mapping[top_emotion_idx]
                            
                            # Update current emotion with hold logic to prevent rapid switching
                            if top_emotion == self.current_emotion:
                                self.emotion_hold_frames += 1
                            else:
                                if self.emotion_hold_frames >= self.emotion_min_hold:
                                    self.current_emotion = top_emotion
                                    self.emotion_hold_frames = 0
                                else:
                                    self.emotion_hold_frames += 1
                    
                    # Update last update time
                    last_update_time = current_time
                
                # Mark as done
                self.audio_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue
    
    def _preprocess_audio(self, audio_data):
        """
        Preprocess audio data for inference
        """
        # Apply preemphasis
        preemphasis_coef = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - preemphasis_coef * audio_data[:-1])
        
        # Make sure audio is in the range [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Reshape for model input: [batch_size, channels, samples]
        processed_audio = audio_data.reshape(1, 1, -1)
        
        return processed_audio
    
    def start(self):
        """
        Start real-time emotion recognition
        """
        if self.running:
            print("Already running!")
            return
        
        # Set running flag
        self.running = True
        
        # Start audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()
        
        print("Real-time emotion recognition started.")
    
    def stop(self):
        """
        Stop real-time emotion recognition
        """
        if not self.running:
            print("Not running!")
            return
        
        # Clear running flag
        self.running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        print("Real-time emotion recognition stopped.")
    
    def get_emotion_probs(self):
        """
        Get current emotion probabilities
        """
        with self.emotion_lock:
            return self.emotion_probs.copy(), self.current_emotion, self.is_speech
    
    def close(self):
        """
        Close the recognizer
        """
        self.stop()
        self.audio.terminate()


def visualize_emotion_probs(recognizer, interval=50):
    """
    Visualize emotion probabilities in real-time
    """
    # Get emotion list
    emotion_list = recognizer.emotion_list
    
    # Create figure and axes
    plt.ion()
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Real-Time Emotion Recognition')
    
    # Create bar plot
    y_pos = np.arange(len(emotion_list))
    bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(emotion_list)))
    bars = axes[0].bar(y_pos, np.zeros(len(emotion_list)), align='center', alpha=0.7, color=bar_colors)
    axes[0].set_xticks(y_pos)
    axes[0].set_xticklabels(emotion_list, rotation=45)
    axes[0].set_ylabel('Probability')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Emotion Probabilities')
    
    # Create text annotation for current emotion
    emotion_text = axes[0].text(0.5, 0.90, "Current emotion: neutral", transform=axes[0].transAxes,
                              horizontalalignment='center', fontsize=14, 
                              bbox=dict(facecolor='white', alpha=0.8))
    
    # Create speech detection indicator
    speech_indicator = plt.Rectangle((0.05, 0.05), 0.1, 0.1, color='red')
    axes[0].add_patch(speech_indicator)
    speech_text = axes[0].text(0.11, 0.10, "No Speech", transform=axes[0].transAxes,
                             fontsize=10, verticalalignment='center')
    
    # Create time-series plot
    line_colors = bar_colors
    lines = []
    
    # Time data
    time_steps = 100
    time_data = np.zeros((time_steps, len(emotion_list)))
    x = np.arange(time_steps)
    
    for i, emotion in enumerate(emotion_list):
        line, = axes[1].plot(x, time_data[:, i], label=emotion, color=line_colors[i])
        lines.append(line)
    
    axes[1].set_ylim(0, 1)
    axes[1].set_xlim(0, time_steps - 1)
    axes[1].set_title('Emotion Probabilities Over Time')
    axes[1].set_ylabel('Probability')
    axes[1].set_xlabel('Time')
    axes[1].legend(loc='upper left')
    
    # Update function for animation
    def update_plot(_):
        nonlocal time_data
        
        # Get current emotion probabilities
        probs, current_emotion, is_speech = recognizer.get_emotion_probs()
        
        # Update bar heights
        for i, bar in enumerate(bars):
            bar.set_height(probs[i])
        
        # Update current emotion text
        emotion_text.set_text(f"Current emotion: {current_emotion}")
        
        # Update speech detection indicator
        if is_speech:
            speech_indicator.set_color('green')
            speech_text.set_text("Speech Detected")
        else:
            speech_indicator.set_color('red')
            speech_text.set_text("No Speech")
        
        # Update time-series data
        time_data = np.roll(time_data, -1, axis=0)
        time_data[-1, :] = probs
        
        # Update lines
        for i, line in enumerate(lines):
            line.set_ydata(time_data[:, i])
        
        return bars + lines + [emotion_text, speech_indicator, speech_text]
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)
    plt.tight_layout()
    plt.show()
    
    return ani


def console_mode(recognizer):
    """
    Run in console mode without visualization
    """
    try:
        print("Starting real-time emotion recognition...")
        print("Press Ctrl+C to stop")
        
        # Start recognizer
        recognizer.start()
        
        # Main loop
        while True:
            # Get current emotion probabilities
            _, current_emotion, is_speech = recognizer.get_emotion_probs()
            
            # Only print if speech is detected
            if is_speech:
                print(f"Current emotion: {current_emotion}")
            
            # Sleep for a short time
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop and close recognizer
        recognizer.stop()
        recognizer.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Real-time RAVDESS emotion recognition")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu/cuda)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--buffer_seconds', type=float, default=3.0,
                        help='Audio buffer size in seconds')
    parser.add_argument('--update_interval', type=float, default=0.1,
                        help='How often to update predictions (in seconds)')
    parser.add_argument('--energy_threshold', type=float, default=0.01,
                        help='Energy threshold for voice activity detection')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Run in console mode without visualization')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Create recognizer
    recognizer = RealTimeEmotionRecognizer(
        model_path=args.model_path,
        device=args.device,
        sample_rate=args.sample_rate,
        buffer_seconds=args.buffer_seconds,
        update_interval=args.update_interval,
        energy_threshold=args.energy_threshold
    )
    
    try:
        # Run in appropriate mode
        if args.no_visualize:
            console_mode(recognizer)
        else:
            # Start recognizer
            recognizer.start()
            
            # Visualize emotion probabilities
            ani = visualize_emotion_probs(recognizer)
            plt.show()
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        recognizer.stop()
        recognizer.close()


if __name__ == "__main__":
    main() 