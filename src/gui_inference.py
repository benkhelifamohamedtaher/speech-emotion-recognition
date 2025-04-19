#!/usr/bin/env python3
"""
GUI-based Real-time Speech Emotion Recognition
This module provides a graphical user interface for real-time speech emotion recognition.
"""

import sys

def safe_index(value, idx=0, default=0.0):
    '''Safely index a value that might be a scalar or array.'''
    if isinstance(value, (float, int, bool)):
        return value
    try:
        return value[idx]
    except (IndexError, TypeError):
        return default

import os
import torch
import numpy as np
import pyaudio
import threading
import queue
import time
from pathlib import Path
import yaml
import argparse
from collections import deque
import logging

# For GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Custom imports
from model import SpeechEmotionRecognitionModel
from model_enhanced import EnhancedSpeechEmotionRecognitionModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Emotion labels
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
EMOTION_COLORS = ['#808080', '#FFD700', '#1E90FF', '#FF4500', '#800080', '#006400', '#FF69B4']

class EmotionRecognitionGUI:
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("Real-time Speech Emotion Recognition")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self.create_control_panel()
        
        # Create visualization panel
        self.create_visualization_panel()
        
        # Audio processing variables
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.buffer_seconds = 3
        self.buffer_size = self.sample_rate * self.buffer_seconds
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.audio_queue = queue.Queue()
        
        # Model variables
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.is_running = False
        self.processing_thread = None
        
        # Emotion history for smoothing
        self.emotion_history = deque(maxlen=10)
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        
        # Audio stream
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Update status
        self.status_var.set("Ready. Load a model to start.")
        
        # Setup periodic UI update
        self.update_ui()
    
    def create_control_panel(self):
        """Create the control panel with buttons and settings"""
        # Control frame
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        model_btn = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        model_btn.grid(row=0, column=2, padx=5)
        
        # Config selection
        config_frame = ttk.Frame(control_frame)
        config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Config Path:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.config_path_var = tk.StringVar()
        config_entry = ttk.Entry(config_frame, textvariable=self.config_path_var, width=50)
        config_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        config_btn = ttk.Button(config_frame, text="Browse", command=self.browse_config)
        config_btn.grid(row=0, column=2, padx=5)
        
        # Model options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.enhanced_var = tk.BooleanVar(value=False)
        enhanced_check = ttk.Checkbutton(options_frame, text="Use Enhanced Model", variable=self.enhanced_var)
        enhanced_check.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.load_btn = ttk.Button(button_frame, text="Load Model", command=self.load_model)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(button_frame, text="Start Recognition", command=self.start_recognition, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)
    
    def create_visualization_panel(self):
        """Create the visualization panel with matplotlib figure"""
        viz_frame = ttk.LabelFrame(self.main_frame, text="Emotion Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.bars = self.ax.bar(EMOTION_LABELS, np.zeros(len(EMOTION_LABELS)), color=EMOTION_COLORS)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Detected Emotions')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Current emotion indicator
        emotion_frame = ttk.Frame(viz_frame)
        emotion_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(emotion_frame, text="Current Emotion:").pack(side=tk.LEFT, padx=5)
        
        self.emotion_var = tk.StringVar(value="N/A")
        self.emotion_label = ttk.Label(emotion_frame, textvariable=self.emotion_var, font=("Helvetica", 14, "bold"))
        self.emotion_label.pack(side=tk.LEFT, padx=5)
    
    def browse_model(self):
        """Open file dialog to select model path"""
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if filepath:
            self.model_path_var.set(filepath)
    
    def browse_config(self):
        """Open file dialog to select configuration file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")]
        )
        if filepath:
            self.config_path_var.set(filepath)
    
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_path_var.get()
        config_path = self.config_path_var.get()
        use_enhanced = self.enhanced_var.get()
        
        if not model_path:
            messagebox.showerror("Error", "Please select a model file.")
            return
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Load configuration if provided
            if config_path:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                model_name = self.config.get('model', {}).get('model_name', 'facebook/wav2vec2-base-960h')
            else:
                model_name = 'facebook/wav2vec2-base-960h'
                self.config = {'model': {'model_name': model_name}}
            
            # Initialize model
            if use_enhanced:
                self.model = EnhancedSpeechEmotionRecognitionModel(model_name=model_name)
            else:
                self.model = SpeechEmotionRecognitionModel(model_name=model_name)
            
            # Load model checkpoint
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.to(self.device)
            self.model.eval()
            
            self.status_var.set(f"Model loaded successfully. Device: {self.device}")
            self.start_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            logger.error(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
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
                self.status_var.set(f"Error: {str(e)}")
    
    def start_recognition(self):
        """Start real-time emotion recognition"""
        if self.is_running:
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
        
        self.status_var.set("Recognition started. Speak now...")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.DISABLED)
    
    def stop_recognition(self):
        """Stop real-time emotion recognition"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
            self.processing_thread = None
        
        # Reset emotion probabilities
        self.emotion_probs = np.zeros(len(EMOTION_LABELS))
        self.emotion_history.clear()
        
        self.status_var.set("Recognition stopped.")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.NORMAL)
        
        # Update UI
        self.update_visualization()
    
    def update_visualization(self):
        """Update the visualization with current emotion probabilities"""
        if not hasattr(self, 'bars'):
            return
            
        # Update bar heights
        for bar, prob in zip(self.bars, self.emotion_probs):
            bar.set_height(prob)
        
        # Highlight the predicted emotion
        predicted_idx = np.argmax(self.emotion_probs)
        
        # Update current emotion text
        if np.max(self.emotion_probs) > 0:
            self.emotion_var.set(EMOTION_LABELS[predicted_idx].capitalize())
        else:
            self.emotion_var.set("N/A")
        
        # Redraw canvas
        self.canvas.draw()
    
    def update_ui(self):
        """Periodically update the UI"""
        self.update_visualization()
        self.root.after(100, self.update_ui)  # Schedule next update
    
    def on_closing(self):
        """Handle window closing event"""
        if self.is_running:
            self.stop_recognition()
        
        self.pyaudio_instance.terminate()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="GUI for Real-time Speech Emotion Recognition")
    parser.add_argument("--model", type=str, default="", help="Path to the trained model")
    parser.add_argument("--config", type=str, default="", help="Path to the configuration file")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced model")
    
    args = parser.parse_args()
    
    # Create the root window
    root = tk.Tk()
    app = EmotionRecognitionGUI(root)
    
    # Set command line arguments if provided
    if args.model:
        app.model_path_var.set(args.model)
    if args.config:
        app.config_path_var.set(args.config)
    if args.enhanced:
        app.enhanced_var.set(True)
    
    # Set closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main() 