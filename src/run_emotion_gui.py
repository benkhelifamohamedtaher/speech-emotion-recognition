#!/usr/bin/env python3
"""
Launcher script for the Real-time Speech Emotion Recognition GUI
"""

import os
import sys
import subprocess
import argparse

def find_best_model():
    """Find the best model in the project directory"""
    model_paths = [
        "models/ravdess_simple/best_model.pt",
        "models/simplified/best_model.pt",
        "models/ravdess_high_accuracy/best_model.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model: {path}")
            return path
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Launch Real-time Speech Emotion Recognition GUI")
    parser.add_argument("--simplified", action="store_true", default=True,
                        help="Use simplified emotion set (4 emotions)")
    parser.add_argument("--advanced", action="store_true", default=False,
                        help="Use advanced model architecture")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained model (optional)")
    
    args = parser.parse_args()
    
    # Find best model if not specified
    model_path = args.model
    if not model_path:
        model_path = find_best_model()
        if not model_path:
            print("Error: Could not find a model file.")
            print("Please specify a model path using --model")
            return 1
    
    # Get the current script's absolute path to reference emotion_recognition_gui.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gui_script = os.path.join(current_dir, "emotion_recognition_gui.py")
    
    # Construct command to run the GUI directly
    cmd = [
        sys.executable,  # Use same Python that's running this script
        gui_script,
        "--model", model_path
    ]
    
    if args.simplified:
        cmd.append("--simplified")
    
    if args.advanced:
        cmd.append("--advanced")
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the GUI
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 