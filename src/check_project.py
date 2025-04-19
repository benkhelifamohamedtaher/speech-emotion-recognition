#!/usr/bin/env python3
"""
System check utility for the Real-Time Speech Emotion Recognition project.
Verifies that all required components and dependencies are correctly installed.
"""

import os
import sys
import platform
import importlib
from pathlib import Path
import subprocess
import warnings

# Terminal colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{BOLD}=== {text} ==={ENDC}\n")

def print_success(text):
    """Print a success message."""
    print(f"{GREEN}✓ {text}{ENDC}")

def print_warning(text):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {text}{ENDC}")

def print_error(text):
    """Print an error message."""
    print(f"{RED}✗ {text}{ENDC}")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    version = platform.python_version()
    print(f"Detected Python version: {version}")
    
    major, minor, _ = map(int, version.split('.'))
    if major < 3 or (major == 3 and minor < 8):
        print_error("Python 3.8 or newer is required.")
        return False
    else:
        print_success("Python version is compatible.")
        return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Required Dependencies")
    required_packages = [
        "torch", "torchaudio", 
        "numpy", "matplotlib", 
        "librosa", "pyaudio",
        "pyyaml", "tqdm"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print_success(f"{package} is installed.")
        except ImportError:
            print_error(f"{package} is NOT installed. Run 'pip install {package}'")
            all_installed = False
    
    return all_installed

def check_audio_device():
    """Check if an audio input device is available."""
    print_header("Checking Audio Device")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Check for input devices
        input_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
        
        p.terminate()
        
        if input_devices:
            print_success(f"Found {len(input_devices)} audio input device(s):")
            for idx, name in input_devices:
                print(f"  - Device {idx}: {name}")
            return True
        else:
            print_error("No audio input devices found. A microphone is required for real-time inference.")
            return False
    except Exception as e:
        print_error(f"Error checking audio devices: {str(e)}")
        return False

def check_project_structure():
    """Check if the project structure is correct."""
    print_header("Checking Project Structure")
    
    # Get the project root (parent of the directory containing this script)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Define expected directories and files
    expected_dirs = [
        project_root / "src",
        project_root / "models",
        project_root / "dataset"
    ]
    
    expected_files = [
        script_dir / "model.py",
        script_dir / "model_enhanced.py",
        script_dir / "inference.py",
        script_dir / "train_fixed.py",
        script_dir / "run_interactive_demo.sh"
    ]
    
    # Check directories
    all_found = True
    print("Checking directories:")
    for directory in expected_dirs:
        if directory.exists() and directory.is_dir():
            print_success(f"{directory.relative_to(project_root)} exists.")
        else:
            print_error(f"{directory.relative_to(project_root)} is missing.")
            all_found = False
    
    # Check files
    print("\nChecking key files:")
    for file in expected_files:
        if file.exists() and file.is_file():
            print_success(f"{file.relative_to(project_root)} exists.")
        else:
            print_error(f"{file.relative_to(project_root)} is missing.")
            all_found = False
    
    return all_found

def check_gpu():
    """Check if CUDA is available for GPU acceleration."""
    print_header("Checking GPU Acceleration")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print_success(f"CUDA is available! Found {device_count} device(s).")
            print(f"Primary device: {device_name}")
            return True
        else:
            print_warning("CUDA is not available. Using CPU only.")
            print("Note: Training will be significantly slower without GPU acceleration.")
            return False
    except Exception as e:
        print_error(f"Error checking GPU: {str(e)}")
        return False

def check_models():
    """Check if pre-trained models are available."""
    print_header("Checking Pre-trained Models")
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        print_warning("Models directory not found.")
        return False
    
    expected_models = [
        models_dir / "base_model" / "model.pt",
        models_dir / "enhanced_model" / "model.pt",
        models_dir / "ensemble_model" / "model.pt"
    ]
    
    available_models = []
    for model_path in expected_models:
        if model_path.exists():
            available_models.append(model_path.relative_to(project_root))
    
    if available_models:
        print_success(f"Found {len(available_models)} pre-trained model(s):")
        for model in available_models:
            print(f"  - {model}")
        return True
    else:
        print_warning("No pre-trained models found. You need to train models first.")
        return False

def main():
    """Run all system checks."""
    print(f"{BOLD}Real-Time Speech Emotion Recognition - System Check{ENDC}")
    print("-" * 50)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project structure", check_project_structure),
        ("Audio device", check_audio_device),
        ("GPU acceleration", check_gpu),
        ("Pre-trained models", check_models)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Print summary
    print_header("Summary")
    for name, result in results.items():
        if result:
            print_success(f"{name}: OK")
        elif name == "GPU acceleration" or name == "Pre-trained models":
            print_warning(f"{name}: Warning")
        else:
            print_error(f"{name}: Failed")
    
    # Critical checks that must pass
    critical = ["Python version", "Dependencies", "Project structure"]
    critical_failed = any(not results[check] for check in critical)
    
    if critical_failed:
        print(f"\n{RED}{BOLD}Critical checks failed. Please fix the issues above before proceeding.{ENDC}")
        sys.exit(1)
    elif not all(results.values()):
        print(f"\n{YELLOW}{BOLD}Some non-critical checks failed. The system may work but with limited functionality.{ENDC}")
        sys.exit(0)
    else:
        print(f"\n{GREEN}{BOLD}All checks passed! The system is ready to use.{ENDC}")
        sys.exit(0)

if __name__ == "__main__":
    main() 