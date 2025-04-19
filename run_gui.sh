#!/bin/bash
# Run the Real-time Speech Emotion Recognition GUI
# This script runs the GUI with the best model (Simplified model with 50.5% accuracy)

# Ensure we're in the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Use system Python to avoid tkinter issues with pyenv
PYTHON_PATH=$(which python3)
if [ -z "$PYTHON_PATH" ]; then
    PYTHON_PATH=$(which python)
fi

if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Could not find Python. Please make sure Python is installed."
    exit 1
fi

echo "Using Python at: $PYTHON_PATH"

# Check if required packages are installed
CHECK_PACKAGES() {
    REQUIRED_PACKAGES=("numpy" "torch" "matplotlib" "pyaudio")
    MISSING_PACKAGES=()
    
    for PACKAGE in "${REQUIRED_PACKAGES[@]}"; do
        if ! $PYTHON_PATH -c "import $PACKAGE" &> /dev/null; then
            MISSING_PACKAGES+=("$PACKAGE")
        fi
    done
    
    # Check for tkinter
    if ! $PYTHON_PATH -c "import tkinter" &> /dev/null; then
        echo "Warning: tkinter is not available. GUI may not work properly."
        echo "On macOS, try installing it with: brew install python-tk"
        echo "On Linux, try: sudo apt-get install python3-tk"
        echo "Continuing anyway..."
    fi
    
    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        echo "Error: Missing required packages: ${MISSING_PACKAGES[*]}"
        echo "Please install them using pip: $PYTHON_PATH -m pip install ${MISSING_PACKAGES[*]}"
        return 1
    fi
    
    return 0
}

# Check for Python and required packages
echo "Checking required packages..."
CHECK_PACKAGES || exit 1

# Run the GUI
echo "Starting Real-time Speech Emotion Recognition GUI..."
$PYTHON_PATH src/run_emotion_gui.py "$@" 