#!/usr/bin/env python3
"""
Example script for using the advanced training pipeline for speech emotion recognition.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.train_advanced import train_advanced

if __name__ == "__main__":
    # Example of using the training script with a custom config
    config_path = "src/configs/advanced_training.yaml"
    
    # You can override any config parameters directly
    train_advanced(
        config_path=config_path,
        dataset_root="path/to/your/dataset",
        batch_size=8,
        epochs=30,
        learning_rate=2e-4,
        fp16_training=True,
        augmentation_strength=0.7,
    )
    
    print("Training completed successfully!") 