#!/usr/bin/env python3
"""
Example script demonstrating how to use the advanced training module
for speech emotion recognition.
"""

import sys
import os
import argparse

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_advanced import train_advanced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model using the advanced training script")
    parser.add_argument("--config", type=str, default="src/configs/advanced_training.yaml", 
                        help="Path to the configuration YAML file")
    parser.add_argument("--dataset_root", type=str, 
                        help="Path to the dataset root directory (overrides config)")
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for training (overrides config)")
    parser.add_argument("--epochs", type=int, 
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float, 
                        help="Learning rate (overrides config)")
    parser.add_argument("--fp16", action="store_true", 
                        help="Enable mixed precision training (overrides config)")
    parser.add_argument("--checkpoint_dir", type=str, 
                        help="Directory to save checkpoints (overrides config)")
    
    args = parser.parse_args()
    
    # Prepare overrides dictionary from command line arguments
    overrides = {}
    if args.dataset_root:
        overrides["dataset.dataset_root"] = args.dataset_root
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.learning_rate:
        overrides["training.learning_rate"] = args.learning_rate
    if args.fp16:
        overrides["training.fp16_training"] = True
    if args.checkpoint_dir:
        overrides["logging.checkpoint_dir"] = args.checkpoint_dir
    
    print(f"Starting training with configuration from {args.config}")
    train_advanced(args.config, **overrides)
    print("Training completed!") 