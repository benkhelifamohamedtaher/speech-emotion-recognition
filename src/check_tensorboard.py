#!/usr/bin/env python3
"""
TensorBoard Verification Script
Checks that TensorBoard is properly installed and can be used with PyTorch
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Verify TensorBoard is working")
    parser.add_argument('--log_dir', type=str, default='../tensorboard_test',
                      help='Directory to save the test logs')
    args = parser.parse_args()
    
    # Setup colored terminal output
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    
    print(f"{BLUE}TensorBoard Verification Script{ENDC}")
    print("-" * 40)
    
    # Check if TensorBoard is available
    if not TENSORBOARD_AVAILABLE:
        print(f"{RED}ERROR: TensorBoard is not available. Please install it with:{ENDC}")
        print("pip install tensorboard")
        sys.exit(1)
    
    print(f"{GREEN}✓ TensorBoard is available{ENDC}")
    
    # Create log directory
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"{BLUE}Writing test logs to {log_dir}{ENDC}")
    
    # Create a SummaryWriter instance
    writer = SummaryWriter(log_dir=log_dir)
    
    # Generate some dummy data
    print(f"{YELLOW}Generating test data...{ENDC}")
    for i in range(100):
        # Simulate training metrics
        train_loss = 1 - 0.99 * (1 - np.exp(-0.1 * i))
        val_loss = 1 - 0.95 * (1 - np.exp(-0.1 * i)) + 0.05 * np.sin(i * 0.4)
        train_acc = 1 - train_loss
        val_acc = 1 - val_loss + 0.05 * np.cos(i * 0.4)
        
        # Log scalars
        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/validation', val_loss, i)
        writer.add_scalar('Accuracy/train', train_acc, i)
        writer.add_scalar('Accuracy/validation', val_acc, i)
        
        # Log learning rate
        lr = 0.001 * (0.9 ** (i // 10))
        writer.add_scalar('Learning_rate', lr, i)
    
    # Create and log a simple model
    print(f"{YELLOW}Creating test model graph...{ENDC}")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = torch.nn.Conv1d(1, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv1d(16, 32, 3, padding=1)
            self.pool = torch.nn.MaxPool1d(2)
            self.fc1 = torch.nn.Linear(32 * 25, 64)
            self.fc2 = torch.nn.Linear(64, 8)  # 8 emotions
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 32 * 25)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    dummy_input = torch.rand(1, 1, 50)
    
    # Add model graph to TensorBoard
    writer.add_graph(model, dummy_input)
    
    # Add attention visualization example
    print(f"{YELLOW}Creating attention visualization...{ENDC}")
    attention_weights = torch.softmax(torch.randn(8, 10), dim=1)  # 8 emotions, 10 time steps
    writer.add_image('Attention_weights', 
                    attention_weights.unsqueeze(0), 
                    dataformats='CHW')
    
    # Generate emotion probability distributions
    print(f"{YELLOW}Creating emotion probability distributions...{ENDC}")
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    for i in range(5):
        # Simulate different emotion distributions
        probs = torch.softmax(torch.randn(8) * (i+1), dim=0)
        writer.add_scalars('Emotion_Probabilities', 
                         {emotion: prob.item() for emotion, prob in zip(emotions, probs)}, 
                         i)
    
    # Close the writer
    writer.close()
    
    print(f"{GREEN}✓ TensorBoard test completed successfully{ENDC}")
    print(f"{BLUE}To view the results, run:{ENDC}")
    print(f"tensorboard --logdir={args.log_dir}")
    print(f"{BLUE}Then open your browser at http://localhost:6006{ENDC}")

if __name__ == "__main__":
    main() 