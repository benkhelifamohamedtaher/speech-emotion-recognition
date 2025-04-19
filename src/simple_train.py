#!/usr/bin/env python3
"""
Simplified training script for Speech Emotion Recognition
Avoids multiprocessing and other complex components that might cause issues
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, f1_score

# Import model and dataset
from model_enhanced import SpeechEmotionRecognitionModelEnhanced
from data_utils import EmotionSpeechDataset


def train_simple(args):
    # Set up device and seed
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets directly (no DataLoader with multiple workers)
    train_dataset = EmotionSpeechDataset(
        root_dir=args.dataset_root,
        split='train',
        target_sr=args.sample_rate,
        transform=None,  # No augmentation to avoid pickling issues
        max_length=args.max_length
    )
    
    test_dataset = EmotionSpeechDataset(
        root_dir=args.dataset_root,
        split='test',
        target_sr=args.sample_rate,
        transform=None,
        max_length=args.max_length
    )
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(test_dataset)} samples")
    
    # Create model
    model = SpeechEmotionRecognitionModelEnhanced(
        num_emotions=4, 
        freeze_encoder=args.freeze_encoder,
        model_name=args.model_name
    )
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        # Create simple manual batches
        num_samples = len(train_dataset)
        indices = torch.randperm(num_samples).tolist()
        
        for start_idx in tqdm(range(0, num_samples, args.batch_size), desc="Training"):
            # Get batch indices
            batch_indices = indices[start_idx:start_idx + args.batch_size]
            
            # Skip the last batch if it's too small (less than 2 samples)
            if len(batch_indices) < 2:
                continue
                
            # Get batch data
            batch_waveforms = []
            batch_labels = []
            
            for idx in batch_indices:
                sample = train_dataset[idx]
                batch_waveforms.append(sample['waveform'])
                batch_labels.append(sample['label'])
            
            # Stack tensors
            waveforms = torch.stack(batch_waveforms).to(device)
            labels = torch.tensor(batch_labels).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(waveforms)
            loss = criterion(outputs['emotion_logits'], labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            train_preds.extend(outputs['emotion_probs'].argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= (num_samples // args.batch_size)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            # Create simple manual validation batches
            num_val_samples = len(test_dataset)
            
            for start_idx in tqdm(range(0, num_val_samples, args.batch_size), desc="Validation"):
                # Get batch indices
                end_idx = min(start_idx + args.batch_size, num_val_samples)
                batch_indices = list(range(start_idx, end_idx))
                
                # Skip the last batch if it's too small (less than 2 samples)
                if len(batch_indices) < 2:
                    continue
                
                # Get batch data
                batch_waveforms = []
                batch_labels = []
                
                for idx in batch_indices:
                    sample = test_dataset[idx]
                    batch_waveforms.append(sample['waveform'])
                    batch_labels.append(sample['label'])
                
                # Stack tensors
                waveforms = torch.stack(batch_waveforms).to(device)
                labels = torch.tensor(batch_labels).to(device)
                
                # Forward pass
                outputs = model(waveforms)
                loss = criterion(outputs['emotion_logits'], labels)
                
                # Track statistics
                val_loss += loss.item()
                val_preds.extend(outputs['emotion_probs'].argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_batches = (num_val_samples // args.batch_size) or 1  # Avoid division by zero
        val_loss /= val_batches
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with F1: {val_f1:.4f}")
        
        # Save last model
        model_path = os.path.join(args.output_dir, 'last_model.pt')
        torch.save(model.state_dict(), model_path)
    
    print(f"Training completed. Best validation F1: {best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Simple Speech Emotion Recognition Training')
    
    # Data parameters
    parser.add_argument('--dataset_root', type=str, default='./processed_dataset',
                        help='Path to processed dataset root')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--max_length', type=int, default=48000,
                        help='Max audio length in samples (3 seconds at 16kHz)')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base-960h',
                        help='Pretrained model name or path')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder parameters')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./models/simple_model',
                        help='Output directory for models')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    train_simple(args)


if __name__ == '__main__':
    main() 