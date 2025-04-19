#!/usr/bin/env python
"""
Simplified training script for RAVDESS dataset
This version avoids using the transformers library to prevent compatibility issues
"""

import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import json
import sys
import random
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# RAVDESS dataset constants
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad', 
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

SIMPLIFIED_EMOTIONS = {
    'neutral': 0,  # Maps neutral and calm
    'happy': 1,    # Maps happy and surprised
    'sad': 2,      # Maps sad and fearful
    'angry': 3     # Maps angry and disgust
}

class RAVDESSDataset(Dataset):
    """Dataset class for the RAVDESS dataset"""
    
    def __init__(self, dataset_root, emotion_set="full", split="train", 
                 sample_rate=16000, max_length=48000, augment=False):
        """
        Args:
            dataset_root (str): Root directory of the RAVDESS dataset
            emotion_set (str): 'full' = all 8 emotions, 'simplified' = 4 basic emotions
            split (str): 'train', 'val', or 'test'
            sample_rate (int): Target sample rate for audio
            max_length (int): Maximum length of audio (will be padded/trimmed)
            augment (bool): Whether to apply data augmentation
        """
        self.dataset_root = dataset_root
        self.emotion_set = emotion_set
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.augment = augment
        
        # Transformations
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        
        # Find all .wav files in the dataset
        all_files = glob(os.path.join(dataset_root, "**/*.wav"), recursive=True)
        logger.info(f"Found {len(all_files)} audio files in {dataset_root}")
        
        # Filter files based on split (train=70%, val=15%, test=15%)
        random.seed(42)  # For reproducibility
        random.shuffle(all_files)
        total_files = len(all_files)
        
        if split == "train":
            self.audio_files = all_files[:int(0.7 * total_files)]
        elif split == "val":
            self.audio_files = all_files[int(0.7 * total_files):int(0.85 * total_files)]
        else:  # test
            self.audio_files = all_files[int(0.85 * total_files):]
        
        logger.info(f"{split} split contains {len(self.audio_files)} files")
        
        # Setup augmentation transforms
        self.time_stretch = torchaudio.transforms.TimeStretch()
        # Initialize PitchShift with default 0 steps (we'll specify steps during application)
        self.pitch_shift = torchaudio.transforms.PitchShift(sample_rate, n_steps=0)
    
    def __len__(self):
        return len(self.audio_files)
    
    def _apply_augmentation(self, waveform):
        """Apply simple noise augmentation to the waveform"""
        if random.random() < 0.5:  # 50% chance of applying noise
            # Add small Gaussian noise
            noise_factor = 0.005
            noise = torch.randn_like(waveform) * noise_factor
            augmented_waveform = waveform + noise
            
            # Ensure the waveform is normalized
            if torch.abs(augmented_waveform).max() > 1.0:
                augmented_waveform = augmented_waveform / torch.abs(augmented_waveform).max()
            
            return augmented_waveform
        
        return waveform
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio file
            waveform, sr = torchaudio.load(audio_path)
            
            # Extract emotion id from filename (RAVDESS format)
            # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
            # Emotion is the third field (index 2)
            filename = os.path.basename(audio_path)
            emotion_id = int(filename.split('-')[2])
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Apply augmentation if enabled
            if self.augment:
                waveform = self._apply_augmentation(waveform)
            
            # Pad or trim
            if self.max_length is not None:
                if waveform.shape[1] < self.max_length:
                    # Pad
                    padding = self.max_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                else:
                    # Trim
                    waveform = waveform[:, :self.max_length]
            
            # Convert to mel spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Convert to log mel spectrogram
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Map to appropriate label
            if self.emotion_set == 'simplified':
                label = self._map_to_simplified_emotion(emotion_id)
            else:
                # RAVDESS labels are 1-indexed, convert to 0-indexed
                label = emotion_id - 1
            
            return mel_spec, label
        
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            # Return a default item to avoid breaking the training loop
            dummy_mel = torch.zeros((1, 128, 94))  # Default mel size for 48000 samples at 16kHz
            return dummy_mel, 0  # Default to neutral emotion
    
    def _extract_emotion_from_filename(self, filepath):
        """Extract emotion ID from RAVDESS filename"""
        # RAVDESS filename format: 03-01-06-01-02-01-12.wav
        # The third part (06) is the emotion
        filename = os.path.basename(filepath)
        parts = filename.split('-')
        
        if len(parts) >= 3:
            return parts[2]
        
        # Return neutral as fallback
        return '01'
    
    def _map_to_simplified_emotion(self, emotion_id):
        """Map RAVDESS emotion ID to simplified emotion set (4 classes)"""
        # RAVDESS emotions:
        # 1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised
        
        # Simplified emotions:
        # 0=neutral, 1=happy, 2=sad, 3=angry
        mapping = {
            1: 0,  # neutral -> neutral
            2: 0,  # calm -> neutral
            3: 1,  # happy -> happy
            4: 2,  # sad -> sad
            5: 3,  # angry -> angry
            6: 2,  # fearful -> sad (closest match)
            7: 3,  # disgust -> angry (closest match)
            8: 1,  # surprised -> happy (closest match)
        }
        return mapping[emotion_id]

class SimpleEmotionRecognitionModel(nn.Module):
    def __init__(self, num_emotions):
        super(SimpleEmotionRecognitionModel, self).__init__()
        
        # Mel spectrogram input shape: (batch_size, 1, 128, time_frames)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the output size after convolutions and pooling
        self.fc_input_dim = 64 * 16 * 11  # For 128x94 input mel spec (with 3 pooling layers)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_emotions)
        
    def forward(self, x):
        # Input x shape: (batch_size, 1, 128, time_frames)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Check input dimensions and provide helpful error
        if x.shape[1] != self.fc_input_dim:
            actual_dim = x.shape[1]
            raise ValueError(f"Expected flattened dimension {self.fc_input_dim}, got {actual_dim}. "
                            f"This may be due to incorrect audio length or mel spectrogram settings.")
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for mel_specs, labels in progress_bar:
        mel_specs, labels = mel_specs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model on validation data"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel_specs, labels in val_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            
            # Track statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(val_loader), correct / total

def main(args):
    """Main training function"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if CUDA is available
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset and data loaders
    logger.info(f"Creating dataset from {args.dataset_root}")
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_root):
        raise ValueError(f"Dataset directory {args.dataset_root} does not exist")
    
    # Create datasets for train, val, and test
    train_dataset = RAVDESSDataset(
        dataset_root=args.dataset_root,
        emotion_set=args.emotion_set,
        split="train",
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        augment=args.augment
    )
    
    val_dataset = RAVDESSDataset(
        dataset_root=args.dataset_root,
        emotion_set=args.emotion_set,
        split="val",
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        augment=False
    )
    
    test_dataset = RAVDESSDataset(
        dataset_root=args.dataset_root,
        emotion_set=args.emotion_set,
        split="test",
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Get number of emotions
    num_emotions = 4 if args.emotion_set == 'simplified' else 8
    
    # Set emotion names
    if args.emotion_set == 'simplified':
        class_names = ['neutral', 'happy', 'sad', 'angry']
    else:
        class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    # Create model
    model = SimpleEmotionRecognitionModel(num_emotions=num_emotions)
    model.to(device)
    
    logger.info(f"Created simple emotion recognition model with {num_emotions} emotions")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    best_model_path = os.path.join(
        args.output_dir,
        f"ravdess_{'simplified' if args.emotion_set == 'simplified' else 'full'}_best.pt"
    )
    
    logger.info(f"Starting training for {args.epochs} epochs")
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'class_names': class_names,
                'num_emotions': num_emotions,
            }, best_model_path)
            
            logger.info(f"Saved best model to {best_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"ravdess_{'simplified' if args.emotion_set == 'simplified' else 'full'}_training_curves.png"))
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save training metadata
    metadata = {
        'num_emotions': num_emotions,
        'emotion_set': args.emotion_set,
        'sample_rate': args.sample_rate,
        'max_length': args.max_length,
        'test_accuracy': test_acc,
        'class_names': class_names,
    }
    
    with open(os.path.join(args.output_dir, f"ravdess_{'simplified' if args.emotion_set == 'simplified' else 'full'}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    logger.info(f"Model and metadata saved to {args.output_dir}")
    
    return best_val_acc, test_acc

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a speech emotion recognition model on RAVDESS dataset')
    
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to RAVDESS dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trained model and results')
    parser.add_argument('--emotion_set', type=str, choices=['full', 'simplified'], required=True,
                        help='Emotion set to use')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='train',
                        help='Split to use')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sample rate for audio')
    parser.add_argument('--max_length', type=int, default=48000,
                        help='Maximum length of audio in samples (16000 = 1 second at 16kHz)')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation during training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cpu or cuda)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 