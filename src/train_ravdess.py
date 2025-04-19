#!/usr/bin/env python
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

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.ravdess_model import RAVDESSEmotionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    
    def __init__(self, root_dir, sample_rate=16000, max_length=None, use_simplified=False, augment=False):
        """
        Initialize the dataset
        
        Args:
            root_dir: Root directory of RAVDESS dataset
            sample_rate: Target sample rate
            max_length: Maximum length of audio in samples (will pad/crop)
            use_simplified: Whether to use simplified emotion categories
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.use_simplified = use_simplified
        self.augment = augment
        
        # Find all .wav files
        self.files = []
        self._find_wav_files(root_dir)
        
        if len(self.files) == 0:
            raise ValueError(f"No audio files found in {root_dir}. Check path or dataset structure.")
        
        logger.info(f"Found {len(self.files)} audio files in {root_dir}")
        
        # Setup augmentation transforms
        self.time_stretch = torchaudio.transforms.TimeStretch()
        self.pitch_shift = torchaudio.transforms.PitchShift(sample_rate)
    
    def _find_wav_files(self, directory):
        """Recursively find all WAV files in directory"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.wav'):
                    self.files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.files)
    
    def _apply_augmentation(self, waveform):
        """Apply random augmentations to waveform"""
        if not self.augment:
            return waveform
            
        # Apply random augmentations with 50% probability each
        if random.random() > 0.5:
            # Time stretching (speed up or slow down)
            stretch_factor = random.uniform(0.9, 1.1)
            try:
                # Need to add frequency dimension for TimeStretch
                waveform_for_stretch = waveform.unsqueeze(0)  # [1, channels, time]
                waveform = self.time_stretch(waveform_for_stretch, stretch_factor).squeeze(0)
            except Exception as e:
                logger.warning(f"Time stretch failed: {e}")
        
        if random.random() > 0.5:
            # Pitch shifting
            shift_steps = random.randint(-2, 2)
            try:
                waveform = self.pitch_shift(waveform, shift_steps)
            except Exception as e:
                logger.warning(f"Pitch shift failed: {e}")
        
        if random.random() > 0.5:
            # Add small amount of noise
            noise_factor = random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
        
        # Make sure we're still normalized
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        audio_path = self.files[idx]
        
        try:
            # Load audio file
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Apply augmentation
            waveform = self._apply_augmentation(waveform)
            
            # Handle length
            if self.max_length is not None:
                if waveform.shape[1] < self.max_length:
                    # Pad
                    padded = torch.zeros(1, self.max_length)
                    padded[0, :waveform.shape[1]] = waveform
                    waveform = padded
                else:
                    # Crop from center
                    start = (waveform.shape[1] - self.max_length) // 2
                    waveform = waveform[:, start:start + self.max_length]
            
            # Extract emotion from filename
            emotion_id = self._extract_emotion_from_filename(audio_path)
            
            # Map to appropriate label
            if self.use_simplified:
                label = self._map_to_simplified_emotion(emotion_id)
            else:
                # For full emotions, map to 0-indexed
                label = int(emotion_id) - 1
            
            return waveform.squeeze(), label
            
        except Exception as e:
            logger.error(f"Error loading file {audio_path}: {e}")
            # Return a zero waveform and neutral emotion as fallback
            waveform = torch.zeros(self.max_length if self.max_length else 16000)
            label = 0
            return waveform, label
    
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
        """Map RAVDESS emotion ID to simplified category"""
        # Get the emotion name from RAVDESS ID
        emotion_name = RAVDESS_EMOTIONS.get(emotion_id, 'neutral')
        
        # Map to simplified categories
        if emotion_name in ['neutral', 'calm']:
            return 0  # neutral
        elif emotion_name in ['happy', 'surprised']:
            return 1  # happy
        elif emotion_name in ['sad', 'fearful']:
            return 2  # sad
        elif emotion_name in ['angry', 'disgust']:
            return 3  # angry
        else:
            return 0  # Default to neutral

def create_data_loaders(dataset, batch_size, val_ratio=0.15, test_ratio=0.15):
    """Create train, validation, and test data loaders"""
    # Calculate lengths
    total_len = len(dataset)
    test_len = int(total_len * test_ratio)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len - test_len
    
    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len]
    )
    
    logger.info(f"Dataset split: train={train_len}, val={val_len}, test={test_len}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for waveforms, labels in train_loader:
        # Move to device
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(waveforms)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for waveforms, labels in val_loader:
            # Move to device
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(waveforms)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate validation statistics
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc

def evaluate_model(model, test_loader, device, class_names, output_dir):
    """Evaluate model and generate metrics and visualizations"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels in test_loader:
            # Move to device
            waveforms = waveforms.to(device)
            
            # Forward pass
            outputs = model(waveforms)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = 100 * np.mean(all_preds == all_labels)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    # Save classification report as JSON
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Saved classification report to {report_path}")
    
    # Print classification report
    logger.info("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return accuracy, report

def plot_training_history(history, output_dir):
    """Plot training and validation loss/accuracy curves"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    history_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_path)
    logger.info(f"Saved training history to {history_path}")

def train_model(args):
    """Main training function"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "ravdess_simplified" if args.use_simplified else "ravdess_full"
    output_dir = os.path.join(args.output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    try:
        dataset = RAVDESSDataset(
            root_dir=args.dataset_root,
            sample_rate=args.sample_rate,
            max_length=args.max_length,
            use_simplified=args.use_simplified,
            augment=args.augment
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, 
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Define class names for reporting
    if args.use_simplified:
        class_names = ["neutral", "happy", "sad", "angry"]
    else:
        class_names = list(RAVDESS_EMOTIONS.values())
    
    # Create model
    model = RAVDESSEmotionModel(
        num_emotions=len(class_names),
        use_simplified=args.use_simplified
    )
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Save model
            model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'history': history
            }, model_path)
            
            logger.info(f"Saved best model with validation accuracy {val_acc:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'history': history
    }, final_model_path)
    
    logger.info(f"Saved final model to {final_model_path}")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Load best model for evaluation
    best_model = RAVDESSEmotionModel(
        num_emotions=len(class_names),
        use_simplified=args.use_simplified
    )
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    # Evaluate model
    logger.info("Evaluating best model on test set")
    test_acc, _ = evaluate_model(
        best_model, test_loader, device, class_names, output_dir
    )
    
    logger.info(f"Final test accuracy: {test_acc:.2f}%")
    logger.info(f"Training completed. Results saved to {output_dir}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train RAVDESS emotion recognition model'
    )
    
    parser.add_argument(
        '--dataset_root', type=str, required=True,
        help='Root directory of RAVDESS dataset'
    )
    
    parser.add_argument(
        '--output_dir', type=str, default='models/ravdess',
        help='Directory to save models and results'
    )
    
    parser.add_argument(
        '--sample_rate', type=int, default=16000,
        help='Audio sample rate'
    )
    
    parser.add_argument(
        '--max_length', type=int, default=48000,
        help='Maximum audio length in samples (3 seconds at 16kHz)'
    )
    
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--val_ratio', type=float, default=0.15,
        help='Validation set ratio'
    )
    
    parser.add_argument(
        '--test_ratio', type=float, default=0.15,
        help='Test set ratio'
    )
    
    parser.add_argument(
        '--use_simplified', action='store_true',
        help='Use simplified emotion categories (4 instead of 8)'
    )
    
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use for training (cuda or cpu)'
    )
    
    parser.add_argument(
        '--augment', action='store_true',
        help='Apply data augmentation during training'
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1) 