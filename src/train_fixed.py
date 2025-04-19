#!/usr/bin/env python3
"""
Fixed training script for speech emotion recognition models.
This version handles tensor dimension issues and ensures proper training.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import logging
import argparse
from pathlib import Path
from sklearn.metrics import f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing the model - handle potential errors
try:
    from model import SpeechEmotionRecognitionModel
    from model_enhanced import EnhancedSpeechEmotionRecognitionModel
except RuntimeError as e:
    logger.warning(f"Error importing models, falling back to local implementation: {e}")
    # We'll define simplified versions of the models below

# Try to import data utilities - implement basic versions if import fails
try:
    from data_utils import EmotionSpeechDataset, create_data_splits
except ImportError:
    logger.warning("Could not import data utilities, using built-in implementations")
    
    class EmotionSpeechDataset(Dataset):
        """Dataset for speech emotion recognition"""
        def __init__(self, audio_paths, emotion_labels, sample_rate=16000, max_length=None):
            self.audio_paths = audio_paths
            self.emotion_labels = emotion_labels
            self.sample_rate = sample_rate
            self.max_length = max_length
            self.num_emotions = len(set(emotion_labels))
            
            # Map emotion labels to indices
            unique_emotions = sorted(set(emotion_labels))
            self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
            
        def __len__(self):
            return len(self.audio_paths)
            
        def __getitem__(self, idx):
            audio_path = self.audio_paths[idx]
            emotion = self.emotion_labels[idx]
            emotion_idx = self.emotion_to_idx[emotion]
            
            # Load audio file
            try:
                import librosa
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                
                # Apply fixed length
                if self.max_length:
                    target_length = int(self.sample_rate * self.max_length)
                    if len(waveform) > target_length:
                        waveform = waveform[:target_length]
                    else:
                        # Pad with zeros
                        padding = np.zeros(int(target_length - len(waveform)))
                        waveform = np.concatenate([waveform, padding])
                
                # Convert to tensor
                waveform = torch.from_numpy(waveform).float()
                
                return {
                    'waveform': waveform,
                    'emotion': emotion_idx
                }
            except Exception as e:
                logger.error(f"Error loading {audio_path}: {e}")
                # Return a dummy sample as fallback
                dummy_length = int(self.sample_rate * (self.max_length or 3))
                dummy_waveform = torch.zeros(dummy_length).float()
                return {
                    'waveform': dummy_waveform,
                    'emotion': emotion_idx
                }
                
    def create_data_splits(dataset_root, sample_rate=16000, max_length=5, 
                          train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Create train, validation, and test splits from a directory of audio files"""
        import os
        import glob
        
        # Get all audio files
        audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(dataset_root, '**', ext), recursive=True))
        
        # Check if we found any audio files
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in {dataset_root}. Please check the path or audio file formats.")
        
        logger.info(f"Found {len(audio_files)} audio files in {dataset_root}")
        
        # Extract emotion labels from filenames or parent directory names
        emotion_labels = []
        for file_path in audio_files:
            # Try to extract emotion from parent directory name
            parent_dir = os.path.basename(os.path.dirname(file_path))
            
            # Check if parent directory is an emotion label (you may need to customize this)
            common_emotions = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise']
            if parent_dir.lower() in common_emotions:
                emotion_labels.append(parent_dir.lower())
            else:
                # Try to extract from filename
                filename = os.path.basename(file_path)
                for emotion in common_emotions:
                    if emotion in filename.lower():
                        emotion_labels.append(emotion)
                        break
                else:
                    # Default to neutral if no emotion is found
                    emotion_labels.append('neutral')
        
        # Shuffle the data
        indices = list(range(len(audio_files)))
        random.shuffle(indices)
        audio_files = [audio_files[i] for i in indices]
        emotion_labels = [emotion_labels[i] for i in indices]
        
        # Create splits
        train_size = int(len(audio_files) * train_ratio)
        val_size = int(len(audio_files) * val_ratio)
        
        train_files = audio_files[:train_size]
        train_emotions = emotion_labels[:train_size]
        
        val_files = audio_files[train_size:train_size+val_size]
        val_emotions = emotion_labels[train_size:train_size+val_size]
        
        test_files = audio_files[train_size+val_size:]
        test_emotions = emotion_labels[train_size+val_size:]
        
        # Ensure we have at least one sample in each split
        if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
            logger.warning("Not enough data for proper splits. Creating dummy splits.")
            # Create dummy splits with at least one sample each
            if len(audio_files) < 3:
                # If we have fewer than 3 files, duplicate them
                while len(audio_files) < 3:
                    audio_files.extend(audio_files)
                    emotion_labels.extend(emotion_labels)
            
            train_files = [audio_files[0]]
            train_emotions = [emotion_labels[0]]
            val_files = [audio_files[1]]
            val_emotions = [emotion_labels[1]]
            test_files = [audio_files[2]]
            test_emotions = [emotion_labels[2]]
        
        logger.info(f"Created splits: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test samples")
        
        # Create datasets
        train_set = EmotionSpeechDataset(train_files, train_emotions, sample_rate, max_length)
        val_set = EmotionSpeechDataset(val_files, val_emotions, sample_rate, max_length)
        test_set = EmotionSpeechDataset(test_files, test_emotions, sample_rate, max_length)
        
        return train_set, val_set, test_set

# Define a simple fallback model that ensures correct tensor dimensions
class SimpleEmotionRecognitionModel(nn.Module):
    """Simple speech emotion recognition model that handles tensor dimensions properly"""
    def __init__(self, num_emotions=7):
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling to handle variable length
        )
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )
        
        # Voice activity detection branch
        self.vad = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Handle different input shapes
        if x.dim() == 2:  # [batch_size, sequence_length]
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, sequence_length]
        elif x.dim() == 3 and x.shape[1] != 1:  # Wrong channel dimension
            x = x.transpose(1, 2)  # Transpose to [batch_size, 1, sequence_length]
        elif x.dim() == 4:  # Extra dimension
            x = x.squeeze(2)  # Remove extra dimension
            
        # Apply convolutional layers
        features = self.conv_layers(x)
        features = features.squeeze(-1)  # Remove last dimension after global pooling
        
        # Apply classifiers
        emotion_logits = self.classifier(features)
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        
        # Voice activity detection
        vad_probs = self.vad(features)
        
        return emotion_probs, vad_probs

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(args):
    """Train a speech emotion recognition model with fixes for dimension issues"""
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create datasets and data loaders
    logger.info(f"Loading dataset from {args.dataset_root}")
    train_set, val_set, test_set = create_data_splits(
        dataset_root=args.dataset_root,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    logger.info(f"Training on {len(train_set)} samples")
    logger.info(f"Validating on {len(val_set)} samples")
    
    # Create data loaders with appropriate num_workers
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Initialize model
    num_emotions = train_set.num_emotions
    if args.use_enhanced_model:
        try:
            model = EnhancedSpeechEmotionRecognitionModel(num_emotions=num_emotions)
            logger.info("Using Enhanced Speech Emotion Recognition Model")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced model: {e}")
            logger.info("Falling back to simple model")
            model = SimpleEmotionRecognitionModel(num_emotions=num_emotions)
    else:
        try:
            model = SpeechEmotionRecognitionModel(num_emotions=num_emotions)
            logger.info("Using Standard Speech Emotion Recognition Model")
        except Exception as e:
            logger.warning(f"Could not initialize standard model: {e}")
            logger.info("Falling back to simple model")
            model = SimpleEmotionRecognitionModel(num_emotions=num_emotions)
    
    # Move model to device
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize learning rate scheduler
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs * len(train_loader)
        )
    elif args.lr_scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            verbose=True
        )
    else:
        scheduler = None
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Track best model
    best_val_f1 = 0.0
    best_epoch = 0
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            waveforms = batch['waveform'].to(device)
            emotions = batch['emotion'].to(device)
            
            # Fix input dimensions if needed
            if waveforms.dim() == 4:  # If shape is [batch, 1, 1, seq_len]
                waveforms = waveforms.squeeze(2)  # Remove the extra dimension
                
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # Handle different model output formats
                outputs = model(waveforms)
                
                # Check if output is a dictionary or tuple
                if isinstance(outputs, dict):
                    emotion_logits = outputs.get('emotion_logits', outputs.get('emotion_probs'))
                    # If it's probabilities, convert to logits for loss calculation
                    if torch.all((emotion_logits >= 0) & (emotion_logits <= 1)):
                        # Add a small epsilon to avoid log(0)
                        emotion_logits = torch.log(emotion_logits + 1e-7)
                elif isinstance(outputs, tuple):
                    emotion_logits = outputs[0]
                    # If it's probabilities, convert to logits
                    if torch.all((emotion_logits >= 0) & (emotion_logits <= 1)):
                        emotion_logits = torch.log(emotion_logits + 1e-7)
                else:
                    emotion_logits = outputs
                
                # Calculate loss
                loss = criterion(emotion_logits, emotions)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Update learning rate if using a step scheduler
                if scheduler is not None and args.lr_scheduler != "reduce_on_plateau":
                    scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                _, predicted = emotion_logits.max(1)
                train_total += emotions.size(0)
                train_correct += predicted.eq(emotions).sum().item()
                
                # Save predictions for F1 score
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(emotions.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{train_loss/(batch_idx+1):.4f}",
                    'acc': f"{100.*train_correct/train_total:.2f}%"
                })
                
            except RuntimeError as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                # Skip this batch and continue
                continue
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logger.info(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                waveforms = batch['waveform'].to(device)
                emotions = batch['emotion'].to(device)
                
                # Fix input dimensions if needed
                if waveforms.dim() == 4:  # If shape is [batch, 1, 1, seq_len]
                    waveforms = waveforms.squeeze(2)  # Remove the extra dimension
                
                try:
                    # Forward pass
                    outputs = model(waveforms)
                    
                    # Check if output is a dictionary or tuple
                    if isinstance(outputs, dict):
                        emotion_logits = outputs.get('emotion_logits', outputs.get('emotion_probs'))
                        # If it's probabilities, convert to logits for loss calculation
                        if torch.all((emotion_logits >= 0) & (emotion_logits <= 1)):
                            emotion_logits = torch.log(emotion_logits + 1e-7)
                    elif isinstance(outputs, tuple):
                        emotion_logits = outputs[0]
                        # If it's probabilities, convert to logits
                        if torch.all((emotion_logits >= 0) & (emotion_logits <= 1)):
                            emotion_logits = torch.log(emotion_logits + 1e-7)
                    else:
                        emotion_logits = outputs
                    
                    # Calculate loss
                    loss = criterion(emotion_logits, emotions)
                    
                    # Update metrics
                    val_loss += loss.item()
                    _, predicted = emotion_logits.max(1)
                    val_total += emotions.size(0)
                    val_correct += predicted.eq(emotions).sum().item()
                    
                    # Save predictions for F1 score
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(emotions.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{val_loss/(batch_idx+1):.4f}",
                        'acc': f"{100.*val_correct/val_total:.2f}%"
                    })
                    
                except RuntimeError as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    # Skip this batch and continue
                    continue
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logger.info(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Update learning rate scheduler if using ReduceLROnPlateau
        if scheduler is not None and args.lr_scheduler == "reduce_on_plateau":
            scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'num_emotions': num_emotions
            }, output_dir / 'best_model.pt')
            
            logger.info(f"Saved best model with F1: {val_f1:.4f}")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'num_emotions': num_emotions
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'num_emotions': num_emotions
    }, output_dir / 'last_model.pt')
    
    logger.info(f"Training completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Test best model
    logger.info("Evaluating best model on test set...")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test phase
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            waveforms = batch['waveform'].to(device)
            emotions = batch['emotion'].to(device)
            
            # Fix input dimensions if needed
            if waveforms.dim() == 4:  # If shape is [batch, 1, 1, seq_len]
                waveforms = waveforms.squeeze(2)  # Remove the extra dimension
            
            try:
                # Forward pass
                outputs = model(waveforms)
                
                # Check if output is a dictionary or tuple
                if isinstance(outputs, dict):
                    emotion_logits = outputs.get('emotion_logits', outputs.get('emotion_probs'))
                    # If it's probabilities, convert to logits for loss calculation
                    if torch.all((emotion_logits >= 0) & (emotion_logits <= 1)):
                        emotion_logits = torch.log(emotion_logits + 1e-7)
                elif isinstance(outputs, tuple):
                    emotion_logits = outputs[0]
                    # If it's probabilities, convert to logits
                    if torch.all((emotion_logits >= 0) & (emotion_logits <= 1)):
                        emotion_logits = torch.log(emotion_logits + 1e-7)
                else:
                    emotion_logits = outputs
                
                # Calculate loss
                loss = criterion(emotion_logits, emotions)
                
                # Update metrics
                test_loss += loss.item()
                _, predicted = emotion_logits.max(1)
                test_total += emotions.size(0)
                test_correct += predicted.eq(emotions).sum().item()
                
                # Save predictions for F1 score
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(emotions.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{test_loss/(batch_idx+1):.4f}",
                    'acc': f"{100.*test_correct/test_total:.2f}%"
                })
                
            except RuntimeError as e:
                logger.error(f"Error in test batch {batch_idx}: {e}")
                # Skip this batch and continue
                continue
    
    # Calculate test metrics
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    torch.save(test_results, output_dir / 'test_results.pt')
    
    return {
        'best_val_f1': best_val_f1,
        'test_f1': test_f1,
        'test_acc': test_acc
    }

def main():
    parser = argparse.ArgumentParser(description="Train speech emotion recognition models")
    
    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--max_length", type=float, default=5.0, help="Maximum audio length in seconds")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    # Model arguments
    parser.add_argument("--use_enhanced_model", action="store_true", help="Use enhanced model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        choices=["cosine", "reduce_on_plateau", "none"], 
                        help="Learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    
    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="./models/new_model", help="Output directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_root):
        logger.error(f"Dataset path does not exist: {args.dataset_root}")
        sys.exit(1)
        
    # Train model
    try:
        results = train_model(args)
        logger.info(f"Training complete with best val F1: {results['best_val_f1']:.4f}, test F1: {results['test_f1']:.4f}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 