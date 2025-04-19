#!/usr/bin/env python3
"""
Ultimate RAVDESS Speech Emotion Recognition Training Script
Integrates advanced techniques for maximum accuracy with all 8 emotions
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from tqdm import tqdm
import random
import warnings
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Training will continue without visualization.")

# Try to import transformers
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available. Using custom CNN encoder instead.")

# Set up argument parser
parser = argparse.ArgumentParser(description="Ultimate RAVDESS Emotion Recognition Training Script")
parser.add_argument("--dataset_root", type=str, required=True, help="Path to prepared RAVDESS dataset")
parser.add_argument("--output_dir", type=str, default="../models/ravdess_ultimate", help="Output directory")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=3e-5, help="Weight decay for regularization")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                    help="Device to use (cuda or cpu)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision training")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
parser.add_argument("--attention_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--hidden_layers", type=int, default=4, help="Number of transformer layers")

# Parse args
args = parser.parse_args()

# Set device
device = torch.device(args.device)

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Advanced dataset class for RAVDESS
class AdvancedRAVDESSDataset(Dataset):
    """
    Advanced dataset class for RAVDESS with advanced augmentation
    """
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    EMOTION_ID_TO_IDX = {
        '01': 0, '02': 1, '03': 2, '04': 3,
        '05': 4, '06': 5, '07': 6, '08': 7
    }
    
    def __init__(self, root_dir, split='train', sample_rate=16000, max_duration=5.0, 
                 use_augmentation=True):
        """
        Initialize the dataset.
        Args:
            root_dir: Root directory of the prepared dataset
            split: 'train', 'val', or 'test'
            sample_rate: Audio sample rate
            max_duration: Maximum audio duration in seconds
            use_augmentation: Whether to use data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_sample_length = int(max_duration * sample_rate)
        self.use_augmentation = use_augmentation and split == 'train'
        
        # Get all audio files for the specified split
        split_dir = self.root_dir / split
        self.file_paths = []
        
        if split_dir.exists():
            for actor_dir in split_dir.glob("Actor_*"):
                if actor_dir.is_dir():
                    for file_path in actor_dir.glob("*.wav"):
                        self.file_paths.append(file_path)
        
        print(f"RAVDESS {split} dataset: {len(self.file_paths)} files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Parse filename for metadata
        parts = file_path.stem.split('-')
        emotion_id = parts[2]  # Emotion ID
        actor_id = parts[6]  # Actor ID
        gender = 'female' if int(actor_id) % 2 == 0 else 'male'
        
        # Load audio
        waveform, orig_sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if orig_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sample_rate, self.sample_rate
            )
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply augmentation (only for training)
        if self.use_augmentation:
            waveform = self._apply_augmentation(waveform)
        
        # Normalize audio
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Pad or truncate to max_sample_length
        if waveform.shape[1] < self.max_sample_length:
            padding = self.max_sample_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_sample_length]
        
        # Convert emotion ID to index
        emotion_idx = self.EMOTION_ID_TO_IDX[emotion_id]
        
        # Create metadata dictionary
        metadata = {
            'emotion_id': emotion_id,
            'emotion_name': self.EMOTIONS[emotion_id],
            'actor_id': actor_id,
            'gender': gender,
            'file_path': str(file_path)
        }
        
        return {
            'waveform': waveform,
            'emotion': emotion_idx,
            'gender': 1 if gender == 'female' else 0,
            'metadata': metadata
        }
    
    def _apply_augmentation(self, waveform):
        """Apply various audio augmentations to improve model robustness"""
        # Random noise augmentation
        if random.random() < 0.5:
            noise_level = random.uniform(0.001, 0.005)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        # Time shift augmentation (shift the audio slightly)
        if random.random() < 0.5:
            shift_factor = random.randint(-self.sample_rate // 10, self.sample_rate // 10)
            waveform = torch.roll(waveform, shifts=shift_factor, dims=1)
        
        # Spectral augmentation (frequency masking)
        if random.random() < 0.3:
            try:
                # Convert to spectrogram
                spec = torchaudio.transforms.Spectrogram()(waveform)
                # Apply frequency masking
                freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
                spec = freq_mask(spec)
                # Convert back to waveform (approximate)
                inverse_spec = torchaudio.transforms.InverseSpectrogram()(spec)
                if not torch.isnan(inverse_spec).any():
                    waveform = inverse_spec
            except Exception:
                # If something goes wrong, just return the original waveform
                pass
        
        return waveform
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for the dataloader"""
        waveforms = torch.stack([item['waveform'] for item in batch])
        emotions = torch.tensor([item['emotion'] for item in batch], dtype=torch.long)
        genders = torch.tensor([item['gender'] for item in batch], dtype=torch.long)
        metadata = [item['metadata'] for item in batch]
        
        return {
            'waveform': waveforms,
            'emotion': emotions,
            'gender': genders,
            'metadata': metadata
        }


# Advanced model with attention and multi-task learning
class UltimateSpeechEmotionRecognizer(nn.Module):
    """
    Advanced speech emotion recognition model with attention and multi-task learning
    """
    def __init__(self, num_emotions=8, attention_heads=8, hidden_layers=4, dropout_rate=0.3,
                 use_gender_branch=True):
        super().__init__()
        
        self.use_gender_branch = use_gender_branch
        self.num_emotions = num_emotions
        
        if TRANSFORMERS_AVAILABLE:
            # Use pre-trained Wav2Vec2 as encoder
            self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            
            # Freeze feature extractor (lower layers)
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
                
            # Get hidden size from wav2vec
            hidden_size = self.wav2vec.config.hidden_size
            
        else:
            # Custom CNN encoder if transformers not available
            hidden_size = 512
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                
                nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                
                nn.AdaptiveAvgPool1d(output_size=1)
            )
        
        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=hidden_layers
        )
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        # Gender classification head (optional)
        if use_gender_branch:
            self.gender_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout_rate/2),
                nn.Linear(hidden_size // 4, 2)
            )
    
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        if TRANSFORMERS_AVAILABLE:
            # Use Wav2Vec2 encoder
            x = x.squeeze(1)  # Remove channel dimension
            outputs = self.wav2vec(x)
            features = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        else:
            # Use custom CNN encoder
            features = self.conv_layers(x)  # (batch_size, hidden_size, 1)
            features = features.transpose(1, 2)  # (batch_size, 1, hidden_size)
            features = features.repeat(1, 10, 1)  # Create sequence dimension
        
        # Apply transformer encoder for contextual representation
        features = self.transformer_encoder(features)
        
        # Global average pooling
        pooled = torch.mean(features, dim=1)  # (batch_size, hidden_size)
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(pooled)
        
        # Output dictionary
        outputs = {
            'emotion_logits': emotion_logits,
            'features': pooled
        }
        
        # Gender classification (optional)
        if self.use_gender_branch:
            gender_logits = self.gender_classifier(pooled)
            outputs['gender_logits'] = gender_logits
        
        return outputs


def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, num_epochs, use_amp=False):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    correct_emotion = 0
    correct_gender = 0
    total = 0
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        waveforms = batch['waveform'].to(device)
        emotion_targets = batch['emotion'].to(device)
        gender_targets = batch['gender'].to(device) if 'gender' in batch else None
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(waveforms)
            emotion_logits = outputs['emotion_logits']
            
            # Main emotion classification loss
            emotion_loss = F.cross_entropy(emotion_logits, emotion_targets)
            loss = emotion_loss
            
            # Optional gender classification loss
            if 'gender_logits' in outputs and gender_targets is not None:
                gender_logits = outputs['gender_logits']
                gender_loss = F.cross_entropy(gender_logits, gender_targets)
                # Multi-task loss with weighting
                loss = emotion_loss + 0.2 * gender_loss
        
        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Emotion accuracy
        _, predicted = emotion_logits.max(1)
        correct_emotion += predicted.eq(emotion_targets).sum().item()
        
        # Gender accuracy (if applicable)
        if 'gender_logits' in outputs and gender_targets is not None:
            _, g_pred = outputs['gender_logits'].max(1)
            correct_gender += g_pred.eq(gender_targets).sum().item()
        
        total += emotion_targets.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{epoch_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct_emotion/total:.2f}%"
        })
    
    # Calculate final metrics
    avg_loss = epoch_loss / len(train_loader)
    emotion_acc = 100. * correct_emotion / total
    gender_acc = 100. * correct_gender / total if correct_gender > 0 else 0.0
    
    return avg_loss, emotion_acc, gender_acc


def validate(model, val_loader, device, epoch, num_epochs):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct_emotion = 0
    correct_gender = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    # Progress bar
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            waveforms = batch['waveform'].to(device)
            emotion_targets = batch['emotion'].to(device)
            gender_targets = batch['gender'].to(device) if 'gender' in batch else None
            
            # Forward pass
            outputs = model(waveforms)
            emotion_logits = outputs['emotion_logits']
            
            # Main emotion classification loss
            emotion_loss = F.cross_entropy(emotion_logits, emotion_targets)
            loss = emotion_loss
            
            # Optional gender classification loss
            if 'gender_logits' in outputs and gender_targets is not None:
                gender_logits = outputs['gender_logits']
                gender_loss = F.cross_entropy(gender_logits, gender_targets)
                # Multi-task loss with weighting
                loss = emotion_loss + 0.2 * gender_loss
            
            # Update metrics
            val_loss += loss.item()
            
            # Emotion accuracy
            _, predicted = emotion_logits.max(1)
            correct_emotion += predicted.eq(emotion_targets).sum().item()
            
            # Gender accuracy (if applicable)
            if 'gender_logits' in outputs and gender_targets is not None:
                _, g_pred = outputs['gender_logits'].max(1)
                correct_gender += g_pred.eq(gender_targets).sum().item()
            
            total += emotion_targets.size(0)
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(emotion_targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{val_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct_emotion/total:.2f}%"
            })
    
    # Calculate final metrics
    avg_loss = val_loss / len(val_loader)
    emotion_acc = 100. * correct_emotion / total
    gender_acc = 100. * correct_gender / total if correct_gender > 0 else 0.0
    
    return avg_loss, emotion_acc, gender_acc, np.array(all_preds), np.array(all_targets)


def main():
    """Main function"""
    print(f"Training with the following parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Create datasets
    train_dataset = AdvancedRAVDESSDataset(
        root_dir=args.dataset_root,
        split='train',
        use_augmentation=True
    )
    
    val_dataset = AdvancedRAVDESSDataset(
        root_dir=args.dataset_root,
        split='val',
        use_augmentation=False
    )
    
    test_dataset = AdvancedRAVDESSDataset(
        root_dir=args.dataset_root,
        split='test',
        use_augmentation=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=AdvancedRAVDESSDataset.collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AdvancedRAVDESSDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AdvancedRAVDESSDataset.collate_fn
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Create model
    model = UltimateSpeechEmotionRecognizer(
        num_emotions=8,
        attention_heads=args.attention_heads,
        hidden_layers=args.hidden_layers,
        dropout_rate=0.3,
        use_gender_branch=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,}/{total_params:,} trainable")
    
    # Optimizer with weight decay (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Automatic mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_emotion_acc, train_gender_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args.epochs, args.use_amp
        )
        
        # Validate
        val_loss, val_emotion_acc, val_gender_acc, preds, targets = validate(
            model, val_loader, device, epoch, args.epochs
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_emotion_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_emotion_acc:.2f}%")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_emotion_acc, epoch)
            writer.add_scalar('Accuracy/val', val_emotion_acc, epoch)
            writer.add_scalar('Gender_Accuracy/train', train_gender_acc, epoch)
            writer.add_scalar('Gender_Accuracy/val', val_gender_acc, epoch)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint if it's the best model so far
        if val_emotion_acc > best_val_acc:
            best_val_acc = val_emotion_acc
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': val_emotion_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with validation accuracy: {val_emotion_acc:.2f}%")
        
        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': val_emotion_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_acc': val_emotion_acc,
        'args': vars(args)
    }
    torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Saved final model with validation accuracy: {val_emotion_acc:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_emotion_acc, test_gender_acc, test_preds, test_targets = validate(
        model, test_loader, device, args.epochs, args.epochs
    )
    print(f"Test accuracy: {test_emotion_acc:.2f}%")
    
    # Print confusion matrix
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test_targets, test_preds)
        emotion_names = [AdvancedRAVDESSDataset.EMOTIONS[f"{i+1:02d}"] for i in range(8)]
        
        print("\nConfusion Matrix:")
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(test_targets, test_preds, target_names=emotion_names))
    except:
        print("Could not generate detailed metrics, scikit-learn might be missing.")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main() 