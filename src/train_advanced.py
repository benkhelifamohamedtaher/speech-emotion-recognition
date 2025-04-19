#!/usr/bin/env python
"""
Advanced training script for speech emotion recognition
Optimized to handle large datasets and prevent dimension issues
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sns
from pathlib import Path
import json
import sys
import random
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import time
import warnings

# Import our advanced model
try:
    from advanced_model import AdvancedSpeechEmotionModel, EMOTION_MAPPINGS
except ImportError:
    print("Error importing AdvancedSpeechEmotionModel. Make sure advanced_model.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AudioDataset(Dataset):
    """Audio dataset that can handle large audio files with proper error handling"""
    
    def __init__(self, 
                 dataset_root, 
                 emotion_set="full", 
                 split="train", 
                 sample_rate=16000, 
                 max_length=48000, 
                 augment=False,
                 audio_extensions=(".wav", ".mp3", ".flac")):
        """
        Args:
            dataset_root (str): Root directory of the dataset
            emotion_set (str): 'full' = all 8 emotions, 'simplified' = 4 basic emotions
            split (str): 'train', 'val', or 'test'
            sample_rate (int): Target sample rate for audio
            max_length (int): Maximum length of audio (will be padded/trimmed)
            augment (bool): Whether to apply data augmentation
            audio_extensions (tuple): Extensions of audio files to include
        """
        self.dataset_root = dataset_root
        self.emotion_set = emotion_set
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.augment = augment
        self.audio_extensions = audio_extensions
        
        # Finding audio files
        logger.info(f"Scanning dataset directory: {dataset_root}")
        all_files = []
        
        # Recursive scan to find all audio files
        for ext in self.audio_extensions:
            all_files.extend(glob(os.path.join(dataset_root, f"**/*{ext}"), recursive=True))
        
        # Check if files were found
        if len(all_files) == 0:
            raise ValueError(f"No audio files found in {dataset_root}. Check the path or audio file extensions.")
        
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
        
        # Ensure we have at least one sample in each split
        if len(self.audio_files) == 0:
            logger.warning(f"No files found for {split} split. Using a small subset of all files.")
            if total_files < 3:
                # If we have very few files, duplicate them to ensure we have enough
                self.audio_files = all_files
            else:
                # Otherwise use a small subset
                self.audio_files = all_files[:max(1, int(0.1 * total_files))]
        
        logger.info(f"{split} split contains {len(self.audio_files)} files")
        
        # Set up transforms
        self.transform = torchaudio.transforms.Resample(
            orig_freq=None,  # Will be set during loading
            new_freq=sample_rate
        )
    
    def __len__(self):
        return len(self.audio_files)
    
    def _extract_emotion_from_filename(self, filepath):
        """Extract emotion from filename based on common patterns"""
        filename = os.path.basename(filepath)
        
        # RAVDESS pattern: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        if '-' in filename and len(filename.split('-')) >= 3:
            parts = filename.split('-')
            try:
                # RAVDESS emotion is the third part (index 2), 1-indexed in the dataset
                emotion_id = int(parts[2]) - 1  # Convert to 0-indexed
                return emotion_id
            except (ValueError, IndexError):
                pass
        
        # Try to find emotion in the path (e.g., .../angry/audio.wav)
        emotion_keywords = {
            "neutral": 0, "calm": 1, "happy": 2, "sad": 3, 
            "angry": 4, "fear": 5, "fearful": 5, "disgust": 6, "surprise": 7, "surprised": 7
        }
        
        for keyword, emotion_id in emotion_keywords.items():
            if keyword in filepath.lower():
                return emotion_id
        
        # Default to neutral
        logger.warning(f"Could not determine emotion for {filepath}, defaulting to neutral")
        return 0  # Neutral
    
    def _apply_augmentation(self, waveform):
        """Apply audio augmentations with proper error handling"""
        if not self.augment or random.random() > 0.5:
            return waveform
            
        try:
            aug_type = random.choice(['noise', 'volume', 'pitch', 'speed'])
            
            if aug_type == 'noise':
                # Add Gaussian noise
                noise_level = random.uniform(0.001, 0.005)
                noise = torch.randn_like(waveform) * noise_level
                waveform = waveform + noise
                
            elif aug_type == 'volume':
                # Random volume change
                volume_factor = random.uniform(0.8, 1.2)
                waveform = waveform * volume_factor
                
            elif aug_type == 'pitch':
                # Time domain pitch shift (simplified)
                stretch_factor = random.uniform(0.9, 1.1)
                orig_len = waveform.shape[1]
                # Interpolate to stretch or compress
                indices = torch.linspace(0, orig_len - 1, int(orig_len * stretch_factor))
                indices = indices.long().clamp(min=0, max=orig_len - 1)
                waveform = waveform[:, indices]
                
            elif aug_type == 'speed':
                # Speed change (time stretching/compression)
                speed_factor = random.uniform(0.9, 1.1)
                orig_len = waveform.shape[1]
                # Interpolate to stretch or compress
                indices = torch.linspace(0, orig_len - 1, int(orig_len / speed_factor))
                indices = indices.long().clamp(min=0, max=orig_len - 1)
                waveform = waveform[:, indices]
            
            # Ensure the waveform is normalized
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
                
            return waveform
        except Exception as e:
            logger.warning(f"Error in augmentation: {str(e)}. Returning original waveform.")
            return waveform
    
    def _map_to_simplified_emotion(self, emotion_id):
        """Map emotion ID to simplified emotion set (4 classes)"""
        # Convert to the simplified emotion mapping
        if emotion_id in [0, 1]:  # neutral, calm
            return 0  # neutral
        elif emotion_id in [2, 7]:  # happy, surprised
            return 1  # happy
        elif emotion_id in [3, 5]:  # sad, fearful
            return 2  # sad
        elif emotion_id in [4, 6]:  # angry, disgust
            return 3  # angry
        else:
            return 0  # default to neutral
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with improved error handling"""
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio file
            waveform, sr = torchaudio.load(audio_path)
            
            # Extract emotion from the filepath
            emotion_id = self._extract_emotion_from_filename(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Update the resampler's original sample rate and resample
            if sr != self.sample_rate:
                self.transform.orig_freq = sr
                waveform = self.transform(waveform)
            
            # Apply augmentation if enabled
            if self.augment and self.split == "train":
                waveform = self._apply_augmentation(waveform)
            
            # Normalize audio to [-1, 1]
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            # Pad or trim
            if self.max_length is not None:
                if waveform.shape[1] < self.max_length:
                    # Pad
                    padding = self.max_length - waveform.shape[1]
                    waveform = F.pad(waveform, (0, padding))
                else:
                    # Trim
                    waveform = waveform[:, :self.max_length]
            
            # Map to appropriate label
            if self.emotion_set == 'simplified':
                label = self._map_to_simplified_emotion(emotion_id)
            else:
                # Use the full emotion mapping
                label = emotion_id
            
            return waveform, label
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            # Return a default item to avoid breaking the training loop
            dummy_waveform = torch.zeros((1, self.max_length))
            return dummy_waveform, 0  # Default to neutral

# Improved collate function to handle dimension issues
def collate_fn(batch):
    """Custom collate function to handle samples with different shapes"""
    # Filter out invalid samples (e.g., empty waveforms)
    batch = [(waveform, label) for waveform, label in batch if waveform.shape[1] > 0]
    
    if len(batch) == 0:
        # If all samples were invalid, return a dummy batch
        return torch.zeros((1, 1, 48000)), torch.zeros(1, dtype=torch.long)
    
    # Extract waveforms and labels
    waveforms, labels = zip(*batch)
    
    # Stack waveforms and convert labels to tensor
    waveforms = torch.stack(waveforms)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return waveforms, labels

# Dataloader creation function
def create_dataloaders(args):
    """Create training, validation, and test dataloaders with error handling"""
    try:
        # Check if dataset path exists
        dataset_path = args.dataset_root
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        # Create datasets
        train_dataset = AudioDataset(
            dataset_root=dataset_path,
            emotion_set=args.emotion_set,
            split="train",
            sample_rate=args.sample_rate,
            max_length=args.max_length,
            augment=args.augment
        )
        
        val_dataset = AudioDataset(
            dataset_root=dataset_path,
            emotion_set=args.emotion_set,
            split="val",
            sample_rate=args.sample_rate,
            max_length=args.max_length,
            augment=False
        )
        
        test_dataset = AudioDataset(
            dataset_root=dataset_path,
            emotion_set=args.emotion_set,
            split="test",
            sample_rate=args.sample_rate,
            max_length=args.max_length,
            augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
            collate_fn=collate_fn,
            persistent_workers=args.num_workers > 0,
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        raise

def update_config_with_overrides(config, overrides):
    """Update configuration with command line overrides"""
    for key, value in overrides.items():
        keys = key.split('.')
        cfg = config
        for k in keys[:-1]:
            cfg = cfg[k]
        cfg[keys[-1]] = value
    return config

def load_config(config_path, **overrides):
    """Load configuration from YAML file and apply overrides"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides if any
    if overrides:
        config = update_config_with_overrides(config, overrides)
    
    return config

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_scheduler(optimizer, config, num_training_steps):
    """Configure learning rate scheduler"""
    scheduler_type = config['scheduler']['scheduler_type']
    
    if scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        warmup_steps = config['scheduler']['warmup_steps']
        return get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        warmup_steps = config['scheduler']['warmup_steps']
        return get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "onecycle":
        warmup_steps = config['scheduler']['warmup_steps']
        pct_start = warmup_steps / num_training_steps
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            total_steps=num_training_steps,
            pct_start=pct_start,
            anneal_strategy='cos'
        )
    else:
        logger.info(f"Scheduler type {scheduler_type} not recognized, using constant schedule")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

def train_advanced(config_path, **overrides):
    """Advanced training function for speech emotion recognition models"""
    # Load config
    config = load_config(config_path, **overrides)
    
    # Set random seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize datasets
    dataset_root = config['dataset']['dataset_root']
    sample_rate = config['dataset']['sample_rate']
    max_length = config['dataset']['max_length']
    
    logger.info(f"Creating datasets from {dataset_root}")
    
    train_set, val_set, test_set = create_data_splits(
        dataset_root=dataset_root,
        sample_rate=sample_rate,
        max_length=max_length,
        train_ratio=config['dataset']['train_ratio'],
        val_ratio=config['dataset']['val_ratio'],
        test_ratio=config['dataset']['test_ratio']
    )
    
    # Initialize augmenter if enabled
    augmenter = None
    if any([config['advanced'].get(aug, False) for aug in ['pitch_shift', 'time_stretch', 'add_noise', 'time_mask', 'freq_mask']]):
        augmenter = AudioAugmenter(
            pitch_shift=config['advanced'].get('pitch_shift', False),
            time_stretch=config['advanced'].get('time_stretch', False),
            add_noise=config['advanced'].get('add_noise', False),
            time_mask=config['advanced'].get('time_mask', False),
            freq_mask=config['advanced'].get('freq_mask', False)
        )
        train_set.set_augmenter(augmenter)
        logger.info("Data augmentation enabled")
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    logger.info(f"Initializing model with {config['model']['model_name']}")
    
    if config.get('model', {}).get('use_enhanced_model', False):
        model = EnhancedSpeechEmotionRecognitionModel(
            pretrained_model_name=config['model']['model_name'],
            num_emotions=train_set.num_emotions
        )
        logger.info("Using Enhanced Speech Emotion Recognition Model")
    else:
        model = SpeechEmotionRecognitionModel(
            pretrained_model_name=config['model']['model_name'],
            num_emotions=train_set.num_emotions
        )
        logger.info("Using Standard Speech Emotion Recognition Model")
    
    model.to(device)
    
    # Freeze encoder initially if specified
    if config['model'].get('unfreeze_encoder_epoch', 0) > 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info(f"Encoder frozen for {config['model']['unfreeze_encoder_epoch']} epochs")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize EMA if enabled
    ema = None
    if config['advanced'].get('ema_decay', 0) > 0:
        ema = EMA(model, config['advanced']['ema_decay'])
        logger.info(f"EMA enabled with decay {config['advanced']['ema_decay']}")
    
    # Initialize gradient scaler for mixed precision training
    scaler = None
    if config['training'].get('fp16_training', False) and AMP_AVAILABLE:
        scaler = GradScaler()
        logger.info("Mixed precision training enabled")
    
    # Track metrics for early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    no_improvement_epochs = 0
    early_stopping_patience = config['advanced'].get('early_stopping_patience', float('inf'))
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config['training']['epochs']):
        # Unfreeze encoder if it's time
        if epoch == config['model'].get('unfreeze_encoder_epoch', 0) and epoch > 0:
            for param in model.encoder.parameters():
                param.requires_grad = True
            logger.info("Encoder unfrozen")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        for batch_idx, batch in enumerate(pbar):
            waveforms, emotions = batch['waveform'].to(device), batch['emotion'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with autocast():
                    outputs = model(waveforms)
                    loss = criterion(outputs['emotion_logits'], emotions)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping if enabled
                if config['training'].get('clip_grad', 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
                
                # Update weights with gradient scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(waveforms)
                loss = criterion(outputs['emotion_logits'], emotions)
                loss.backward()
                
                # Gradient clipping if enabled
                if config['training'].get('clip_grad', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
                
                optimizer.step()
            
            # Update EMA if enabled
            if ema is not None:
                ema.update()
            
            # Update learning rate
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs['emotion_logits'].max(1)
            train_total += emotions.size(0)
            train_correct += predicted.eq(emotions).sum().item()
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{train_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%",
                'lr': f"{current_lr:.6f}"
            })
            
            # Log metrics
            if (batch_idx + 1) % config['logging']['log_interval'] == 0:
                logger.info(f"Train Epoch: {epoch+1} [{batch_idx+1}/{len(train_loader)}]"
                            f" Loss: {train_loss/(batch_idx+1):.4f}"
                            f" Acc: {100.*train_correct/train_total:.2f}%"
                            f" LR: {current_lr:.6f}")
        
        # Calculate training metrics for the epoch
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        if (epoch + 1) % config['logging']['eval_interval'] == 0:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            # Apply EMA if enabled
            if ema is not None:
                ema.apply_shadow()
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
                for batch_idx, batch in enumerate(pbar):
                    waveforms, emotions = batch['waveform'].to(device), batch['emotion'].to(device)
                    
                    outputs = model(waveforms)
                    loss = criterion(outputs['emotion_logits'], emotions)
                    
                    val_loss += loss.item()
                    _, predicted = outputs['emotion_logits'].max(1)
                    val_total += emotions.size(0)
                    val_correct += predicted.eq(emotions).sum().item()
                    
                    pbar.set_postfix({
                        'loss': f"{val_loss/(batch_idx+1):.4f}",
                        'acc': f"{100.*val_correct/val_total:.2f}%"
                    })
            
            # Restore model parameters if EMA was applied
            if ema is not None:
                ema.restore()
            
            # Calculate validation metrics
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            logger.info(f"Validation Epoch: {epoch+1}"
                        f" Loss: {val_loss:.4f}"
                        f" Acc: {val_acc:.2f}%")
            
            # Check for improvement
            improved = False
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                improved = True
                
                # Save best model
                if ema is not None:
                    ema.apply_shadow()
                
                checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'config': config
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
                
                if ema is not None:
                    ema.restore()
            
            # Early stopping check
            if improved:
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                
            if no_improvement_epochs >= early_stopping_patience and early_stopping_patience > 0:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save regular checkpoint
        if (epoch + 1) % config['logging'].get('checkpoint_interval', 5) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            
            if ema is not None:
                ema.apply_shadow()
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss if 'val_loss' in locals() else None,
                'val_acc': val_acc if 'val_acc' in locals() else None,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'config': config
            }, checkpoint_path)
            
            if ema is not None:
                ema.restore()
                
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final evaluation on test set
    logger.info("Training completed. Evaluating on test set...")
    model.eval()
    
    # Load best model for evaluation
    best_checkpoint_files = list(checkpoint_dir.glob("best_model_*.pt"))
    if best_checkpoint_files:
        best_checkpoint_path = max(best_checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_checkpoint_path}")
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(pbar):
            waveforms, emotions = batch['waveform'].to(device), batch['emotion'].to(device)
            
            outputs = model(waveforms)
            loss = criterion(outputs['emotion_logits'], emotions)
            
            test_loss += loss.item()
            _, predicted = outputs['emotion_logits'].max(1)
            test_total += emotions.size(0)
            test_correct += predicted.eq(emotions).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(emotions.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{test_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*test_correct/test_total:.2f}%"
            })
    
    # Calculate test metrics
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    logger.info(f"Test results:"
                f" Loss: {test_loss:.4f}"
                f" Acc: {test_acc:.2f}%")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = checkpoint_dir / f"test_results_{timestamp}.pt"
    torch.save({
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': all_predictions,
        'targets': all_targets,
        'config': config
    }, results_path)
    logger.info(f"Saved test results to {results_path}")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc
    }

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler=None):
    """Train model for one epoch with robust error handling and dimension fixing"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Use tqdm for a nice progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (waveforms, labels) in enumerate(progress_bar):
        try:
            # Move data to device
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with autocast():
                    outputs = model(waveforms)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect for F1 score
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
            
            # Update learning rate if using OneCycleLR
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
                
        except RuntimeError as e:
            # Handle dimension issues
            if "size mismatch" in str(e) or "dimension" in str(e) or "shape" in str(e):
                logger.warning(f"Dimension error in batch {batch_idx}: {str(e)}")
                continue
            else:
                logger.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Update learning rate for other schedulers
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(f1)
        else:
            scheduler.step()
    
    return avg_loss, accuracy, f1

def validate(model, val_loader, criterion, device):
    """Validate model with robust error handling"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Use tqdm for a nice progress bar
    progress_bar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (waveforms, labels) in enumerate(progress_bar):
            try:
                # Move data to device
                waveforms, labels = waveforms.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect for F1 score
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def train_model(args):
    """Main training function with robust error handling"""
    try:
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        # Set device
        device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(args)
        
        logger.info(f"Training on {len(train_loader.dataset)} samples")
        logger.info(f"Validating on {len(val_loader.dataset)} samples")
        logger.info(f"Testing on {len(test_loader.dataset)} samples")
        
        # Determine number of emotions based on emotion set
        num_emotions = 8 if args.emotion_set == "full" else 4
        
        # Create model
        model = AdvancedSpeechEmotionModel(
            num_emotions=num_emotions,
            sample_rate=args.sample_rate,
            use_transformer=args.use_transformer,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            num_context_layers=args.num_context_layers,
            freeze_feature_extractor=args.freeze_feature_extractor
        )
        model.to(device)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * args.epochs
        
        if args.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs
            )
        elif args.lr_scheduler == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif args.lr_scheduler == "one_cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.learning_rate,
                total_steps=total_steps
            )
        else:
            scheduler = None
        
        # Mixed precision training
        scaler = GradScaler() if args.mixed_precision and device.type == "cuda" else None
        
        # Track best model
        best_val_f1 = 0.0
        best_epoch = 0
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(1, args.epochs + 1):
            # Training phase
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, 
                scheduler if args.lr_scheduler == "one_cycle" else None, 
                device, epoch, scaler
            )
            
            logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            # Validation phase
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
            
            logger.info(f"Epoch {epoch}/{args.epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Update learning rate
            if scheduler is not None and args.lr_scheduler != "one_cycle":
                if args.lr_scheduler == "cosine":
                    scheduler.step()
                elif args.lr_scheduler == "reduce_on_plateau":
                    scheduler.step(val_f1)
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                
                # Save model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "train_f1": train_f1,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "num_emotions": num_emotions,
                    "emotion_set": args.emotion_set,
                    "sample_rate": args.sample_rate,
                    "metadata": model.metadata
                }, os.path.join(args.output_dir, "best_model.pt"))
                
                logger.info(f"Saved best model with F1: {best_val_f1:.4f}")
            
            # Save checkpoint periodically
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "train_f1": train_f1,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "num_emotions": num_emotions,
                    "emotion_set": args.emotion_set,
                    "sample_rate": args.sample_rate,
                    "metadata": model.metadata
                }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        logger.info(f"Training completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
        
        # Evaluate on test set
        logger.info("Evaluating best model on test set...")
        
        # Load best model
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Test evaluation
        test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
        
        logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Create confusion matrix
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for waveforms, labels in tqdm(test_loader, desc="Generating confusion matrix"):
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Create class names based on emotion set
        class_names = list(EMOTION_MAPPINGS['simplified'].values()) if args.emotion_set == "simplified" else list(EMOTION_MAPPINGS['full'].values())
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        
        # Save test results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump({
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "confusion_matrix": cm.tolist(),
                "class_names": class_names
            }, f, indent=2)
        
        logger.info(f"Training complete with best val F1: {best_val_f1:.4f}, test F1: {test_f1:.4f}")
        logger.info(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train an advanced speech emotion recognition model")
    
    # Dataset parameters
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--emotion_set", type=str, default="full", choices=["full", "simplified"], help="Emotion set to use")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--max_length", type=int, default=48000, help="Maximum audio length in samples (3 seconds at 16kHz)")
    
    # Model parameters
    parser.add_argument("--use_transformer", action="store_true", help="Use Wav2Vec2 transformer if available")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--num_context_layers", type=int, default=4, help="Number of context transformer layers")
    parser.add_argument("--freeze_feature_extractor", action="store_true", help="Freeze feature extractor layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        choices=["cosine", "reduce_on_plateau", "one_cycle", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./models/advanced", help="Output directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    
    # Resource parameters
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    start_time = time.time()
    train_model(args)
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 