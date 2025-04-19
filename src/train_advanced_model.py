#!/usr/bin/env python3
"""
Training script for the Advanced Speech Emotion Recognition Model
Includes data augmentation, class balancing, and effective training strategies
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
import logging
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import torchaudio
import sys
import traceback

# Add src directory to path to ensure imports work properly
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

# Configure logging first before other imports that might use logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the model training functions
try:
    from train_one_epoch import train_one_epoch, validate
    logger.info("Successfully imported training functions")
except ImportError as e:
    logger.warning(f"Could not import train_one_epoch: {e}")
    # Will use internal functions instead

# Import the raw dataset
try:
    from ravdess_raw_dataset import RAVDESSRawDataset
    logger.info("Successfully imported RAVDESSRawDataset")
    DatasetClass = RAVDESSRawDataset
except ImportError as e:
    logger.warning(f"Could not import RAVDESSRawDataset: {e}")
    # Fall back to other dataset options
    try:
        from ravdess_dataset import RAVDESSDataset
        logger.info("Successfully imported RAVDESSDataset")
        DatasetClass = RAVDESSDataset
    except ImportError as e:
        logger.error(f"Could not import any dataset classes: {e}")
        logger.error("Make sure ravdess_raw_dataset.py or ravdess_dataset.py is in your path.")
        raise

# Try to import our model
try:
    from advanced_emotion_model import AdvancedEmotionRecognitionModel
    logger.info("Successfully imported AdvancedEmotionRecognitionModel")
except ImportError as e:
    logger.error(f"Could not import AdvancedEmotionRecognitionModel: {e}")
    logger.error("Make sure advanced_emotion_model.py is in your path.")
    raise

# Import torch.nn.functional for our custom dataset
import torch.nn.functional as F

# Check if tensorboard is available
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")

class AudioAugmenter:
    """Audio augmentation techniques for training"""
    def __init__(self, 
                 sample_rate=16000,
                 time_shift_prob=0.5, 
                 noise_prob=0.5,
                 noise_level=(0.005, 0.02),
                 pitch_shift_prob=0.3,
                 pitch_shift_range=(-2, 2),
                 speed_perturb_prob=0.3,
                 speed_perturb_range=(0.9, 1.1),
                 time_stretch_prob=0.3,
                 time_stretch_range=(0.8, 1.2)):
        self.sample_rate = sample_rate
        self.time_shift_prob = time_shift_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_perturb_range = speed_perturb_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        
        # Disable advanced augmentations for now as they're causing shape issues
        self.has_torchaudio = False
        logger.warning("Disabling advanced torchaudio augmentations due to compatibility issues")
    
    def __call__(self, waveform):
        """Apply augmentations to waveform"""
        # Time shift
        if random.random() < self.time_shift_prob:
            waveform = self.apply_time_shift(waveform)
        
        # Add noise
        if random.random() < self.noise_prob:
            waveform = self.add_noise(waveform)
        
        # We'll skip the advanced transformations for now as they're causing issues
        # with tensor shapes
        return waveform
    
    def apply_time_shift(self, waveform):
        """Shift audio in time"""
        shift_amount = int(random.random() * waveform.size(-1) * 0.2)  # Shift up to 20%
        return torch.roll(waveform, shifts=shift_amount, dims=-1)
    
    def add_noise(self, waveform):
        """Add random noise to waveform"""
        noise_level = random.uniform(*self.noise_level)
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train advanced emotion recognition model')
    
    # Dataset parameters
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of RAVDESS dataset')
    parser.add_argument('--audio_only', action='store_true', help='Use only audio files')
    parser.add_argument('--speech_only', action='store_true', help='Use only speech files')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--max_duration', type=float, default=5.0, help='Maximum audio duration in seconds')
    parser.add_argument('--cache_waveforms', action='store_true', help='Cache waveforms in memory')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--transformer_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=3.0, help='Gradient clipping norm')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Augmentation parameters
    parser.add_argument('--augment', action='store_true', help='Use audio augmentation')
    parser.add_argument('--time_shift_prob', type=float, default=0.5, help='Time shift probability')
    parser.add_argument('--noise_prob', type=float, default=0.5, help='Noise addition probability')
    parser.add_argument('--use_specaugment', action='store_true', help='Use SpecAugment for spectrogram augmentation')
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Model saving interval (epochs)')
    
    # Advanced training strategies
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'multistep'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision')
    
    return parser.parse_args()


def create_dataloaders(args):
    """Create train, validation, and test dataloaders"""
    logger.info(f"Creating dataloaders from {args.dataset_root}")
    
    # Create augmenter if needed
    train_transforms = None
    if args.augment:
        train_transforms = AudioAugmenter(
            sample_rate=args.sample_rate,
            time_shift_prob=args.time_shift_prob,
            noise_prob=args.noise_prob
        )
    
    # Create datasets
    try:
        train_dataset = DatasetClass(
            root_dir=args.dataset_root,
            split='train',
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            transforms=train_transforms,
            audio_only=args.audio_only,
            speech_only=args.speech_only,
            cache_waveforms=args.cache_waveforms
        )
        
        val_dataset = DatasetClass(
            root_dir=args.dataset_root,
            split='val',
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            transforms=None,
            audio_only=args.audio_only,
            speech_only=args.speech_only,
            cache_waveforms=args.cache_waveforms
        )
        
        test_dataset = DatasetClass(
            root_dir=args.dataset_root,
            split='test',
            sample_rate=args.sample_rate,
            max_duration=args.max_duration,
            transforms=None,
            audio_only=args.audio_only,
            speech_only=args.speech_only,
            cache_waveforms=args.cache_waveforms
        )
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        logger.error(f"Training dataset is empty. Check the path: {args.dataset_root}")
        raise ValueError("Training dataset is empty. Cannot proceed with training.")
    
    if len(val_dataset) == 0:
        logger.warning(f"Validation dataset is empty. Will use a portion of training data for validation.")
        # Create a split from training data for validation if needed
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    if len(test_dataset) == 0:
        logger.warning(f"Test dataset is empty. Will use a portion of training data for testing.")
        # Only create test set from training if validation was also empty
        if len(val_dataset) < len(train_dataset) * 0.1:
            train_size = int(0.9 * len(train_dataset))
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, test_size]
            )
    
    # Calculate samples per class for balanced loss with proper error handling
    samples_per_class = None
    if len(train_dataset) > 0:
        try:
            # Get emotion targets from dataset with explicit type checking
            train_targets = []
            for i in range(min(len(train_dataset), 1000)):  # Limit to 1000 samples for efficiency
                item = train_dataset[i]
                emotion = item['emotion']
                # Convert tensor to int if needed
                if isinstance(emotion, torch.Tensor):
                    emotion = emotion.item()
                # Ensure emotion is an integer
                train_targets.append(int(emotion))
            
            # Make sure we have at least some samples
            if len(train_targets) > 0:
                # Convert to tensor and ensure it's long type for bincount
                train_targets_tensor = torch.tensor(train_targets, dtype=torch.long)
                
                # Find max class index to ensure bincount has proper size
                max_class = max(train_targets)
                
                # Use bincount with explicit long tensor to avoid Float error
                samples_per_class = torch.bincount(
                    input=train_targets_tensor, 
                    minlength=max_class+1
                ).float()  # Convert to float after bincount
                
                # Log the distribution
                logger.info(f"Samples per class in training set: {samples_per_class}")
                
                # Check for empty classes and handle them
                if torch.any(samples_per_class == 0):
                    logger.warning(f"Some classes have zero samples: {samples_per_class}")
                    # Set minimum count to 1 to avoid division by zero later
                    samples_per_class = torch.clamp(samples_per_class, min=1)
            else:
                logger.warning("No samples found for class distribution calculation")
        except Exception as e:
            logger.warning(f"Error calculating class distribution: {e}")
            logger.warning("Using equal class weights")
            logger.warning(traceback.format_exc())  # Add traceback for debugging
    else:
        logger.warning("Training dataset is empty. Using equal class weights.")
    
    # Create dataloaders - use smaller batch size if dataset is small
    actual_batch_size = min(args.batch_size, max(1, len(train_dataset) // 4)) if len(train_dataset) > 0 else args.batch_size
    if actual_batch_size != args.batch_size:
        logger.warning(f"Reduced batch size to {actual_batch_size} due to small dataset size")
    
    collate_fn = DatasetClass.collate_fn if hasattr(DatasetClass, 'collate_fn') else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # Avoid issues with small batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    return train_loader, val_loader, test_loader, samples_per_class


def get_optimizer(model, args):
    """Get optimizer and learning rate scheduler"""
    # Prepare parameter groups with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    # Use AdamW optimizer with parameter groups
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create learning rate scheduler
    if args.scheduler == 'cosine':
        # Cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=args.learning_rate * 0.01
        )
    elif args.scheduler == 'plateau':
        # Reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif args.scheduler == 'multistep':
        # Multi-step LR decay
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
    
    # Create warmup scheduler if needed
    if args.warmup_epochs > 0:
        try:
            from torch.optim.lr_scheduler import LambdaLR
            
            # Define warmup scheduler
            def warmup_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return epoch / args.warmup_epochs
                return 1.0
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            
            # Return both schedulers
            return optimizer, (warmup_scheduler, scheduler)
        except ImportError:
            logger.warning("Warmup scheduler not available. Using base scheduler only.")
            return optimizer, scheduler
    
    return optimizer, scheduler


def _fallback_train_one_epoch(model, train_loader, optimizer, scaler, samples_per_class, args, epoch):
    """Fallback training function for one epoch with basic error handling"""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Get data
            waveforms = batch['waveform'].to(args.device)
            emotion_targets = batch['emotion'].to(args.device)
            
            # Ensure emotion targets are long
            if emotion_targets.dtype != torch.long:
                emotion_targets = emotion_targets.long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                waveform=waveforms, 
                emotion_targets=emotion_targets,
                apply_mixup=False,  # Disable mixup in fallback for safety
                apply_augmentation=False  # Disable augmentation for safety
            )
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs['emotion_logits'].max(1)
            total += emotion_targets.size(0)
            correct += predicted.eq(emotion_targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Calculate final metrics
    avg_loss = epoch_loss / max(1, len(train_loader))
    accuracy = 100. * correct / max(1, total)
    
    return avg_loss, accuracy


def _fallback_validate(model, val_loader, args, epoch=None):
    """Fallback validation function with basic error handling"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    # Description for progress bar
    desc = f"Epoch {epoch+1}/{args.epochs} [Val]" if epoch is not None else "Validation"
    
    # Create progress bar
    progress_bar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Get data
                waveforms = batch['waveform'].to(args.device)
                emotion_targets = batch['emotion'].to(args.device)
                
                # Ensure emotion targets are long
                if emotion_targets.dtype != torch.long:
                    emotion_targets = emotion_targets.long()
                
                # Forward pass
                outputs = model(
                    waveform=waveforms, 
                    emotion_targets=emotion_targets,
                    apply_mixup=False,
                    apply_augmentation=False
                )
                loss = outputs['loss']
                
                # Update metrics
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs['emotion_logits'].max(1)
                total += emotion_targets.size(0)
                correct += predicted.eq(emotion_targets).sum().item()
                
                # Store predictions and targets for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(emotion_targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{val_loss/(batch_idx+1):.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Calculate final metrics
    avg_loss = val_loss / max(1, len(val_loader))
    accuracy = 100. * correct / max(1, total)
    
    return avg_loss, accuracy, all_preds, all_targets


# Choose appropriate training functions
if 'train_one_epoch' not in globals():
    logger.warning("Using fallback training functions")
    train_one_epoch = _fallback_train_one_epoch
    validate = _fallback_validate


def test_model(model, test_loader, args, output_dir):
    """Test the model on the test set"""
    logger.info("Testing model on test set...")
    
    # Validate on test set
    test_loss, test_acc, all_preds, all_targets = validate(model, test_loader, args)
    
    logger.info(f"Test accuracy: {test_acc:.2f}%")
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Get classification report
    emotion_names = DatasetClass.get_emotion_mapping()
    target_names = [emotion_names[i] for i in range(len(emotion_names))]
    report = classification_report(all_targets, all_preds, target_names=target_names)
    logger.info(f"Classification Report:\n{report}")
    
    # Save test results
    test_results = {
        'test_accuracy': test_acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    return test_acc, cm, report


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, output_dir, is_best=False, filename=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, os.path.join(output_dir, filename or 'latest_checkpoint.pt'))
    
    # Save epoch checkpoint if needed
    if (epoch + 1) % args.save_interval == 0:
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_ep{epoch+1}.pt'))
    
    # Save best model
    if is_best:
        torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
        
        # Save model only (for inference)
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_weights.pt'))


def train_model(args):
    """Train the model"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Additional settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"advanced_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log command-line arguments
    logger.info(f"Command-line arguments: {args}")
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        args.device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")
    
    # Create dataloaders with error handling
    try:
        train_loader, val_loader, test_loader, samples_per_class = create_dataloaders(args)
        logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise
    
    # Create model with error handling
    try:
        # Get number of emotions from dataset or config
        num_emotions = 8  # Default
        if hasattr(DatasetClass, 'get_num_classes'):
            try:
                num_emotions = DatasetClass.get_num_classes()
                logger.info(f"Using {num_emotions} emotion classes from dataset")
            except Exception as e:
                logger.warning(f"Could not get num_classes from dataset: {e}, using default 8 classes")
        
        # Create model parameters dictionary for better error tracking
        model_params = {
            'num_emotions': num_emotions,
            'sample_rate': args.sample_rate,
            'feature_dim': args.feature_dim,
            'hidden_dim': args.hidden_dim,
            'transformer_layers': args.transformer_layers,
            'transformer_heads': args.transformer_heads,
            'dropout': args.dropout
        }
        
        # Handle samples_per_class - either None or a valid tensor
        if samples_per_class is not None:
            if torch.is_tensor(samples_per_class) and len(samples_per_class) > 0:
                model_params['samples_per_class'] = samples_per_class
                logger.info(f"Using weighted loss with class distribution: {samples_per_class}")
            else:
                logger.warning("Invalid samples_per_class, using uniform weighting")
                model_params['samples_per_class'] = None
        else:
            logger.info("Using uniform class weighting (no samples_per_class provided)")
            model_params['samples_per_class'] = None
        
        # Create the model with expanded parameters
        logger.info(f"Creating model with parameters: {model_params}")
        model = AdvancedEmotionRecognitionModel(**model_params).to(args.device)
        
        # Set mixup alpha if using mixup
        if args.use_mixup:
            model.mixup_alpha = args.mixup_alpha
            logger.info(f"Set mixup alpha to {args.mixup_alpha}")
        
        # Print model summary
        logger.info(f"Model architecture:\n{model}")
        
        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, args)
    
    # Unpack schedulers if warmup is used
    warmup_scheduler = None
    if isinstance(scheduler, tuple):
        warmup_scheduler, scheduler = scheduler
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs')) if TENSORBOARD_AVAILABLE else None
    
    # Initialize best validation accuracy
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Training loop
    logger.info("Starting training...")
    
    try:
        for epoch in range(args.epochs):
            # Train for one epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, samples_per_class, args, epoch
            )
            
            # Update learning rate with warmup if needed
            if warmup_scheduler is not None and epoch < args.warmup_epochs:
                warmup_scheduler.step()
            elif args.scheduler == 'cosine' or args.scheduler == 'multistep':
                scheduler.step()
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, args, epoch)
            
            # Update plateau scheduler if used
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            
            # Log results
            logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                       f"LR: {optimizer.param_groups[0]['lr']:.7f}")
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
                
                # Log confusion matrix every 5 epochs
                if (epoch + 1) % 5 == 0:
                    try:
                        cm = confusion_matrix(val_targets, val_preds)
                        fig = plt.figure(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                        writer.add_figure('Confusion Matrix', fig, epoch)
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Failed to create confusion matrix: {e}")
            
            # Check if this is the best model
            is_best = val_acc > best_val_accuracy
            if is_best:
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                early_stop_counter = 0
                
                # Log best model info
                logger.info(f"New best model with val accuracy: {val_acc:.2f}%")
            else:
                early_stop_counter += 1
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc, output_dir, is_best
            )
            
            # Early stopping
            if early_stop_counter >= args.early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model for testing
        try:
            best_checkpoint_path = os.path.join(output_dir, 'best_model.pt')
            if os.path.exists(best_checkpoint_path):
                best_checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
                model.load_state_dict(best_checkpoint['model_state_dict'])
                logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']+1} "
                          f"with val accuracy: {best_checkpoint['val_acc']:.2f}%")
            else:
                logger.warning("Best model checkpoint not found. Using the latest model for testing.")
        except Exception as e:
            logger.error(f"Failed to load best model: {e}")
        
        # Test model
        test_acc, confusion_matrix, classification_report = test_model(model, test_loader, args, output_dir)
        
        # Close TensorBoard writer
        if writer:
            writer.close()
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
        logger.info(f"Test accuracy: {test_acc:.2f}%")
        
        return model, test_acc
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save the current model before exiting
        save_checkpoint(
            model, optimizer, scheduler, epoch, float('inf'), 0.0, 
            output_dir, is_best=False, filename='interrupted_checkpoint.pt'
        )
        logger.info(f"Saved interrupted model to {os.path.join(output_dir, 'interrupted_checkpoint.pt')}")
        return model, 0.0
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(traceback.format_exc())  # Print full traceback for debugging
        raise


def main():
    """Main function"""
    args = parse_args()
    
    # Set defaults for missing arguments to prevent AttributeError
    if not hasattr(args, 'use_mixup'):
        args.use_mixup = False
    if not hasattr(args, 'use_specaugment'):
        args.use_specaugment = False
    
    try:
        model, test_acc = train_model(args)
        logger.info(f"Training completed. Test accuracy: {test_acc:.2f}%")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(traceback.format_exc())  # Print full traceback for debugging
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 