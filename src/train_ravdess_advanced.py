#!/usr/bin/env python3
"""
Advanced Training Script for RAVDESS Emotion Recognition
Uses cutting-edge techniques to train a high-accuracy emotion recognition model
"""

import os
import sys
import json
import yaml
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# Try to import SummaryWriter, but don't fail if not available
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Training will continue without TensorBoard logging.")

# Import our model and dataset
from ravdess_model import AdvancedSpeechEmotionRecognizer
from ravdess_dataset import RAVDESSDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def create_dataloaders(args):
    """
    Create train, validation, and test dataloaders
    """
    logger.info(f"Creating dataloaders from {args.dataset_root}")
    
    # Data augmentation for training
    train_transforms = None  # Will be handled within the dataset
    
    # Create datasets
    train_dataset = RAVDESSDataset(
        root_dir=args.dataset_root,
        split='train',
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        transforms=train_transforms,
        audio_only=args.audio_only,
        speech_only=args.speech_only,
        cache_waveforms=args.cache_waveforms,
        subset=args.emotion_subset
    )
    
    val_dataset = RAVDESSDataset(
        root_dir=args.dataset_root,
        split='val',
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        transforms=None,
        audio_only=args.audio_only,
        speech_only=args.speech_only,
        cache_waveforms=args.cache_waveforms,
        subset=args.emotion_subset
    )
    
    test_dataset = RAVDESSDataset(
        root_dir=args.dataset_root,
        split='test',
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        transforms=None,
        audio_only=args.audio_only,
        speech_only=args.speech_only,
        cache_waveforms=args.cache_waveforms,
        subset=args.emotion_subset
    )
    
    # Get the number of classes
    num_classes = RAVDESSDataset.get_num_classes(args.emotion_subset)
    logger.info(f"Number of emotion classes: {num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=RAVDESSDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=RAVDESSDataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=RAVDESSDataset.collate_fn
    )
    
    return train_loader, val_loader, test_loader, num_classes


def create_model(args, num_classes):
    """
    Create and initialize the model
    """
    logger.info(f"Creating model with {num_classes} emotion classes")
    
    model = AdvancedSpeechEmotionRecognizer(
        num_emotions=num_classes,
        wav2vec_model_name=args.wav2vec_model,
        freeze_feature_extractor=args.freeze_extractor,
        context_layers=args.context_layers,
        attention_heads=args.attention_heads,
        dropout_rate=args.dropout_rate,
        use_gender_branch=args.use_gender_branch,
        use_spectrogram_branch=args.use_spectrogram_branch
    )
    
    # Load from checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    
    # Move model to device
    model = model.to(args.device)
    
    return model


def get_optimizer(model, args):
    """
    Create optimizer and scheduler
    """
    # Split parameters into those that require gradient and those that don't
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = optim.SGD(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            momentum=0.9
        )
    
    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=args.cosine_t0 if args.cosine_t0 else args.epochs // 3, 
            T_mult=args.cosine_t_mult,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=args.min_lr
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_one_epoch(model, train_loader, optimizer, scaler, args, epoch):
    """
    Train the model for one epoch
    """
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Create tqdm progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        waveforms = batch['waveform'].to(args.device)
        emotion_targets = batch['emotion'].to(args.device)
        metadata = batch['metadata']
        
        # Extract gender targets if using gender branch
        if args.use_gender_branch:
            gender_targets = torch.tensor(
                [1 if m['gender'] == 'female' else 0 for m in metadata],
                dtype=torch.long
            ).to(args.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=args.use_amp):
            outputs = model(waveforms)
            emotion_logits = outputs['emotion_logits']
            
            # Emotion classification loss
            emotion_loss = F.cross_entropy(emotion_logits, emotion_targets)
            
            # Gender classification loss (if applicable)
            if args.use_gender_branch:
                gender_logits = outputs['gender_logits']
                gender_loss = F.cross_entropy(gender_logits, gender_targets)
                # Combined loss with weighting
                loss = emotion_loss + args.gender_loss_weight * gender_loss
            else:
                loss = emotion_loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if args.clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        
        # Update weights with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = emotion_logits.max(1)
        correct += predicted.eq(emotion_targets).sum().item()
        total += emotion_targets.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{epoch_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    # Calculate final metrics
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, args, epoch=None):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    # Description for progress bar
    desc = f"Epoch {epoch+1}/{args.epochs} [Val]" if epoch is not None else "Validation"
    
    # Create tqdm progress bar
    progress_bar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            waveforms = batch['waveform'].to(args.device)
            emotion_targets = batch['emotion'].to(args.device)
            metadata = batch['metadata']
            
            # Extract gender targets if using gender branch
            if args.use_gender_branch:
                gender_targets = torch.tensor(
                    [1 if m['gender'] == 'female' else 0 for m in metadata],
                    dtype=torch.long
                ).to(args.device)
            
            # Forward pass
            outputs = model(waveforms)
            emotion_logits = outputs['emotion_logits']
            
            # Emotion classification loss
            emotion_loss = F.cross_entropy(emotion_logits, emotion_targets)
            
            # Gender classification loss (if applicable)
            if args.use_gender_branch:
                gender_logits = outputs['gender_logits']
                gender_loss = F.cross_entropy(gender_logits, gender_targets)
                # Combined loss with weighting
                loss = emotion_loss + args.gender_loss_weight * gender_loss
            else:
                loss = emotion_loss
            
            # Update metrics
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = emotion_logits.max(1)
            correct += predicted.eq(emotion_targets).sum().item()
            total += emotion_targets.size(0)
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(emotion_targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{val_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
    
    # Calculate final metrics
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def test_model(model, test_loader, args):
    """
    Test the model on the test set and calculate metrics
    """
    # Same as validation, but on test set
    test_loss, test_accuracy, all_preds, all_targets = validate(model, test_loader, args)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_targets, all_preds)
    
    # Get emotion mapping
    emotion_mapping = RAVDESSDataset.get_emotion_mapping()
    
    # Generate class names
    class_names = [emotion_mapping[i] for i in range(len(emotion_mapping))]
    
    # Generate classification report
    report = classification_report(
        all_targets, 
        all_preds,
        target_names=class_names,
        digits=4
    )
    
    logger.info(f"Test accuracy: {test_accuracy:.2f}%")
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"Classification report:\n{report}")
    
    return test_loss, test_accuracy, cm, report


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_accuracy, args, metrics, is_best=False):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'args': vars(args),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model separately
    if is_best:
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save(checkpoint, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    
    # Save final model
    if epoch == args.epochs - 1:
        final_model_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(checkpoint, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")


def save_model_config(args, num_classes):
    """
    Save model configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create config dictionary
    config = {
        'model_type': 'AdvancedSpeechEmotionRecognizer',
        'num_classes': num_classes,
        'wav2vec_model': args.wav2vec_model,
        'sample_rate': args.sample_rate,
        'context_layers': args.context_layers,
        'attention_heads': args.attention_heads,
        'use_gender_branch': args.use_gender_branch,
        'use_spectrogram_branch': args.use_spectrogram_branch,
        'emotion_subset': args.emotion_subset,
        'training_args': vars(args)
    }
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Model configuration saved to {config_path}")


def train_model(args):
    """
    Train the model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create tensorboard writer if available
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(args)
    
    # Save model configuration
    save_model_config(args, num_classes)
    
    # Create model
    model = create_model(args, num_classes)
    
    # Print model summary
    logger.info(f"Model architecture:\n{model}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, args)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Initialize best validation accuracy
    best_val_accuracy = 0.0
    
    # Training loop
    logger.info("Starting training")
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, scaler, args, epoch)
        
        # Validate
        val_loss, val_accuracy, val_preds, val_targets = validate(model, val_loader, args, epoch)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Log to tensorboard if available
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
            # Add learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Check if this is the best model
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_accuracy, args, metrics, is_best)
    
    # Final evaluation on test set
    logger.info("Evaluating model on test set")
    test_loss, test_accuracy, confusion_matrix, classification_report = test_model(model, test_loader, args)
    
    # Log test metrics
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, args.epochs)
        writer.add_scalar('Accuracy/test', test_accuracy, args.epochs)
        
        # Close tensorboard writer
        writer.close()
    
    # Save final model info
    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_accuracy
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    logger.info(f"Final test accuracy: {test_accuracy:.2f}%")
    logger.info(f"Training completed. Models and metrics saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model on RAVDESS")
    
    # Dataset arguments
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to the RAVDESS dataset directory')
    parser.add_argument('--output_dir', type=str, default='../models/ravdess_advanced',
                        help='Output directory for saving models and logs')
    parser.add_argument('--emotion_subset', type=str, default=None, choices=['basic4', 'basic6', None],
                        help='Subset of emotions to use (basic4, basic6, or all)')
    parser.add_argument('--audio_only', action='store_true', default=True,
                        help='Only use audio-only files')
    parser.add_argument('--speech_only', action='store_true', default=True,
                        help='Only use speech files (not song)')
    parser.add_argument('--cache_waveforms', action='store_true', default=False,
                        help='Cache waveforms in memory for faster training')
    
    # Model arguments
    parser.add_argument('--wav2vec_model', type=str, default='facebook/wav2vec2-base',
                        help='Wav2Vec 2.0 model to use')
    parser.add_argument('--freeze_extractor', action='store_true', default=True,
                        help='Freeze the feature extractor part of the model')
    parser.add_argument('--context_layers', type=int, default=2,
                        help='Number of context transformer layers')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--use_gender_branch', action='store_true', default=True,
                        help='Use gender classification branch for multi-task learning')
    parser.add_argument('--gender_loss_weight', type=float, default=0.2,
                        help='Weight for gender classification loss')
    parser.add_argument('--use_spectrogram_branch', action='store_true', default=True,
                        help='Use spectrogram branch for additional features')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler to use')
    parser.add_argument('--cosine_t0', type=int, default=None,
                        help='T_0 parameter for cosine annealing scheduler')
    parser.add_argument('--cosine_t_mult', type=int, default=2,
                        help='T_mult parameter for cosine annealing scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--max_duration', type=float, default=5.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Print arguments
    logger.info("Training with the following parameters:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Train model
    train_model(args)


if __name__ == "__main__":
    main() 