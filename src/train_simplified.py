#!/usr/bin/env python3
"""
Simplified training script for speech emotion recognition.
Focuses on being error-resistant and reliable over feature-rich.
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
import logging
import json
from datetime import datetime
from pathlib import Path
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path to ensure imports work properly
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Try to import our modules, with fallbacks
try:
    from ravdess_raw_dataset import RAVDESSRawDataset as Dataset
    logger.info("Using RAVDESSRawDataset")
except ImportError:
    try:
        from ravdess_dataset import RAVDESSDataset as Dataset
        logger.info("Using RAVDESSDataset")
    except ImportError:
        logger.error("Could not import any dataset. Make sure dataset files are in your path.")
        sys.exit(1)

try:
    from advanced_emotion_model import AdvancedEmotionRecognitionModel
    logger.info("Using AdvancedEmotionRecognitionModel")
except ImportError:
    logger.error("Could not import model. Make sure advanced_emotion_model.py is in your path.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments with safe defaults"""
    parser = argparse.ArgumentParser(description='Train emotion recognition model with safe defaults')
    
    # Dataset parameters
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of RAVDESS dataset')
    parser.add_argument('--audio_only', action='store_true', help='Use only audio files')
    parser.add_argument('--speech_only', action='store_true', help='Use only speech files')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loader workers (0 for main thread)')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def create_datasets(args):
    """Create datasets with robust error handling"""
    logger.info(f"Creating datasets from {args.dataset_root}")
    
    try:
        # Create train dataset
        train_dataset = Dataset(
            root_dir=args.dataset_root,
            split='train',
            sample_rate=args.sample_rate,
            audio_only=args.audio_only,
            speech_only=args.speech_only
        )
        logger.info(f"Created training dataset with {len(train_dataset)} samples")
        
        # Create validation dataset
        val_dataset = Dataset(
            root_dir=args.dataset_root,
            split='val',
            sample_rate=args.sample_rate,
            audio_only=args.audio_only,
            speech_only=args.speech_only
        )
        logger.info(f"Created validation dataset with {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def create_dataloaders(train_dataset, val_dataset, args):
    """Create data loaders with fallbacks for empty datasets"""
    # Handle empty datasets
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Cannot proceed.")
        sys.exit(1)
    
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty. Using a subset of training data.")
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Determine class distribution for balanced loss
    class_counts = {}
    for i in range(min(100, len(train_dataset))):  # Sample for efficiency
        try:
            item = train_dataset[i]
            emotion = int(item['emotion'].item() if torch.is_tensor(item['emotion']) else item['emotion'])
            class_counts[emotion] = class_counts.get(emotion, 0) + 1
        except Exception as e:
            logger.warning(f"Error processing item {i}: {e}")
    
    logger.info(f"Class distribution (sampled): {class_counts}")
    
    # Create data loaders with minimal settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=getattr(Dataset, 'collate_fn', None)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=getattr(Dataset, 'collate_fn', None)
    )
    
    return train_loader, val_loader

def create_model(num_classes=8):
    """Create model with minimal parameters"""
    try:
        model = AdvancedEmotionRecognitionModel(
            num_emotions=num_classes,
            feature_dim=256,
            hidden_dim=512,
            transformer_layers=4,
            transformer_heads=8,
            dropout=0.2,
            samples_per_class=None  # No class weighting for stability
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        return model, device
    
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch with maximum error resistance"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            # Get data
            waveforms = batch['waveform'].to(device)
            emotion_targets = batch['emotion']
            
            # Ensure targets are long tensors
            if emotion_targets.dtype != torch.long:
                emotion_targets = emotion_targets.long()
            emotion_targets = emotion_targets.to(device)
            
            # Forward pass with safe parameter passing
            optimizer.zero_grad()
            outputs = model(waveform=waveforms, emotion_targets=emotion_targets,
                           apply_mixup=False, apply_augmentation=False)
            
            # Compute loss
            loss = outputs['loss']
            
            # Handle NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Skipping batch {batch_idx} - NaN/Inf loss detected")
                continue
                
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predicted = outputs['emotion_logits'].argmax(dim=1)
            batch_total = emotion_targets.size(0)
            batch_correct = (predicted == emotion_targets).sum().item()
            
            # Update stats
            running_loss += loss.item()
            total += batch_total
            correct += batch_correct
            
            # Logging
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, '
                            f'Loss: {running_loss/(batch_idx+1):.4f}, '
                            f'Acc: {100.0*correct/total:.2f}%')
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    return running_loss / max(1, len(train_loader)), 100.0 * correct / max(1, total)

def validate(model, val_loader, device):
    """Validate model with error handling"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Get data
                waveforms = batch['waveform'].to(device)
                emotion_targets = batch['emotion']
                
                # Ensure targets are long tensors
                if emotion_targets.dtype != torch.long:
                    emotion_targets = emotion_targets.long()
                emotion_targets = emotion_targets.to(device)
                
                # Forward pass
                outputs = model(waveform=waveforms, emotion_targets=emotion_targets,
                               apply_mixup=False, apply_augmentation=False)
                
                # Compute loss
                loss = outputs['loss']
                
                # Handle NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Calculate accuracy
                predicted = outputs['emotion_logits'].argmax(dim=1)
                batch_total = emotion_targets.size(0)
                batch_correct = (predicted == emotion_targets).sum().item()
                
                # Update stats
                running_loss += loss.item()
                total += batch_total
                correct += batch_correct
            
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return running_loss / max(1, len(val_loader)), 100.0 * correct / max(1, total)

def save_checkpoint(model, optimizer, epoch, output_dir, filename='checkpoint.pt'):
    """Save model checkpoint with error handling"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(output_dir, filename))
        logger.info(f"Saved checkpoint to {os.path.join(output_dir, filename)}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")

def main():
    """Main training function with robust error handling"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"simple_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    try:
        # Create datasets and data loaders
        train_dataset, val_dataset = create_datasets(args)
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args)
        
        # Create model
        model, device = create_model()
        logger.info(f"Using device: {device}")
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        
        for epoch in range(args.epochs):
            try:
                # Train for one epoch
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
                logger.info(f'Epoch: {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                
                # Validate
                val_loss, val_acc = validate(model, val_loader, device)
                logger.info(f'Epoch: {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save checkpoint
                save_checkpoint(model, optimizer, epoch, output_dir, f'checkpoint_ep{epoch+1}.pt')
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(model, optimizer, epoch, output_dir, 'best_model.pt')
                    logger.info(f'New best model with validation accuracy: {val_acc:.2f}%')
            
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                save_checkpoint(model, optimizer, epoch, output_dir, 'interrupted.pt')
                break
            
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {e}")
                logger.error(traceback.format_exc())
                save_checkpoint(model, optimizer, epoch, output_dir, f'error_ep{epoch+1}.pt')
                continue
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 