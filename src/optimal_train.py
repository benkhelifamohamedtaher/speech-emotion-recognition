#!/usr/bin/env python3
"""
Optimized Training Script for Speech Emotion Recognition
Designed to achieve state-of-the-art results on the RAVDESS dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import logging
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Import our custom dataset and model
from optimal_dataset import create_dataloaders, EMOTION_DICT, SIMPLIFIED_EMOTIONS
from optimal_model import OptimalSpeechEmotionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_lr_scheduler(optimizer, args, num_training_steps):
    """Create learning rate scheduler based on arguments"""
    if args.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer, 
            T_max=num_training_steps, 
            eta_min=args.min_lr
        )
    elif args.lr_scheduler == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.min_lr / args.learning_rate,
            total_iters=num_training_steps
        )
    elif args.lr_scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.lr_scheduler == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            threshold=0.01,
            threshold_mode="abs",
            verbose=True
        )
    elif args.lr_scheduler == "one_cycle":
        from torch.optim.lr_scheduler import OneCycleLR
        return OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=num_training_steps,
            pct_start=0.2,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=10.0,
            final_div_factor=100.0
        )
    else:
        logger.warning(f"Unknown scheduler: {args.lr_scheduler}. Using constant learning rate.")
        return None

class EarlyStopping:
    """Early stopping implementation to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'min' for loss, 'max' for accuracy or F1

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return True
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return True
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return False

def validate(model, val_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Validation")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs['emotion_logits'], targets)
            
            # Calculate metrics
            val_loss += loss.item()
            _, predicted = torch.max(outputs['emotion_logits'], 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for F1 score
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total
    val_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return val_loss, val_acc, val_f1, all_preds, all_targets

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, args):
    """Train model for one epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs['emotion_logits'], targets)
        
        # Backward and optimize
        loss.backward()
        
        # Gradient clipping
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        # Update scheduler if using one_cycle policy
        if args.lr_scheduler == "one_cycle" and scheduler is not None:
            scheduler.step()
        
        # Calculate metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs['emotion_logits'], 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Store predictions and targets for F1 score
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{train_loss / (batch_idx + 1):.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })
    
    # Calculate metrics
    train_loss /= len(train_loader)
    train_acc = 100.0 * correct / total
    train_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return train_loss, train_acc, train_f1

def test_model(model, test_loader, criterion, device, emotion_labels, output_dir):
    """Test model on test set and save results"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs['emotion_logits'], targets)
            
            # Calculate metrics
            test_loss += loss.item()
            _, predicted = torch.max(outputs['emotion_logits'], 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Print results
    logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1: {test_f1:.4f}")
    
    # Generate classification report
    class_report = classification_report(all_targets, all_preds, target_names=emotion_labels, digits=4)
    logger.info(f"\nClassification Report:\n{class_report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Save classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(class_report)
    
    # Save all results
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "targets": all_targets
    }
    
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return test_loss, test_acc, test_f1

def plot_training_history(history, output_dir):
    """Plot training and validation metrics"""
    # Create figures directory
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "loss.png"))
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "accuracy.png"))
    
    # Plot F1 score
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "f1_score.png"))
    
    # Plot learning rate
    if 'learning_rate' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(history['learning_rate'])
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, "learning_rate.png"))

def train(args):
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args for reproducibility
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    emotion_type = "simplified" if args.simplified else "full"
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        duration=args.duration,
        emotion_type=emotion_type,
        augment=args.augment,
        feature_type=args.feature_type,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True
    )
    
    logger.info(f"Dataset loaded with {len(train_loader.dataset)} training, "
               f"{len(val_loader.dataset)} validation, and {len(test_loader.dataset)} test samples")
    
    # Set emotion labels based on emotion type
    if emotion_type == "simplified":
        num_emotions = 4
        emotion_labels = list(SIMPLIFIED_EMOTIONS.keys())
    else:
        num_emotions = 8
        emotion_labels = list(EMOTION_DICT.values())
    
    # Create model
    model = OptimalSpeechEmotionModel(
        num_emotions=num_emotions,
        input_channels=1,
        sample_rate=args.sample_rate,
        hidden_size=args.hidden_size,
        num_transformer_layers=args.num_transformer_layers,
        num_attention_heads=args.num_attention_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        input_type=args.feature_type
    )
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    
    # Define loss function
    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        from torch.nn import functional as F
        
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2, reduction='mean'):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
            
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    # Define optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Get total training steps
    num_training_steps = len(train_loader) * args.epochs
    
    # Define learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, args, num_training_steps)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping, mode='max')
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    
    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, args
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler if not OneCycleLR
        if args.lr_scheduler != "one_cycle" and scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_f1)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        logger.info(f"Epoch {epoch}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, F1: {train_f1:.4f} | "
                   f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, F1: {val_f1:.4f} | "
                   f"LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'config': {
                    'num_emotions': num_emotions,
                    'sample_rate': args.sample_rate,
                    'feature_type': args.feature_type,
                    'emotion_type': emotion_type,
                    'hidden_size': args.hidden_size,
                    'num_transformer_layers': args.num_transformer_layers,
                    'num_attention_heads': args.num_attention_heads,
                    'dropout': args.dropout
                },
                'metadata': model.metadata
            }, os.path.join(output_dir, "best_model.pt"))
            
            logger.info(f"Saved best model with validation F1: {best_val_f1:.4f}")
        
        # Check for early stopping
        if early_stopping(val_f1):
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save checkpoint every few epochs
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'config': {
                    'num_emotions': num_emotions,
                    'sample_rate': args.sample_rate,
                    'feature_type': args.feature_type,
                    'emotion_type': emotion_type,
                    'hidden_size': args.hidden_size,
                    'num_transformer_layers': args.num_transformer_layers,
                    'num_attention_heads': args.num_attention_heads,
                    'dropout': args.dropout
                },
                'metadata': model.metadata
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Load best model for testing
    best_model_path = os.path.join(output_dir, "best_model.pt")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded best model from epoch {best_epoch} with validation F1: {best_val_f1:.4f}")
    
    # Test model
    test_loss, test_acc, test_f1 = test_model(model, test_loader, criterion, device, emotion_labels, output_dir)
    
    # Print final results
    logger.info(f"Training completed!")
    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1: {test_f1:.4f}")
    
    return {
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1
    }

def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Optimal Speech Emotion Recognition Training")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed dataset directory")
    parser.add_argument("--output_dir", type=str, default="./models/optimal", help="Directory to save models and results")
    parser.add_argument("--simplified", action="store_true", help="Use simplified emotions (4 classes)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration", type=float, default=3.0, help="Audio duration in seconds")
    parser.add_argument("--feature_type", type=str, default="waveform", choices=["waveform", "melspec", "mfcc"], 
                        help="Type of features to use")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_transformer_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=1024, help="Feed-forward dimension in transformer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], 
                        help="Optimizer to use")
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["cross_entropy", "focal"], 
                        help="Loss function to use")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    parser.add_argument("--lr_scheduler", type=str, default="one_cycle", 
                        choices=["cosine", "linear", "step", "plateau", "one_cycle", "none"], 
                        help="Learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR scheduler")
    parser.add_argument("--early_stopping", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--save_interval", type=int, default=5, help="Save model every N epochs")
    
    args = parser.parse_args()
    
    # Print args
    logger.info("Training arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Start training
    start_time = time.time()
    results = train(args)
    end_time = time.time()
    
    # Print training time
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    return results

if __name__ == "__main__":
    main() 