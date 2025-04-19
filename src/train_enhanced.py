#!/usr/bin/env python3
"""
Enhanced training script for Speech Emotion Recognition with the improved model architecture.
Includes advanced training techniques like:
- Learning rate scheduling
- Mixed precision training
- Gradient accumulation
- EMA model averaging
- Advanced metrics tracking
"""

import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import torch.nn.functional as F

# Try importing optional modules with fallbacks
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    warnings.warn("torch.cuda.amp not available, mixed precision training will be disabled")
    # Define placeholder for older PyTorch versions
    class GradScaler:
        def __init__(self, *args, **kwargs):
            warnings.warn("Mixed precision not available")
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
    
    def autocast(*args, **kwargs):
        class DummyContext:
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass
        return DummyContext()

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    warnings.warn("TensorBoard not available, logging will be limited")
    class SummaryWriter:
        def __init__(self, log_dir):
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Import model with error handling
try:
    from model_enhanced import SpeechEmotionRecognitionModelEnhanced
except Exception as e:
    print(f"Error importing model: {e}")
    print("Please check model_enhanced.py for issues")
    raise

# Import data utils
try:
    from data_utils import create_dataloaders
except ImportError:
    print("data_utils.py not found or contains errors")
    raise


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_audio_transform(waveform):
    """Audio transformations for training data"""
    # Add small amount of Gaussian noise
    waveform = waveform + torch.randn_like(waveform) * 0.005
    
    # Random gain adjustment
    gain = torch.FloatTensor(1).uniform_(0.8, 1.2)
    waveform = waveform * gain
    
    # Time stretching (simple version)
    stretch_factor = torch.FloatTensor(1).uniform_(0.9, 1.1)
    length = waveform.shape[-1]
    new_length = int(length * stretch_factor)
    if new_length > length:
        # Stretch (interpolate)
        waveform = F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
        # Trim to original length
        offset = torch.randint(0, max(1, new_length - length), (1,))
        waveform = waveform[..., offset:offset + length]
    else:
        # Shrink (interpolate)
        waveform = F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
        # Pad to original length
        pad_size = length - new_length
        left_pad = torch.randint(0, max(1, pad_size), (1,))
        right_pad = pad_size - left_pad
        waveform = F.pad(waveform, (left_pad, right_pad))
    
    return waveform

def val_audio_transform(waveform):
    """Audio transformations for validation data"""
    # Simple normalization for validation
    return waveform

def create_dataloaders(args):
    """Create training and validation dataloaders"""
    
    # Load dataset
    if args.dataset_root is None:
        raise ValueError("Dataset root must be specified")
    
    if hasattr(args, 'augment') and args.augment:
        # Use pre-defined transforms (already defined outside for proper pickling)
        train_transform = train_audio_transform
        val_transform = val_audio_transform
    else:
        # No augmentation
        train_transform = val_audio_transform
        val_transform = val_audio_transform


def train_enhanced(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        target_sr=args.sample_rate,
        max_length=args.max_length,
        apply_augmentation=args.augment,
        num_workers=args.num_workers
    )
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(valid_loader.dataset)} samples")
    
    # Create model
    model = SpeechEmotionRecognitionModelEnhanced(
        num_emotions=4, 
        freeze_encoder=args.freeze_encoder,
        model_name=args.model_name
    )
    model.to(device)
    
    # Initialize EMA model averaging
    if args.use_ema:
        ema = EMA(model, decay=0.998)
    
    # Define loss functions
    emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    vad_criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer with weight decay
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
    
    # Check if we're unfreezing the encoder after a certain number of epochs
    if args.unfreeze_encoder_epoch > 0:
        # Store encoder parameters for later unfreezing
        encoder_params = list(model.encoder.parameters())
        initial_freeze_state = args.freeze_encoder
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    
    if args.lr_scheduler == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps
        )
    elif args.lr_scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
    else:
        scheduler = None
    
    # Initialize mixed precision training
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Unfreeze encoder if needed
        if args.unfreeze_encoder_epoch > 0 and epoch == args.unfreeze_encoder_epoch and initial_freeze_state:
            print("Unfreezing encoder weights")
            for param in encoder_params:
                param.requires_grad = True
                
            # Update optimizer with unfrozen parameters
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
            
            # Create new optimizer for unfrozen parameters
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate * 0.1,  # Lower learning rate for fine-tuning
                eps=1e-8
            )
        
        # Train one epoch
        model.train()
        train_loss = 0.0
        train_emotion_preds = []
        train_emotion_labels = []
        
        # Initialize batch accumulation
        accum_loss = 0.0
        accum_iter = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            waveforms = batch['waveform'].to(device)
            emotion_labels = batch['label'].to(device)
            
            # Forward pass with mixed precision
            if args.mixed_precision:
                with autocast():
                    # Forward pass
                    outputs = model(waveforms)
                    
                    # Calculate losses
                    emotion_loss = emotion_criterion(outputs['emotion_logits'], emotion_labels)
                    
                    # Generate VAD targets (assume all training samples have speech)
                    batch_size = waveforms.size(0)
                    sequence_length = outputs['vad_logits'].size(1)
                    vad_targets = torch.ones(batch_size, sequence_length, 1).to(device)
                    vad_loss = vad_criterion(outputs['vad_logits'], vad_targets)
                    
                    # Combined loss with weighting
                    loss = emotion_loss + args.vad_weight * vad_loss
                    
                    # Accumulate loss for gradient accumulation
                    loss = loss / args.gradient_accumulation_steps
                    accum_loss += loss.item() * args.gradient_accumulation_steps
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                
                # Update weights with accumulation
                accum_iter += 1
                if accum_iter == args.gradient_accumulation_steps:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update EMA model
                    if args.use_ema:
                        ema.update()
                    
                    # Reset accumulation counter
                    accum_iter = 0
            else:
                # Standard precision training
                # Forward pass
                outputs = model(waveforms)
                
                # Calculate losses
                emotion_loss = emotion_criterion(outputs['emotion_logits'], emotion_labels)
                
                # Generate VAD targets (assume all training samples have speech)
                batch_size = waveforms.size(0)
                sequence_length = outputs['vad_logits'].size(1)
                vad_targets = torch.ones(batch_size, sequence_length, 1).to(device)
                vad_loss = vad_criterion(outputs['vad_logits'], vad_targets)
                
                # Combined loss with weighting
                loss = emotion_loss + args.vad_weight * vad_loss
                
                # Accumulate loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
                accum_loss += loss.item() * args.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights with accumulation
                accum_iter += 1
                if accum_iter == args.gradient_accumulation_steps:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update EMA model
                    if args.use_ema:
                        ema.update()
                    
                    # Reset accumulation counter
                    accum_iter = 0
            
            # Update statistics
            train_loss += accum_loss
            train_emotion_preds.extend(outputs['emotion_probs'].argmax(dim=1).cpu().numpy())
            train_emotion_labels.extend(emotion_labels.cpu().numpy())
            
            # Update learning rate with schedulers that step every batch
            if scheduler is not None and args.lr_scheduler in ['linear', 'cosine']:
                scheduler.step()
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_accuracy = accuracy_score(train_emotion_labels, train_emotion_preds)
        train_f1 = f1_score(train_emotion_labels, train_emotion_preds, average='macro')
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        # Validation
        if args.use_ema:
            # Apply EMA shadow model for evaluation
            ema.apply_shadow()
        
        model.eval()
        val_loss = 0.0
        val_emotion_preds = []
        val_emotion_labels = []
        val_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                waveforms = batch['waveform'].to(device)
                emotion_labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(waveforms)
                
                # Calculate emotion loss
                emotion_loss = emotion_criterion(outputs['emotion_logits'], emotion_labels)
                
                # Update statistics
                val_loss += emotion_loss.item()
                val_emotion_preds.extend(outputs['emotion_probs'].argmax(dim=1).cpu().numpy())
                val_emotion_labels.extend(emotion_labels.cpu().numpy())
                
                # Calculate confidence scores (difference between top two probabilities)
                probs = outputs['emotion_probs'].cpu()
                for i in range(probs.shape[0]):
                    sorted_probs, _ = torch.sort(probs[i], descending=True)
                    confidence = (sorted_probs[0] - sorted_probs[1]).item()
                    val_confidences.append(confidence)
        
        # Restore original model if using EMA
        if args.use_ema:
            ema.restore()
        
        # Calculate validation metrics
        val_loss /= len(valid_loader)
        val_accuracy = accuracy_score(val_emotion_labels, val_emotion_preds)
        val_f1 = f1_score(val_emotion_labels, val_emotion_preds, average='macro')
        val_mean_confidence = np.mean(val_confidences)
        
        # Log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Confidence/val', val_mean_confidence, epoch)
        
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Confidence: {val_mean_confidence:.4f}")
        
        # Update learning rate for ReduceLROnPlateau scheduler
        if scheduler is not None and args.lr_scheduler == 'reduce':
            scheduler.step(val_loss)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
            # Apply EMA shadow model if using EMA
            if args.use_ema:
                ema.apply_shadow()
                
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with F1: {val_f1:.4f}")
            
            # Restore original model if using EMA
            if args.use_ema:
                ema.restore()
            
        # Save last model
        model_path = os.path.join(args.output_dir, 'last_model.pt')
        
        # Apply EMA shadow model if using EMA
        if args.use_ema:
            ema.apply_shadow()
            
        torch.save(model.state_dict(), model_path)
        
        # Restore original model if using EMA
        if args.use_ema:
            ema.restore()
        
        # Save checkpoint with all information for resuming training
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_f1': best_val_f1,
            'ema': ema.shadow if args.use_ema else None,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pt'))
    
    # Final evaluation on validation set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    model.eval()
    
    test_emotion_preds = []
    test_emotion_labels = []
    test_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Final Evaluation"):
            waveforms = batch['waveform'].to(device)
            emotion_labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(waveforms)
            
            # Update statistics
            test_emotion_preds.extend(outputs['emotion_probs'].argmax(dim=1).cpu().numpy())
            test_emotion_labels.extend(emotion_labels.cpu().numpy())
    
    # Calculate test metrics
    test_time = time.time() - test_start_time
    test_accuracy = accuracy_score(test_emotion_labels, test_emotion_preds)
    test_f1 = f1_score(test_emotion_labels, test_emotion_preds, average='macro')
    test_cm = confusion_matrix(test_emotion_labels, test_emotion_preds)
    test_report = classification_report(test_emotion_labels, test_emotion_preds, target_names=["angry", "happy", "sad", "neutral"])
    
    # Calculate inference speed
    samples_per_sec = len(valid_loader.dataset) / test_time
    
    print(f"\nFinal Evaluation Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Inference Speed: {samples_per_sec:.2f} samples/sec")
    print(f"Confusion Matrix:\n{test_cm}")
    print(f"Classification Report:\n{test_report}")
    
    # Export to ONNX
    if args.export_onnx:
        dummy_input = torch.randn(1, 1, args.max_length).to(device)
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            input_names=['input'],
            output_names=['emotion_probs', 'vad_probs'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'sequence_length'},
                'emotion_probs': {0: 'batch_size'},
                'vad_probs': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=12
        )
        
        print(f"Exported ONNX model to {onnx_path}")
        
    print(f"Training completed. Best validation F1: {best_val_f1:.4f}")
    
    # Save final evaluation metrics
    with open(os.path.join(args.output_dir, 'final_results.txt'), 'w') as f:
        f.write("Final Evaluation Results\n")
        f.write("=======================\n\n")
        f.write(f"Accuracy: {test_accuracy:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"Inference Speed: {samples_per_sec:.2f} samples/sec\n\n")
        f.write(f"Confusion Matrix:\n{test_cm}\n\n")
        f.write(f"Classification Report:\n{test_report}\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Speech Emotion Recognition Training')
    
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
                        help='Freeze encoder parameters (transfer learning)')
    parser.add_argument('--unfreeze_encoder_epoch', type=int, default=-1,
                        help='Epoch to unfreeze encoder (-1 means keep frozen)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--vad_weight', type=float, default=0.2,
                        help='Weight for VAD loss')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--lr_scheduler', type=str, choices=['linear', 'cosine', 'reduce', 'none'], default='linear',
                        help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use Exponential Moving Average for model weights')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./models_enhanced',
                        help='Output directory for models and logs')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for dataloading')
    
    args = parser.parse_args()
    
    train_enhanced(args)


if __name__ == '__main__':
    main() 