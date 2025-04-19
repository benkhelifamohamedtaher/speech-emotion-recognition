#!/usr/bin/env python3
"""
Training functions for advanced speech emotion recognition models.
These handle proper error recovery and device management.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_one_epoch(model, train_loader, optimizer, scaler, samples_per_class, args, epoch):
    """Train for one epoch with proper error handling"""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Get data and move to device
            waveforms = batch['waveform'].to(args.device)
            emotion_targets = batch['emotion'].to(args.device)
            
            # Handle tensor type issues - ensure targets are long
            if emotion_targets.dtype != torch.long:
                emotion_targets = emotion_targets.long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            with autocast(enabled=args.use_amp):
                # Use safe forward pass with explicit parameters
                outputs = model(
                    waveform=waveforms, 
                    emotion_targets=emotion_targets,
                    apply_mixup=args.use_mixup if hasattr(args, 'use_mixup') else False,
                    apply_augmentation=args.use_specaugment if hasattr(args, 'use_specaugment') else False
                )
                loss = outputs['loss']
            
            # Handle NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"NaN or Inf loss detected at batch {batch_idx}. Skipping batch.")
                continue
            
            # Backward pass with gradient scaling if enabled
            if args.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward pass
                loss.backward()
                
                # Gradient clipping
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                # Update weights
                optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                if 'emotion_logits' in outputs:
                    _, predicted = outputs['emotion_logits'].max(1)
                    total += emotion_targets.size(0)
                    correct += predicted.eq(emotion_targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{epoch_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%" if total > 0 else "N/A"
            })
            
        except RuntimeError as e:
            # Handle CUDA out-of-memory and other runtime errors
            logger.error(f"Runtime error in batch {batch_idx}: {e}")
            # If out of memory, try to free up some space
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            continue
        
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Calculate final metrics
    avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(model, val_loader, args, epoch=None):
    """Validate the model with proper error handling"""
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
                # Get data and move to device
                waveforms = batch['waveform'].to(args.device)
                emotion_targets = batch['emotion'].to(args.device)
                
                # Handle tensor type issues - ensure targets are long
                if emotion_targets.dtype != torch.long:
                    emotion_targets = emotion_targets.long()
                
                # Forward pass
                outputs = model(
                    waveform=waveforms, 
                    emotion_targets=emotion_targets, 
                    apply_mixup=False,  # No mixup in validation
                    apply_augmentation=False  # No augmentation in validation
                )
                loss = outputs['loss']
                
                # Handle NaN loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"NaN or Inf loss detected at validation batch {batch_idx}. Skipping batch.")
                    continue
                
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
                    'acc': f"{100.*correct/total:.2f}%" if total > 0 else "N/A"
                })
                
            except Exception as e:
                # Handle exceptions
                logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Calculate final metrics
    avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy, all_preds, all_targets 