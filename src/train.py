import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model import SpeechEmotionRecognitionModel
from data_utils import create_dataloaders


def train(args):
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
    train_loader, test_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        target_sr=args.sample_rate,
        max_length=args.max_length,
        apply_augmentation=args.augment,
        num_workers=args.num_workers
    )
    
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(test_loader.dataset)} samples")
    
    # Create model
    model = SpeechEmotionRecognitionModel(
        num_emotions=4, 
        freeze_encoder=args.freeze_encoder
    )
    model.to(device)
    
    # Define loss functions
    emotion_criterion = nn.CrossEntropyLoss()
    vad_criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Training loop
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        model.train()
        train_loss = 0.0
        train_emotion_preds = []
        train_emotion_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            waveforms = batch['waveform'].to(device)
            emotion_labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(waveforms)
            
            # Calculate losses
            emotion_loss = emotion_criterion(outputs['emotion_logits'], emotion_labels)
            
            # We don't have explicit VAD labels, using a heuristic
            # Assuming all training samples have speech
            batch_size = waveforms.size(0)
            sequence_length = outputs['vad_logits'].size(1)
            vad_targets = torch.ones(batch_size, sequence_length, 1).to(device)
            vad_loss = vad_criterion(outputs['vad_logits'], vad_targets)
            
            # Combined loss with weighting
            loss = emotion_loss + args.vad_weight * vad_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            train_emotion_preds.extend(outputs['emotion_probs'].argmax(dim=1).cpu().numpy())
            train_emotion_labels.extend(emotion_labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_accuracy = accuracy_score(train_emotion_labels, train_emotion_preds)
        train_f1 = f1_score(train_emotion_labels, train_emotion_preds, average='macro')
        
        # Log training metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_emotion_preds = []
        val_emotion_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
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
        
        # Calculate validation metrics
        val_loss /= len(test_loader)
        val_accuracy = accuracy_score(val_emotion_labels, val_emotion_preds)
        val_f1 = f1_score(val_emotion_labels, val_emotion_preds, average='macro')
        
        # Log validation metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with F1: {val_f1:.4f}")
            
        # Save last model
        model_path = os.path.join(args.output_dir, 'last_model.pt')
        torch.save(model.state_dict(), model_path)
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    model.eval()
    
    test_emotion_preds = []
    test_emotion_labels = []
    test_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
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
    
    # Calculate inference speed
    samples_per_sec = len(test_loader.dataset) / test_time
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Inference Speed: {samples_per_sec:.2f} samples/sec")
    print(f"Confusion Matrix:\n{test_cm}")
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Training')
    
    # Data parameters
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                        help='Path to dataset root')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--max_length', type=int, default=48000,
                        help='Max audio length in samples (3 seconds at 16kHz)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--vad_weight', type=float, default=0.2,
                        help='Weight of VAD loss in total loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder parameters')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export model to ONNX format')
    
    args = parser.parse_args()
    train(args) 