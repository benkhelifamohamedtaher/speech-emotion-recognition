#!/bin/bash
# Run RAVDESS emotion recognition model training with all 8 emotions
# Simplified version for faster training

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== RAVDESS Emotion Recognition Training (8 Emotions) =====${NC}"
echo

# Use the correct Python interpreter
PYTHON_CMD="python3"

# Default parameters optimized for faster training
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../models/ravdess_simple"
BATCH_SIZE=8
EPOCHS=10  # Reduced epochs for faster training
LEARNING_RATE=1e-4
DEVICE="cpu"

# Check if PyTorch is available
if ! $PYTHON_CMD -c "import torch" &>/dev/null; then
    echo -e "${RED}PyTorch is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if necessary audio libraries are available
if ! $PYTHON_CMD -c "import soundfile, librosa" &>/dev/null; then
    echo -e "${RED}Required audio libraries are not installed. Please install them first.${NC}"
    echo -e "${YELLOW}Try: pip install soundfile librosa${NC}"
    exit 1
fi

# Check if transformers is available
if ! $PYTHON_CMD -c "import transformers" &>/dev/null; then
    echo -e "${YELLOW}Transformers library not available. Installing...${NC}"
    $PYTHON_CMD -m pip install transformers
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install transformers. Exiting.${NC}"
        exit 1
    else
        echo -e "${GREEN}Transformers installed successfully.${NC}"
    fi
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset_root)
            DATASET_ROOT="$2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift
            shift
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --help)
            echo -e "${BLUE}Usage: ./run_ravdess_simple.sh [options]${NC}"
            echo -e "${GREEN}Options:${NC}"
            echo -e "  --dataset_root DIR      Path to RAVDESS dataset directory (default: ../dataset/RAVDESS)"
            echo -e "  --output_dir DIR        Output directory for models (default: ../models/ravdess_simple)"
            echo -e "  --batch_size N          Batch size (default: 8)"
            echo -e "  --epochs N              Number of epochs (default: 10)"
            echo -e "  --learning_rate N       Learning rate (default: 1e-4)"
            echo -e "  --device DEVICE         Device to use (default: cpu)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option $key${NC}"
            echo -e "${YELLOW}Use --help for usage information${NC}"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Display training configuration
echo -e "${BLUE}Training configuration:${NC}"
echo -e "${GREEN}Dataset:${NC} $DATASET_ROOT"
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Batch size:${NC} $BATCH_SIZE"
echo -e "${GREEN}Epochs:${NC} $EPOCHS"
echo -e "${GREEN}Learning rate:${NC} $LEARNING_RATE"
echo -e "${GREEN}Device:${NC} $DEVICE"
echo

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo -e "${RED}Dataset directory not found: $DATASET_ROOT${NC}"
    echo -e "${YELLOW}Please download and extract the RAVDESS dataset first.${NC}"
    exit 1
fi

# Count audio files in dataset
NUM_FILES=$(find "$DATASET_ROOT" -name "*.wav" | wc -l)
echo -e "${GREEN}Found $NUM_FILES audio files in dataset${NC}"

if [ $NUM_FILES -lt 100 ]; then
    echo -e "${YELLOW}WARNING: Very few audio files found in dataset. Make sure you've extracted the RAVDESS dataset correctly.${NC}"
    echo -e "${YELLOW}The RAVDESS dataset should contain around 1440 audio files.${NC}"
    
    # Ask if user wants to continue
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create a simple Python script for training
TRAIN_SCRIPT="$OUTPUT_DIR/train_temp.py"
echo -e "${BLUE}Creating temporary training script...${NC}"
cat > $TRAIN_SCRIPT << 'EOF'
#!/usr/bin/env python3
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
from transformers import Wav2Vec2Model, Wav2Vec2Config

# Setup argument parser
parser = argparse.ArgumentParser(description="Simple RAVDESS Training Script")
parser.add_argument("--dataset_root", type=str, required=True, help="Path to RAVDESS dataset")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for models")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda or cpu)")
args = parser.parse_args()

# RAVDESS dataset class
class SimpleRAVDESSDataset(Dataset):
    """
    Simple dataset class for RAVDESS
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
    
    def __init__(self, root_dir, sample_rate=16000, max_duration=5.0):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_sample_length = int(max_duration * sample_rate)
        
        # Get all audio files
        self.file_paths = []
        for actor_dir in self.root_dir.glob("Actor_*"):
            if actor_dir.is_dir():
                for file_path in actor_dir.glob("*.wav"):
                    self.file_paths.append(file_path)
        
        print(f"Found {len(self.file_paths)} audio files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Parse filename for metadata
        parts = file_path.stem.split('-')
        emotion_id = parts[2]  # Emotion ID
        
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
        
        return {
            'waveform': waveform,
            'emotion': emotion_idx,
            'file_path': str(file_path)
        }

# Simple model class
class SimpleEmotionRecognizer(nn.Module):
    def __init__(self, num_emotions=8):
        super().__init__()
        # Use a pre-trained wav2vec2 model
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Freeze feature extractor
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
        
        # Define classification head
        hidden_size = self.wav2vec.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        x = x.squeeze(1)  # Remove channel dimension
        
        # Get wav2vec features
        outputs = self.wav2vec(x)
        hidden_states = outputs.last_hidden_state
        
        # Average pooling
        pooled = torch.mean(hidden_states, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# Training function
def train(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    
    for batch in progress_bar:
        waveforms = batch['waveform'].to(device)
        emotions = batch['emotion'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(waveforms)
        loss = F.cross_entropy(logits, emotions)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = logits.max(1)
        correct += predicted.eq(emotions).sum().item()
        total += emotions.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{running_loss/len(progress_bar):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    return running_loss / len(dataloader), 100. * correct / total

# Main function
def main():
    print(f"Training with the following parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = SimpleRAVDESSDataset(args.dataset_root)
    
    # Split dataset: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    
    # Create model
    model = SimpleEmotionRecognizer(num_emotions=8).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, device, epoch, args.epochs)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                waveforms = batch['waveform'].to(device)
                emotions = batch['emotion'].to(device)
                
                logits = model(waveforms)
                loss = F.cross_entropy(logits, emotions)
                
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                correct += predicted.eq(emotions).sum().item()
                total += emotions.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if it's the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Saved final model with validation accuracy: {val_acc:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
EOF

# Make the script executable
chmod +x $TRAIN_SCRIPT

# Start training
echo -e "${BLUE}Starting training...${NC}"
echo -e "${YELLOW}This may take some time.${NC}"
echo

$PYTHON_CMD $TRAIN_SCRIPT \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE

TRAINING_STATUS=$?

# Check if training completed successfully
if [ $TRAINING_STATUS -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to $OUTPUT_DIR${NC}"
    
    # Print instruction to run inference
    echo -e "${YELLOW}To run inference with the trained model, use:${NC}"
    echo -e "${BLUE}python ravdess_inference.py --model_path $OUTPUT_DIR/final_model.pt${NC}"
else
    echo -e "${RED}Training failed with status code $TRAINING_STATUS!${NC}"
fi

# Exit with the same status as the training
exit $TRAINING_STATUS 