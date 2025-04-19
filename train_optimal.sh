#!/bin/bash
# Optimal training script for RAVDESS emotion recognition model
# This script uses the best hyperparameters for training high-quality models

# Set colored output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Optimal Speech Emotion Recognition     ║${NC}"
echo -e "${BLUE}║             Training Pipeline              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

# Define paths and parameters
DATASET_ROOT="./dataset"
OUTPUT_DIR="./models/optimal_model"
SAMPLE_RATE=16000
MAX_LENGTH=48000  # 3 seconds at 16kHz
BATCH_SIZE=16
LEARNING_RATE=0.0001
EPOCHS=50
NUM_WORKERS=0  # Set to 0 to avoid multiprocessing issues
DEVICE="cpu"

# Check for GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected! Using CUDA for training.${NC}"
    DEVICE="cuda"
elif [ "$(uname)" == "Darwin" ] && system_profiler SPHardwareDataType | grep -q "Apple M"; then
    echo -e "${GREEN}Apple Silicon detected! Using MPS for training.${NC}"
    DEVICE="mps"
else
    echo -e "${YELLOW}No GPU detected. Using CPU for training (this will be slower).${NC}"
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo -e "${GREEN}=== Training Setup ===${NC}"
echo -e "Dataset: $DATASET_ROOT"
echo -e "Output directory: $OUTPUT_DIR"
echo -e "Device: $DEVICE"
echo -e "Batch size: $BATCH_SIZE"
echo -e "Learning rate: $LEARNING_RATE"
echo -e "Epochs: $EPOCHS"

# Step 1: Train enhanced RAVDESS model with full emotion set (using simplified script)
echo -e "\n${GREEN}=== Training Enhanced RAVDESS Model (Full 8 Emotions) ===${NC}"
python src/train_ravdess_simple.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "${OUTPUT_DIR}/ravdess_full" \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay 1e-5 \
  --epochs $EPOCHS \
  --device $DEVICE \
  --augment \
  --save_checkpoints

# Step 2: Train enhanced RAVDESS model with simplified emotion set
echo -e "\n${GREEN}=== Training Enhanced RAVDESS Model (Simplified 4 Emotions) ===${NC}"
python src/train_ravdess_simple.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "${OUTPUT_DIR}/ravdess_simplified" \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --weight_decay 1e-5 \
  --epochs $EPOCHS \
  --device $DEVICE \
  --use_simplified \
  --augment \
  --save_checkpoints

echo -e "\n${GREEN}=== Training completed! ===${NC}"
echo -e "Models saved to $OUTPUT_DIR"
echo -e "\n${YELLOW}To test the enhanced RAVDESS model:${NC}"
echo -e "  python src/inference.py --model ${OUTPUT_DIR}/ravdess_full/best_model.pt" 