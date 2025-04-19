#!/bin/bash
# Script to train the RAVDESS emotion recognition model with high-accuracy settings

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Define paths and parameters optimized for high accuracy
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../models/ravdess_high_accuracy"
SAMPLE_RATE=16000
MAX_LENGTH=80000  # 5 seconds at 16kHz for more context
BATCH_SIZE=32
LEARNING_RATE=0.0005  # Lower learning rate for better convergence
EPOCHS=50
NUM_WORKERS=4
DEVICE="cpu"  # Will be set to cuda if available

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}CUDA is available, using GPU${NC}"
    DEVICE="cuda"
    # Larger batch size for GPU
    BATCH_SIZE=64
else
    echo -e "${YELLOW}CUDA not available, using CPU instead${NC}"
fi

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --dataset_root=*)
        DATASET_ROOT="${arg#*=}"
        shift
        ;;
        --output_dir=*)
        OUTPUT_DIR="${arg#*=}"
        shift
        ;;
        --batch_size=*)
        BATCH_SIZE="${arg#*=}"
        shift
        ;;
        --epochs=*)
        EPOCHS="${arg#*=}"
        shift
        ;;
        --learning_rate=*)
        LEARNING_RATE="${arg#*=}"
        shift
        ;;
        --device=*)
        DEVICE="${arg#*=}"
        shift
        ;;
        *)
        # Unknown option
        ;;
    esac
done

# Create main output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Create specific output directories for each emotion set
FULL_OUTPUT_DIR="${OUTPUT_DIR}/full"
BASIC6_OUTPUT_DIR="${OUTPUT_DIR}/basic6"
SIMPLIFIED_OUTPUT_DIR="${OUTPUT_DIR}/simplified"
mkdir -p $FULL_OUTPUT_DIR
mkdir -p $BASIC6_OUTPUT_DIR
mkdir -p $SIMPLIFIED_OUTPUT_DIR

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}=== High-Accuracy RAVDESS Speech Emotion Models ===${NC}"
echo -e "${BLUE}===============================================${NC}"
echo -e "${GREEN}Dataset:${NC} $DATASET_ROOT"
echo -e "${GREEN}Main output directory:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Sample rate:${NC} $SAMPLE_RATE Hz"
echo -e "${GREEN}Max audio length:${NC} $MAX_LENGTH samples ($(($MAX_LENGTH/$SAMPLE_RATE)) seconds)"
echo -e "${GREEN}Batch size:${NC} $BATCH_SIZE"
echo -e "${GREEN}Learning rate:${NC} $LEARNING_RATE"
echo -e "${GREEN}Epochs:${NC} $EPOCHS"
echo -e "${GREEN}Device:${NC} $DEVICE"
echo -e "${BLUE}===============================================${NC}"

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo -e "${RED}ERROR: Dataset directory $DATASET_ROOT does not exist${NC}"
    echo -e "${YELLOW}Please download the RAVDESS dataset and extract it to $DATASET_ROOT${NC}"
    exit 1
fi

# Check number of audio files in dataset
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

# Train with basic6 emotion set (most practical)
echo
echo -e "${BLUE}=== Training model with basic6 emotion set (6 emotions) ===${NC}"
echo -e "${GREEN}Output directory:${NC} $BASIC6_OUTPUT_DIR"
echo -e "${BLUE}Starting training...${NC}"

python train_ravdess_simple.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$BASIC6_OUTPUT_DIR" \
  --emotion_set basic6 \
  --split train \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  --augment \
  --strong_augment \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --dropout 0.4 \
  --early_stopping \
  --patience 10 \
  --tensorboard \
  --device $DEVICE

BASIC6_STATUS=$?

# Train with full emotion set
echo
echo -e "${BLUE}=== Training model with full emotion set (8 emotions) ===${NC}"
echo -e "${GREEN}Output directory:${NC} $FULL_OUTPUT_DIR"
echo -e "${BLUE}Starting training...${NC}"

python train_ravdess_simple.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$FULL_OUTPUT_DIR" \
  --emotion_set full \
  --split train \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  --augment \
  --strong_augment \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --dropout 0.4 \
  --early_stopping \
  --patience 10 \
  --tensorboard \
  --device $DEVICE

FULL_STATUS=$?

# Train with simplified emotion set
echo
echo -e "${BLUE}=== Training model with simplified emotion set (4 emotions) ===${NC}"
echo -e "${GREEN}Output directory:${NC} $SIMPLIFIED_OUTPUT_DIR"
echo -e "${BLUE}Starting training...${NC}"

python train_ravdess_simple.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$SIMPLIFIED_OUTPUT_DIR" \
  --emotion_set simplified \
  --split train \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  --augment \
  --strong_augment \
  --weight_decay 0.0001 \
  --scheduler cosine \
  --dropout 0.4 \
  --early_stopping \
  --patience 10 \
  --tensorboard \
  --device $DEVICE

SIMPLIFIED_STATUS=$?

echo
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}=== Training completed ===${NC}"
echo -e "${YELLOW}Models saved to:${NC}"
echo -e "${GREEN}  - Full emotions (8): $FULL_OUTPUT_DIR${NC}"
echo -e "${GREEN}  - Basic6 emotions (6): $BASIC6_OUTPUT_DIR${NC}"
echo -e "${GREEN}  - Simplified emotions (4): $SIMPLIFIED_OUTPUT_DIR${NC}"
echo

# Print instruction to run inference
echo -e "${YELLOW}To run real-time inference with the trained model, use:${NC}"
echo -e "${BLUE}python ravdess_inference.py --model_path $BASIC6_OUTPUT_DIR/best_model.pt${NC}"

# Check if any training failed
if [ $BASIC6_STATUS -ne 0 ] || [ $FULL_STATUS -ne 0 ] || [ $SIMPLIFIED_STATUS -ne 0 ]; then
    echo -e "${RED}WARNING: One or more training runs failed!${NC}"
    exit 1
fi

exit 0 