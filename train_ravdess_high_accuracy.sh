#!/bin/bash
# Script to train the RAVDESS emotion recognition model with high-accuracy settings

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== High-Accuracy RAVDESS Emotion Recognition Training =====${NC}"
echo

# Define paths and parameters optimized for high accuracy
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../models/ravdess_high_accuracy"
SAMPLE_RATE=16000
MAX_LENGTH=80000  # 5 seconds at 16kHz for more context
BATCH_SIZE=32
LEARNING_RATE=0.0005  # Lower learning rate for better convergence
EPOCHS=50
NUM_WORKERS=4
DEVICE="cpu"

# Enhanced training parameters
USE_AUGMENT="--augment"  # Enable augmentation
USE_STRONG_AUGMENT="--strong_augment"  # Enable stronger augmentation
USE_WEIGHT_DECAY="--weight_decay 0.0001"  # L2 regularization
USE_SCHEDULER="--scheduler cosine"  # Use cosine annealing scheduler
USE_DROPOUT="--dropout 0.4"  # Higher dropout for better generalization

# Check if PyTorch is available
if ! python -c "import torch" &>/dev/null; then
    echo -e "${RED}PyTorch is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if TensorBoard is available
if ! python -c "from torch.utils.tensorboard import SummaryWriter" &>/dev/null; then
    echo -e "${YELLOW}TensorBoard not available. Installing...${NC}"
    python -m pip install tensorboard
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install TensorBoard. Training will continue without logging.${NC}"
    else
        echo -e "${GREEN}TensorBoard installed successfully.${NC}"
    fi
fi

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}CUDA is available, using GPU${NC}"
    DEVICE="cuda"
    # Increase batch size for GPU
    BATCH_SIZE=64
else
    echo -e "${YELLOW}CUDA not available, using CPU instead${NC}"
fi

# Check if any TensorBoard instances are already running
TB_PID=$(ps aux | grep tensorboard | grep -v grep | awk '{print $2}')
if [ ! -z "$TB_PID" ]; then
    echo -e "${YELLOW}Found existing TensorBoard process (PID: $TB_PID). Terminating...${NC}"
    kill $TB_PID &>/dev/null
    sleep 2
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
        --disable_augment)
            USE_AUGMENT=""
            shift
            ;;
        --disable_strong_augment)
            USE_STRONG_AUGMENT=""
            shift
            ;;
        --help)
            echo -e "${BLUE}Usage: ./train_ravdess_high_accuracy.sh [options]${NC}"
            echo -e "${GREEN}Options:${NC}"
            echo -e "  --dataset_root DIR      Path to RAVDESS dataset directory (default: ../dataset/RAVDESS)"
            echo -e "  --output_dir DIR        Output directory for models (default: ../models/ravdess_high_accuracy)"
            echo -e "  --batch_size N          Batch size (default: 32 for CPU, 64 for GPU)"
            echo -e "  --epochs N              Number of epochs (default: 50)"
            echo -e "  --learning_rate N       Learning rate (default: 0.0005)"
            echo -e "  --device DEVICE         Device to use (default: cuda if available, cpu otherwise)"
            echo -e "  --disable_augment       Disable audio augmentation"
            echo -e "  --disable_strong_augment Disable stronger augmentation techniques"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option $key${NC}"
            echo -e "${YELLOW}Use --help for usage information${NC}"
            exit 1
            ;;
    esac
done

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo -e "${RED}Dataset directory not found: $DATASET_ROOT${NC}"
    echo -e "${YELLOW}Please download and extract the RAVDESS dataset first.${NC}"
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

# Create main output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Create specific output directories for each emotion set
FULL_OUTPUT_DIR="${OUTPUT_DIR}/full"
BASIC6_OUTPUT_DIR="${OUTPUT_DIR}/basic6"
SIMPLIFIED_OUTPUT_DIR="${OUTPUT_DIR}/simplified"
mkdir -p $FULL_OUTPUT_DIR
mkdir -p $BASIC6_OUTPUT_DIR
mkdir -p $SIMPLIFIED_OUTPUT_DIR

# Display training configuration
echo -e "${BLUE}Training configuration:${NC}"
echo -e "${GREEN}Dataset:${NC} $DATASET_ROOT"
echo -e "${GREEN}Main output directory:${NC} $OUTPUT_DIR"
echo -e "${GREEN}Sample rate:${NC} $SAMPLE_RATE Hz"
echo -e "${GREEN}Max audio length:${NC} $MAX_LENGTH samples ($(($MAX_LENGTH/$SAMPLE_RATE)) seconds)"
echo -e "${GREEN}Batch size:${NC} $BATCH_SIZE"
echo -e "${GREEN}Learning rate:${NC} $LEARNING_RATE"
echo -e "${GREEN}Epochs:${NC} $EPOCHS"
echo -e "${GREEN}Device:${NC} $DEVICE"
echo -e "${GREEN}Augmentation:${NC} ${USE_AUGMENT:+enabled}"
echo -e "${GREEN}Strong augmentation:${NC} ${USE_STRONG_AUGMENT:+enabled}"
echo

# Tensorboard setup
TENSORBOARD_DIR="$OUTPUT_DIR/logs"
mkdir -p $TENSORBOARD_DIR

# Launch TensorBoard in the background
echo -e "${BLUE}Starting TensorBoard...${NC}"
# Use port 6009 to avoid conflicts
tensorboard --logdir=$TENSORBOARD_DIR --port=6009 --host=0.0.0.0 &
TENSORBOARD_PID=$!

# Check if TensorBoard started successfully
sleep 3
if ps -p $TENSORBOARD_PID > /dev/null; then
    # Print TensorBoard URL
    echo -e "${GREEN}TensorBoard running at http://localhost:6009${NC}"
    echo -e "${YELLOW}(Keep this terminal open to maintain the TensorBoard server)${NC}"
else
    echo -e "${YELLOW}TensorBoard failed to start. Training will continue without visualization.${NC}"
    TENSORBOARD_PID=""
fi
echo

# Train with basic6 emotion set (most useful for applications)
echo -e "${BLUE}=== Training model with basic6 emotion set (6 emotions) ===${NC}"
echo -e "${GREEN}Output directory:${NC} $BASIC6_OUTPUT_DIR"
echo -e "${BLUE}Starting training...${NC}"
echo

python train_ravdess_simple.py \
  --dataset_root $DATASET_ROOT \
  --output_dir $BASIC6_OUTPUT_DIR \
  --emotion_set basic6 \
  --split train \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  $USE_AUGMENT \
  $USE_STRONG_AUGMENT \
  $USE_WEIGHT_DECAY \
  $USE_SCHEDULER \
  $USE_DROPOUT \
  --early_stopping \
  --patience 10 \
  --tensorboard \
  --device $DEVICE

BASIC6_STATUS=$?

# Train with full emotion set
echo -e "${BLUE}=== Training model with full emotion set (8 emotions) ===${NC}"
echo -e "${GREEN}Output directory:${NC} $FULL_OUTPUT_DIR"
echo -e "${BLUE}Starting training...${NC}"
echo

python train_ravdess_simple.py \
  --dataset_root $DATASET_ROOT \
  --output_dir $FULL_OUTPUT_DIR \
  --emotion_set full \
  --split train \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  $USE_AUGMENT \
  $USE_STRONG_AUGMENT \
  $USE_WEIGHT_DECAY \
  $USE_SCHEDULER \
  $USE_DROPOUT \
  --early_stopping \
  --patience 10 \
  --tensorboard \
  --device $DEVICE

FULL_STATUS=$?

# Train with simplified emotion set
echo -e "${BLUE}=== Training model with simplified emotion set (4 emotions) ===${NC}"
echo -e "${GREEN}Output directory:${NC} $SIMPLIFIED_OUTPUT_DIR"
echo -e "${BLUE}Starting training...${NC}"
echo

python train_ravdess_simple.py \
  --dataset_root $DATASET_ROOT \
  --output_dir $SIMPLIFIED_OUTPUT_DIR \
  --emotion_set simplified \
  --split train \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --num_workers $NUM_WORKERS \
  $USE_AUGMENT \
  $USE_STRONG_AUGMENT \
  $USE_WEIGHT_DECAY \
  $USE_SCHEDULER \
  $USE_DROPOUT \
  --early_stopping \
  --patience 10 \
  --tensorboard \
  --device $DEVICE

SIMPLIFIED_STATUS=$?

# Stop TensorBoard server
if [ ! -z "$TENSORBOARD_PID" ] && ps -p $TENSORBOARD_PID > /dev/null; then
    echo -e "${BLUE}Stopping TensorBoard...${NC}"
    kill $TENSORBOARD_PID
fi

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