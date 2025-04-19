#!/bin/bash
# Advanced training script for RAVDESS emotion recognition model
# Uses state-of-the-art techniques for high accuracy emotion recognition

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Advanced RAVDESS Emotion Recognition Training =====${NC}"
echo

# Default parameters
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../models/ravdess_advanced"
BATCH_SIZE=16
EPOCHS=50
LEARNING_RATE=3e-5
EMOTION_SUBSET="basic6"  # Use 6 basic emotions for best performance
USE_GENDER_BRANCH="--use_gender_branch"
USE_SPECTROGRAM_BRANCH="--use_spectrogram_branch"
FREEZE_EXTRACTOR="--freeze_extractor"
SAMPLE_RATE=16000
MAX_DURATION=5.0
AUDIO_ONLY="--audio_only"
SPEECH_ONLY="--speech_only"
NUM_WORKERS=4
OPTIMIZER="adamw"
SCHEDULER="cosine"
WEIGHT_DECAY=1e-4
DEVICE="cuda"

# Check if CUDA is available
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${YELLOW}CUDA not available, using CPU instead${NC}"
    DEVICE="cpu"
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
        --emotion_subset)
            EMOTION_SUBSET="$2"
            shift
            shift
            ;;
        --sample_rate)
            SAMPLE_RATE="$2"
            shift
            shift
            ;;
        *)
            echo -e "${RED}Unknown option $key${NC}"
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
echo -e "${GREEN}Emotion subset:${NC} $EMOTION_SUBSET"
echo -e "${GREEN}Sample rate:${NC} $SAMPLE_RATE"
echo

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

echo -e "${BLUE}Starting training...${NC}"
echo

# Start training
python train_ravdess_advanced.py \
    --dataset_root $DATASET_ROOT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --emotion_subset $EMOTION_SUBSET \
    --sample_rate $SAMPLE_RATE \
    --max_duration $MAX_DURATION \
    $AUDIO_ONLY \
    $SPEECH_ONLY \
    --num_workers $NUM_WORKERS \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --weight_decay $WEIGHT_DECAY \
    $USE_GENDER_BRANCH \
    $USE_SPECTROGRAM_BRANCH \
    $FREEZE_EXTRACTOR \
    --use_amp

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to $OUTPUT_DIR${NC}"
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

# Run evaluation on trained model
echo -e "${BLUE}Evaluating best model...${NC}"
python ravdess_evaluate.py \
    --model_path "$OUTPUT_DIR/best_model.pt" \
    --dataset_root $DATASET_ROOT \
    --device $DEVICE

# Print instruction to run inference
echo -e "${YELLOW}To run real-time inference with the trained model, use:${NC}"
echo -e "${BLUE}python ravdess_inference.py --model_path $OUTPUT_DIR/best_model.pt${NC}" 