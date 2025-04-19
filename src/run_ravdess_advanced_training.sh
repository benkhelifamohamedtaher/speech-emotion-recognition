#!/bin/bash
# Run advanced RAVDESS emotion recognition model training
# Configures TensorBoard and uses optimal training parameters

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
SAMPLE_RATE=16000
MAX_DURATION=5.0
NUM_WORKERS=4
DEVICE="cuda"

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
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${YELLOW}CUDA not available, using CPU instead${NC}"
    DEVICE="cpu"
    # Smaller batch size for CPU
    BATCH_SIZE=8
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
        --help)
            echo -e "${BLUE}Usage: ./run_ravdess_advanced_training.sh [options]${NC}"
            echo -e "${GREEN}Options:${NC}"
            echo -e "  --dataset_root DIR    Path to RAVDESS dataset directory (default: ../dataset/RAVDESS)"
            echo -e "  --output_dir DIR      Output directory for models (default: ../models/ravdess_advanced)"
            echo -e "  --batch_size N        Batch size (default: 16 for GPU, 8 for CPU)"
            echo -e "  --epochs N            Number of epochs (default: 50)"
            echo -e "  --learning_rate N     Learning rate (default: 3e-5)"
            echo -e "  --device DEVICE       Device to use (default: cuda if available, cpu otherwise)"
            echo -e "  --emotion_subset SET  Emotion subset to use (default: basic6)"
            echo -e "  --sample_rate N       Audio sample rate (default: 16000)"
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

# Tensorboard setup
TENSORBOARD_DIR="$OUTPUT_DIR/logs"
mkdir -p $TENSORBOARD_DIR

# Launch TensorBoard in the background
echo -e "${BLUE}Starting TensorBoard...${NC}"
tensorboard --logdir=$TENSORBOARD_DIR --port=6006 --host=0.0.0.0 &
TENSORBOARD_PID=$!

# Print TensorBoard URL
echo -e "${GREEN}TensorBoard running at http://localhost:6006${NC}"
echo -e "${YELLOW}(Keep this terminal open to maintain the TensorBoard server)${NC}"
echo

# Start training
echo -e "${BLUE}Starting training...${NC}"
echo

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
    --audio_only \
    --speech_only \
    --num_workers $NUM_WORKERS \
    --optimizer adamw \
    --scheduler cosine \
    --weight_decay 1e-4 \
    --use_gender_branch \
    --use_spectrogram_branch \
    --freeze_extractor \
    --use_amp

TRAINING_STATUS=$?

# Check if training completed successfully
if [ $TRAINING_STATUS -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to $OUTPUT_DIR${NC}"
    
    # Run evaluation on trained model if it exists
    if [ -f "$OUTPUT_DIR/best_model.pt" ]; then
        echo -e "${BLUE}Evaluating best model...${NC}"
        python ravdess_evaluate.py \
            --model_path "$OUTPUT_DIR/best_model.pt" \
            --dataset_root $DATASET_ROOT \
            --device $DEVICE
    else
        echo -e "${YELLOW}No best model found. Skipping evaluation.${NC}"
    fi
else
    echo -e "${RED}Training failed with status code $TRAINING_STATUS!${NC}"
fi

# Stop TensorBoard server
if [ ! -z "$TENSORBOARD_PID" ]; then
    echo -e "${BLUE}Stopping TensorBoard...${NC}"
    kill $TENSORBOARD_PID
fi

# Print instruction to run inference
echo -e "${YELLOW}To run real-time inference with the trained model, use:${NC}"
echo -e "${BLUE}python ravdess_inference.py --model_path $OUTPUT_DIR/best_model.pt${NC}"

# Exit with the same status as the training
exit $TRAINING_STATUS 