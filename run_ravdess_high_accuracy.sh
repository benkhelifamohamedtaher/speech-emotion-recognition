#!/bin/bash
# Run high-accuracy RAVDESS emotion recognition model training with all 8 emotions
# Designed for maximum accuracy with no early stopping

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== High-Accuracy RAVDESS Emotion Recognition Training (8 Emotions) =====${NC}"
echo

# Default parameters optimized for high accuracy with all 8 emotions
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../models/ravdess_full_emotions"
BATCH_SIZE=32
EPOCHS=50  # Full 50 epochs, no early stopping
LEARNING_RATE=5e-5  # Lower learning rate for better convergence
EMOTION_SUBSET=""  # Empty for full 8 emotions
SAMPLE_RATE=16000
MAX_DURATION=5.0
NUM_WORKERS=4
DEVICE="cuda"
# Higher model complexity for better accuracy
WEIGHT_DECAY=3e-5  # Stronger regularization
OPTIMIZER="adamw"
SCHEDULER="cosine"
COSINE_T0=20
COSINE_T_MULT=2
CLIP_GRAD=1.0
USE_AMP="--use_amp"
CACHE_WAVEFORMS="--cache_waveforms"

# Advanced model parameters
USE_ATTENTION="--use_attention"
USE_CONTEXT_LAYERS="--context_layers 4"
USE_ATTENTION_HEADS="--attention_heads 8"
USE_DROPOUT="--dropout 0.3"
USE_SPECTROGRAM="--use_spectrogram_branch"
USE_GENDER="--use_gender_branch"
FREEZE_EXTRACTOR="--freeze_extractor"

# Check if PyTorch is available
if ! python -c "import torch" &>/dev/null; then
    echo -e "${RED}PyTorch is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if necessary audio libraries are available
if ! python -c "import soundfile, librosa" &>/dev/null; then
    echo -e "${RED}Required audio libraries are not installed. Please install them first.${NC}"
    echo -e "${YELLOW}Try: pip install soundfile librosa${NC}"
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

# Check if any TensorBoard instances are already running
TB_PID=$(ps aux | grep tensorboard | grep -v grep | awk '{print $2}')
if [ ! -z "$TB_PID" ]; then
    echo -e "${YELLOW}Found existing TensorBoard process (PID: $TB_PID). Terminating...${NC}"
    kill $TB_PID &>/dev/null
    sleep 2
fi

# Check if CUDA is available
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${YELLOW}CUDA not available, using CPU instead${NC}"
    DEVICE="cpu"
    # Smaller batch size for CPU
    BATCH_SIZE=16
    # Disable AMP for CPU
    USE_AMP=""
    # Reduce model complexity for CPU
    USE_CONTEXT_LAYERS="--context_layers 2"
    USE_ATTENTION_HEADS="--attention_heads 4"
    echo -e "${YELLOW}Adjusted parameters for CPU training${NC}"
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
            echo -e "${BLUE}Usage: ./run_ravdess_high_accuracy.sh [options]${NC}"
            echo -e "${GREEN}Options:${NC}"
            echo -e "  --dataset_root DIR      Path to RAVDESS dataset directory (default: ../dataset/RAVDESS)"
            echo -e "  --output_dir DIR        Output directory for models (default: ../models/ravdess_full_emotions)"
            echo -e "  --batch_size N          Batch size (default: 32 for GPU, 16 for CPU)"
            echo -e "  --epochs N              Number of epochs (default: 50)"
            echo -e "  --learning_rate N       Learning rate (default: 5e-5)"
            echo -e "  --device DEVICE         Device to use (default: cuda if available, cpu otherwise)"
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
echo -e "${GREEN}Emotion subset:${NC} Full 8 emotions"
echo -e "${GREEN}Optimizer:${NC} $OPTIMIZER"
echo -e "${GREEN}Weight decay:${NC} $WEIGHT_DECAY"
echo -e "${GREEN}Scheduler:${NC} $SCHEDULER"
echo -e "${GREEN}Use AMP:${NC} ${USE_AMP:+yes}"
echo -e "${GREEN}Cache waveforms:${NC} ${CACHE_WAVEFORMS:+yes}"
echo -e "${GREEN}Use attention:${NC} yes"
echo -e "${GREEN}Use spectrogram branch:${NC} yes"
echo -e "${GREEN}Use gender branch:${NC} yes"
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
# Use port 6008 to avoid conflicts
tensorboard --logdir=$TENSORBOARD_DIR --port=6008 --host=0.0.0.0 &
TENSORBOARD_PID=$!

# Check if TensorBoard started successfully
sleep 3
if ps -p $TENSORBOARD_PID > /dev/null; then
    # Print TensorBoard URL
    echo -e "${GREEN}TensorBoard running at http://localhost:6008${NC}"
    echo -e "${YELLOW}(Keep this terminal open to maintain the TensorBoard server)${NC}"
else
    echo -e "${YELLOW}TensorBoard failed to start. Training will continue without visualization.${NC}"
    TENSORBOARD_PID=""
fi
echo

# Start training
echo -e "${BLUE}Starting advanced training with full 8 emotions for all 50 epochs...${NC}"
echo -e "${YELLOW}This will take several hours to complete for maximum accuracy.${NC}"
echo

python train_ravdess_advanced.py \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE \
    --sample_rate $SAMPLE_RATE \
    --max_duration $MAX_DURATION \
    --audio_only \
    --speech_only \
    $CACHE_WAVEFORMS \
    --num_workers $NUM_WORKERS \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --weight_decay $WEIGHT_DECAY \
    --cosine_t0 $COSINE_T0 \
    --cosine_t_mult $COSINE_T_MULT \
    --clip_grad_norm $CLIP_GRAD \
    $USE_AMP \
    $USE_ATTENTION \
    $USE_CONTEXT_LAYERS \
    $USE_ATTENTION_HEADS \
    $USE_DROPOUT \
    $USE_SPECTROGRAM \
    $USE_GENDER \
    $FREEZE_EXTRACTOR \
    --disable_early_stopping \
    --save_checkpoint_steps 5

TRAINING_STATUS=$?

# Check if training completed successfully
if [ $TRAINING_STATUS -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to $OUTPUT_DIR${NC}"
    
    # Evaluate the final model
    if [ -f "$OUTPUT_DIR/final_model.pt" ]; then
        echo -e "${BLUE}Evaluating final model...${NC}"
        python ravdess_evaluate.py \
            --model_path "$OUTPUT_DIR/final_model.pt" \
            --dataset_root "$DATASET_ROOT" \
            --device $DEVICE
    else
        echo -e "${YELLOW}No final model found. Skipping evaluation.${NC}"
    fi
else
    echo -e "${RED}Training failed with status code $TRAINING_STATUS!${NC}"
fi

# Stop TensorBoard server
if [ ! -z "$TENSORBOARD_PID" ] && ps -p $TENSORBOARD_PID > /dev/null; then
    echo -e "${BLUE}Stopping TensorBoard...${NC}"
    kill $TENSORBOARD_PID
fi

# Print instruction to run inference
echo -e "${YELLOW}To run real-time inference with the trained model, use:${NC}"
echo -e "${BLUE}python ravdess_inference.py --model_path $OUTPUT_DIR/final_model.pt${NC}"

# Exit with the same status as the training
exit $TRAINING_STATUS 