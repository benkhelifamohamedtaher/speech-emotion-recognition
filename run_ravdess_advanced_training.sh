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

# Default parameters - Optimized for high accuracy
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../models/ravdess_advanced"
BATCH_SIZE=16
EPOCHS=50
LEARNING_RATE=1e-5 # Lower learning rate for better convergence
EMOTION_SUBSET="basic6"  # Use 6 basic emotions for best performance
SAMPLE_RATE=16000
MAX_DURATION=5.0
NUM_WORKERS=4
DEVICE="cuda"
# Higher model complexity for better accuracy
CONTEXT_LAYERS=3 
ATTENTION_HEADS=4
# Increased dropout for better generalization
DROPOUT_RATE=0.4
# Enable weight decay for better regularization
WEIGHT_DECAY=2e-5
# Use cosine annealing with longer cycle
SCHEDULER="cosine"
COSINE_T0=15
COSINE_T_MULT=2
# Optimizer
OPTIMIZER="adamw"
# Use gender branch for multi-task learning
USE_GENDER_BRANCH="--use_gender_branch"
# Use spectrogram branch for additional features
USE_SPECTROGRAM_BRANCH="--use_spectrogram_branch"
# Freeze feature extractor
FREEZE_EXTRACTOR="--freeze_extractor"
# Use mixed precision
USE_AMP="--use_amp"

# Check if PyTorch is available
if ! python -c "import torch" &>/dev/null; then
    echo -e "${RED}PyTorch is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if transformers is available
if ! python -c "import transformers" &>/dev/null; then
    echo -e "${RED}Transformers library is not installed. Please install it first.${NC}"
    echo -e "${YELLOW}Try: pip install transformers==4.30.2 torch==2.0.1${NC}"
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
    BATCH_SIZE=8
    # Still maintain good model complexity for high accuracy, but reduce slightly for CPU
    CONTEXT_LAYERS=2
    ATTENTION_HEADS=2
    echo -e "${YELLOW}Adjusted model complexity for CPU training while maintaining high accuracy${NC}"
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
        --context_layers)
            CONTEXT_LAYERS="$2"
            shift
            shift
            ;;
        --attention_heads)
            ATTENTION_HEADS="$2"
            shift
            shift
            ;;
        --dropout_rate)
            DROPOUT_RATE="$2"
            shift
            shift
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
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
            echo -e "  --learning_rate N     Learning rate (default: 1e-5)"
            echo -e "  --device DEVICE       Device to use (default: cuda if available, cpu otherwise)"
            echo -e "  --emotion_subset SET  Emotion subset to use (default: basic6)"
            echo -e "  --sample_rate N       Audio sample rate (default: 16000)"
            echo -e "  --context_layers N    Number of context layers (default: 3 for GPU, 2 for CPU)"
            echo -e "  --attention_heads N   Number of attention heads (default: 4 for GPU, 2 for CPU)"
            echo -e "  --dropout_rate N      Dropout rate (default: 0.4)"
            echo -e "  --weight_decay N      Weight decay (default: 2e-5)"
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
echo -e "${GREEN}Context layers:${NC} $CONTEXT_LAYERS"
echo -e "${GREEN}Attention heads:${NC} $ATTENTION_HEADS"
echo -e "${GREEN}Dropout rate:${NC} $DROPOUT_RATE"
echo -e "${GREEN}Weight decay:${NC} $WEIGHT_DECAY"
echo -e "${GREEN}Optimizer:${NC} $OPTIMIZER"
echo -e "${GREEN}Scheduler:${NC} $SCHEDULER"
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
# Use a different port to avoid conflicts
tensorboard --logdir=$TENSORBOARD_DIR --port=6007 --host=0.0.0.0 &
TENSORBOARD_PID=$!

# Check if TensorBoard started successfully
sleep 3
if ps -p $TENSORBOARD_PID > /dev/null; then
    # Print TensorBoard URL
    echo -e "${GREEN}TensorBoard running at http://localhost:6007${NC}"
    echo -e "${YELLOW}(Keep this terminal open to maintain the TensorBoard server)${NC}"
else
    echo -e "${YELLOW}TensorBoard failed to start. Training will continue without visualization.${NC}"
    TENSORBOARD_PID=""
fi
echo

# Start training
echo -e "${BLUE}Starting training with high-accuracy configuration...${NC}"
echo

python train_ravdess_advanced.py \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
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
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --weight_decay $WEIGHT_DECAY \
    --context_layers $CONTEXT_LAYERS \
    --attention_heads $ATTENTION_HEADS \
    --dropout_rate $DROPOUT_RATE \
    --cosine_t0 $COSINE_T0 \
    --cosine_t_mult $COSINE_T_MULT \
    $USE_GENDER_BRANCH \
    $USE_SPECTROGRAM_BRANCH \
    $FREEZE_EXTRACTOR \
    $USE_AMP

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
            --dataset_root "$DATASET_ROOT" \
            --device $DEVICE
    else
        echo -e "${YELLOW}No best model found. Skipping evaluation.${NC}"
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
echo -e "${BLUE}python ravdess_inference.py --model_path $OUTPUT_DIR/best_model.pt${NC}"

# Exit with the same status as the training
exit $TRAINING_STATUS 