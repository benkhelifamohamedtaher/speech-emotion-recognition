#!/bin/bash
# Script to train RAVDESS-specific speech emotion recognition model
# This properly handles the RAVDESS dataset format and emotion categories

set -e  # Exit on error

# Create output directory
mkdir -p models/ravdess

# Install required dependencies if missing
pip install -q librosa scikit-learn matplotlib pandas tqdm

# Set dataset path - use the original RAVDESS dataset path
DATASET_ROOT="/Users/vatsalmehta/Developer/Real-Time Speech Emotion Recognition/dataset/Audio_Speech_Actors_01-24"

# Training parameters
SAMPLE_RATE=16000
MAX_LENGTH=5
BATCH_SIZE=32
NUM_WORKERS=0  # Keep at 0 to avoid multiprocessing issues
LEARNING_RATE=3e-4
EPOCHS=50

echo "===== Training Full RAVDESS Model (8 emotions) ====="
python src/train_ravdess.py \
    --dataset_dir "$DATASET_ROOT" \
    --output_dir models/ravdess/full \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --seed 42

echo "===== Training Simplified RAVDESS Model (4 emotions) ====="
python src/train_ravdess.py \
    --dataset_dir "$DATASET_ROOT" \
    --output_dir models/ravdess/simplified \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --use_simplified_emotions \
    --seed 42

echo "===== All models trained successfully! ====="
echo "Model checkpoints saved in the 'models/ravdess' directory" 