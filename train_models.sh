#!/bin/bash
# Script to train multiple speech emotion recognition models
# This script trains several model variants with different configurations

set -e  # Exit on error

# Create output directories
mkdir -p models/fixed_base
mkdir -p models/fixed_enhanced
mkdir -p models/fixed_simple
mkdir -p models/fixed_ensemble

# Install required dependencies if missing
pip install -q librosa scikit-learn matplotlib pandas tqdm

# Training variables
DATASET_ROOT="./dataset"  # Using absolute path to the dataset in the project root
SAMPLE_RATE=16000
MAX_LENGTH=5
BATCH_SIZE=8
NUM_WORKERS=0  # Keep at 0 to avoid multiprocessing issues
BASE_LR=3e-5
EPOCHS=30

echo "===== Training Fixed Simple Model ====="
python src/train_fixed.py \
    --dataset_root $DATASET_ROOT \
    --output_dir models/fixed_simple \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --learning_rate 1e-4 \
    --lr_scheduler reduce_on_plateau

echo "===== Training Fixed Base Model ====="
python src/train_fixed.py \
    --dataset_root $DATASET_ROOT \
    --output_dir models/fixed_base \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --learning_rate $BASE_LR \
    --lr_scheduler cosine

echo "===== Training Fixed Enhanced Model ====="
python src/train_fixed.py \
    --dataset_root $DATASET_ROOT \
    --output_dir models/fixed_enhanced \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --learning_rate $BASE_LR \
    --lr_scheduler cosine \
    --use_enhanced_model

# Train ensemble model with longer training
echo "===== Training Fixed Ensemble Model ====="
python src/train_fixed.py \
    --dataset_root $DATASET_ROOT \
    --output_dir models/fixed_ensemble \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs 40 \
    --learning_rate 2e-5 \
    --lr_scheduler cosine \
    --use_enhanced_model

echo "===== All models trained successfully! ====="
echo "Model checkpoints saved in the 'models/' directory" 