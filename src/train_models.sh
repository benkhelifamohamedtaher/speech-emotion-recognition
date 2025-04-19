#!/bin/bash
# Script to train multiple models

set -e  # Exit on error

# Create output directories
mkdir -p models/simple_model models/base_model models/enhanced_model models/ensemble_model

# Set global training parameters
DATASET_ROOT="../processed_dataset"
SAMPLE_RATE=16000
MAX_LENGTH=48000  # 3 seconds at 16kHz
BATCH_SIZE=16
NUM_WORKERS=0  # Use 0 to avoid multiprocessing issues
BASE_LR=0.0001
EPOCHS=30

echo "===== Training Fixed Simple Model ====="
python train_fixed.py \
  --dataset_root "$DATASET_ROOT" \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --learning_rate 3e-4 \
  --output_dir ./models/simple_model

echo "===== Training Fixed Base Model ====="
python train_fixed.py \
  --dataset_root "$DATASET_ROOT" \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --learning_rate 1e-4 \
  --output_dir ./models/base_model

echo "===== Training Fixed Enhanced Model ====="
python train_enhanced.py \
  --dataset_root "$DATASET_ROOT" \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --learning_rate 3e-5 \
  --lr_scheduler cosine \
  --output_dir ./models/enhanced_model

echo "===== Training Fixed Ensemble ====="
python train_fixed.py \
  --dataset_root "$DATASET_ROOT" \
  --sample_rate $SAMPLE_RATE \
  --max_length $MAX_LENGTH \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --learning_rate 2e-4 \
  --output_dir ./models/ensemble_model

echo "===== All models trained successfully! ====="
echo "Model checkpoints saved in the 'models/' directory" 