#!/bin/bash
# Script to train the RAVDESS emotion recognition model

# Define paths and parameters
DATASET_ROOT="./dataset/RAVDESS"
OUTPUT_DIR="./models/ravdess"
SAMPLE_RATE=16000
MAX_LENGTH=48000  # 3 seconds at 16kHz
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=50
NUM_WORKERS=4

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset directory $DATASET_ROOT does not exist"
    echo "Please download the RAVDESS dataset and extract it to $DATASET_ROOT"
    exit 1
fi

# Create main output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Create specific output directories for each emotion set
FULL_OUTPUT_DIR="${OUTPUT_DIR}/full"
SIMPLIFIED_OUTPUT_DIR="${OUTPUT_DIR}/simplified"
mkdir -p $FULL_OUTPUT_DIR
mkdir -p $SIMPLIFIED_OUTPUT_DIR

echo "==============================================="
echo "=== Training RAVDESS Speech Emotion Models ==="
echo "==============================================="
echo "Dataset: $DATASET_ROOT"
echo "Main output directory: $OUTPUT_DIR"
echo "Sample rate: $SAMPLE_RATE Hz"
echo "Max audio length: $MAX_LENGTH samples (${MAX_LENGTH}/${SAMPLE_RATE} seconds)"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "==============================================="

# Train with full emotion set
echo ""
echo "=== Training model with full emotion set (8 emotions) ==="
echo "Output directory: $FULL_OUTPUT_DIR"
echo "Starting training..."

python src/train_ravdess_simple.py \
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
  --augment \
  --device cpu

# Train with simplified emotion set
echo ""
echo "=== Training model with simplified emotion set (4 emotions) ==="
echo "Output directory: $SIMPLIFIED_OUTPUT_DIR"
echo "Starting training..."

python src/train_ravdess_simple.py \
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
  --augment \
  --device cpu

echo ""
echo "==============================================="
echo "=== Training completed ==="
echo "Models saved to:"
echo "  - Full emotions: $FULL_OUTPUT_DIR"
echo "  - Simplified emotions: $SIMPLIFIED_OUTPUT_DIR"
echo "===============================================" 