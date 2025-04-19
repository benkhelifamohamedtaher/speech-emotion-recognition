#!/bin/bash

# Script to train the advanced model for speech emotion recognition

# Set main parameters
DATASET_ROOT="./dataset" # Root directory containing all audio datasets
OUTPUT_DIR="./models/advanced"
SAMPLE_RATE=16000
MAX_LENGTH=48000
BATCH_SIZE=16
LEARNING_RATE=1e-4
EPOCHS=30
NUM_WORKERS=4 # Adjust based on your system
DEVICE="cuda" # Use "cpu" if you don't have a GPU

# Create output directory
mkdir -p $OUTPUT_DIR

echo "====================================================="
echo "Starting advanced model training for speech emotion recognition"
echo "Dataset: $DATASET_ROOT"
echo "Output: $OUTPUT_DIR"
echo "====================================================="

# Train the model with simplified emotions (4 classes) first
echo "Training model with simplified emotion set (4 classes)..."
python src/train_advanced.py \
    --dataset_root $DATASET_ROOT \
    --emotion_set "simplified" \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --output_dir "${OUTPUT_DIR}/simplified" \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --lr_scheduler "one_cycle" \
    --mixed_precision \
    --augment \
    --use_transformer

echo "====================================================="
echo "Simplified model training complete!"
echo "====================================================="

# Train the model with full emotions (8 classes)
echo "Training model with full emotion set (8 classes)..."
python src/train_advanced.py \
    --dataset_root $DATASET_ROOT \
    --emotion_set "full" \
    --sample_rate $SAMPLE_RATE \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --output_dir "${OUTPUT_DIR}/full" \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --lr_scheduler "one_cycle" \
    --mixed_precision \
    --augment \
    --use_transformer

echo "====================================================="
echo "Full model training complete!"
echo "====================================================="

echo "All training processes completed successfully!"
echo "Models saved to $OUTPUT_DIR" 