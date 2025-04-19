#!/bin/bash
# Optimal training script for speech emotion recognition models
# This script prepares the dataset and trains high-accuracy models

set -e  # Exit on error

# Create output directories
mkdir -p processed_dataset
mkdir -p models/optimal/full
mkdir -p models/optimal/simplified

echo "=================================================="
echo "Speech Emotion Recognition - Optimal Training"
echo "=================================================="

# Step 1: Prepare dataset
echo "Step 1: Preparing RAVDESS dataset..."
python src/prepare_ravdess.py \
  --dataset_dir ./dataset/RAVDESS \
  --output_dir ./processed_dataset \
  --sample_rate 16000 \
  --duration 3.0

# Wait for dataset preparation to complete
echo "Dataset preparation completed!"
echo "=================================================="

# Step 2: Train full emotion model
echo "Step 2: Training model with full emotions (8 classes)..."
python src/optimal_train.py \
  --data_dir ./processed_dataset \
  --output_dir ./models/optimal/full \
  --feature_type melspec \
  --sample_rate 16000 \
  --duration 3.0 \
  --hidden_size 384 \
  --num_transformer_layers 6 \
  --num_attention_heads 12 \
  --ff_dim 1536 \
  --dropout 0.3 \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 5e-5 \
  --optimizer adamw \
  --loss focal \
  --focal_gamma 2.0 \
  --lr_scheduler one_cycle \
  --clip_grad 1.0 \
  --augment \
  --num_workers 4 \
  --device cpu \
  --early_stopping 10

# Step 3: Train simplified emotion model
echo "Step 3: Training model with simplified emotions (4 classes)..."
python src/optimal_train.py \
  --data_dir ./processed_dataset \
  --output_dir ./models/optimal/simplified \
  --simplified \
  --feature_type melspec \
  --sample_rate 16000 \
  --duration 3.0 \
  --hidden_size 256 \
  --num_transformer_layers 4 \
  --num_attention_heads 8 \
  --ff_dim 1024 \
  --dropout 0.2 \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --optimizer adamw \
  --loss focal \
  --focal_gamma 2.0 \
  --lr_scheduler one_cycle \
  --clip_grad 1.0 \
  --augment \
  --num_workers 4 \
  --device cpu \
  --early_stopping 10

echo "=================================================="
echo "Training completed successfully!"
echo "=================================================="
echo "Models saved to:"
echo "  - Full emotions (8 classes): ./models/optimal/full"
echo "  - Simplified emotions (4 classes): ./models/optimal/simplified"
echo ""
echo "To run inference with the trained models, use:"
echo "  python src/optimal_inference.py --model ./models/optimal/full/best_model.pt"
echo "  or"
echo "  python src/optimal_inference.py --model ./models/optimal/simplified/best_model.pt --simplified"
echo "==================================================" 