#!/bin/bash
# Enhanced Speech Emotion Recognition Project Runner
# This script runs the complete pipeline using the simplified training script that works well

set -e  # Exit on error

# Default paths
RAVDESS_PATH="dataset/Audio_Speech_Actors_01-24"
PROCESSED_PATH="processed_dataset"
MODEL_PATH="models/simple_model"
EVAL_PATH="evaluation_results"
INFERENCE_PATH="inference_results"
LOG_DIR="logs"

# Create directories
mkdir -p $PROCESSED_PATH
mkdir -p $MODEL_PATH
mkdir -p $EVAL_PATH
mkdir -p $INFERENCE_PATH
mkdir -p $LOG_DIR

echo "================================================"
echo "Enhanced Speech Emotion Recognition Pipeline"
echo "================================================"

# Step 1: Process RAVDESS dataset
echo ""
echo "Step 1: Processing RAVDESS dataset"
echo "--------------------------------"
if [ -d "$PROCESSED_PATH/train" ] && [ -d "$PROCESSED_PATH/test" ]; then
    echo "Dataset already processed. Skipping..."
else
    python src/process_ravdess.py --input_dir $RAVDESS_PATH --output_dir $PROCESSED_PATH --speech_only
fi

# Step 2: Train the model with simple training script
echo ""
echo "Step 2: Training the model with simple training"
echo "-------------------------------------------"
python src/simple_train.py \
    --dataset_root $PROCESSED_PATH \
    --output_dir $MODEL_PATH \
    --batch_size 8 \
    --epochs 20 \
    --learning_rate 3e-5 \
    --freeze_encoder

# Step 3: Evaluate the model
echo ""
echo "Step 3: Evaluating the model"
echo "--------------------------"
python src/evaluate_model.py \
    --model_path $MODEL_PATH/best_model.pt \
    --dataset_root $PROCESSED_PATH \
    --output_dir $EVAL_PATH \
    --batch_size 32 \
    --num_workers 0

# Step 4: Run inference on example files
echo ""
echo "Step 4: Running inference on example files"
echo "--------------------------------------"
python src/inference.py \
    --model_path $MODEL_PATH/best_model.pt \
    --dir $RAVDESS_PATH/Actor_01 \
    --output_dir $INFERENCE_PATH \
    --plot

echo ""
echo "================================================"
echo "Pipeline completed successfully!"
echo "================================================"
echo "Model saved at: $MODEL_PATH/best_model.pt"
echo "Evaluation results at: $EVAL_PATH"
echo "Inference examples at: $INFERENCE_PATH" 