#!/bin/bash
# Speech Emotion Recognition Project Runner
# This script runs the complete pipeline from data preparation to training and evaluation

set -e  # Exit on error

# Default paths
RAVDESS_PATH="dataset/Audio_Speech_Actors_01-24"
PROCESSED_PATH="processed_dataset"
MODEL_PATH="models/enhanced_model"
EVAL_PATH="evaluation_results"
LOG_DIR="logs"

# Create directories
mkdir -p $PROCESSED_PATH
mkdir -p $MODEL_PATH
mkdir -p $EVAL_PATH
mkdir -p $LOG_DIR

echo "=================================="
echo "Speech Emotion Recognition Pipeline"
echo "=================================="

# Step 1: Process RAVDESS dataset
echo ""
echo "Step 1: Processing RAVDESS dataset"
echo "--------------------------------"
python src/process_ravdess.py --input_dir $RAVDESS_PATH --output_dir $PROCESSED_PATH --speech_only

# Step 2: Train the enhanced model
echo ""
echo "Step 2: Training the enhanced model"
echo "--------------------------------"
python src/train_enhanced.py \
    --dataset_root $PROCESSED_PATH \
    --output_dir $MODEL_PATH \
    --batch_size 16 \
    --epochs 30 \
    --learning_rate 3e-5 \
    --lr_scheduler cosine \
    --weight_decay 1e-5 \
    --vad_weight 0.2 \
    --freeze_encoder \
    --augment \
    --sample_rate 16000 \
    --max_length 48000 \
    --num_workers 0 \
    --use_ema

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

# Step 4: Export model (optional)
echo ""
echo "Step 4: Exporting the model"
echo "-------------------------"
python src/export.py \
    --model_path $MODEL_PATH/best_model.pt \
    --output_path $MODEL_PATH \
    --export_onnx

# Step 5: Run inference on example files
echo ""
echo "Step 5: Running inference on example files"
echo "--------------------------------------"
python src/inference.py \
    --model_path $MODEL_PATH/best_model.pt \
    --dir $RAVDESS_PATH/Actor_01 \
    --output_dir $EVAL_PATH/samples \
    --plot

echo ""
echo "==========================="
echo "Pipeline completed successfully!"
echo "===========================" 