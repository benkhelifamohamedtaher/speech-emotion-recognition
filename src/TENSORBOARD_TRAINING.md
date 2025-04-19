# Using TensorBoard with Speech Emotion Recognition Training

This guide explains how to use TensorBoard to visualize and monitor the training process for speech emotion recognition models.

## Prerequisites

- TensorBoard (installed automatically by the training script if not available)
- Python 3.8 or newer
- PyTorch with CUDA support (for GPU acceleration)

## Running Training with TensorBoard

The enhanced training script `run_ravdess_advanced_training.sh` automatically configures and launches TensorBoard for you.

```bash
# Basic usage
./run_ravdess_advanced_training.sh

# With custom dataset location
./run_ravdess_advanced_training.sh --dataset_root /path/to/ravdess/dataset

# With custom output directory
./run_ravdess_advanced_training.sh --output_dir ../models/my_custom_model
```

When you run the script, it will:

1. Install TensorBoard if needed
2. Start a TensorBoard server on port 6006
3. Begin training the model with logging enabled
4. Automatically clean up the TensorBoard server when training completes

## Accessing TensorBoard

While training is running, open your web browser and navigate to:

```
http://localhost:6006
```

## What You Can Monitor

TensorBoard provides real-time visualizations of:

- Training and validation loss
- Accuracy metrics
- Learning rate changes
- Model architecture
- Attention weights visualization
- Emotion probability distributions

## Customizing TensorBoard

You can customize the TensorBoard experience by modifying the script parameters:

```bash
# Change TensorBoard port
# Edit the script and change --port=6006 to your preferred port
```

## Troubleshooting

If you encounter issues with TensorBoard:

1. Make sure port 6006 is not in use by another application
2. Check that the training script has permission to write to the logs directory
3. If TensorBoard crashes, you can restart it manually:

```bash
tensorboard --logdir=../models/ravdess_advanced/logs --port=6006
```

## Advanced Usage

### Using a Specific Device

```bash
# Use CPU explicitly
./run_ravdess_advanced_training.sh --device cpu

# Use specific GPU (if you have multiple)
CUDA_VISIBLE_DEVICES=1 ./run_ravdess_advanced_training.sh
```

### Training with Specific Emotion Subset

```bash
# Train with 4 basic emotions (neutral, happy, sad, angry)
./run_ravdess_advanced_training.sh --emotion_subset basic4

# Train with 6 emotions (adds fearful and surprised)
./run_ravdess_advanced_training.sh --emotion_subset basic6

# Train with all 8 RAVDESS emotions
./run_ravdess_advanced_training.sh --emotion_subset ""
```

## Comparing Multiple Runs

If you want to compare different model configurations:

1. Use different output directories for each run
2. Point TensorBoard to the parent directory:

```bash
tensorboard --logdir=../models
```

This will show all model runs in the same TensorBoard instance, allowing easy comparison. 