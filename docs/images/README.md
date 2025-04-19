# Speech Emotion Recognition Visualizations

This directory contains visualizations and results from our Speech Emotion Recognition project.

## Directory Structure

- **models/** - Model-specific visualizations
  - **simplified/** - Best performing model (50.5% accuracy)
  - **ultimate/** - Complex transformer model (33.3% accuracy)
  - **enhanced/** - Model with attention mechanisms (31.5% accuracy)
  - **base/** - Initial implementation (29.7% accuracy)

- **emotion_distribution.png** - Distribution of emotions in the RAVDESS dataset
- **emotion_probabilities.png** - Real-time visualization of emotion probabilities
- **confusion_matrix.png** - General confusion matrix for reference

## Model Progression

The visualizations in this directory show our model development journey from the initial Base model to our best-performing Simplified model. Each subdirectory contains results specific to that model architecture.

### Key Findings

1. The Simplified model (50.5% accuracy) significantly outperformed more complex architectures
2. More complex architectures suffered from training instability and overfitting
3. The focused, error-resistant approach of the Simplified model proved most effective for this task

## Interactive Demo

For real-time visualizations, run the interactive demo:
```bash
python src/interactive_demo.py
``` 