# Advanced RAVDESS Emotion Recognition Training

This document provides instructions for training high-accuracy speech emotion recognition models using the RAVDESS dataset.

## Dataset

The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset is a high-quality multimodal dataset for emotion recognition. It consists of:
- 24 professional actors (12 female, 12 male)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprise
- Approximately 7,356 total files (including audio-only, audio-visual, and video-only)

### File Naming Convention

RAVDESS files follow a specific naming convention:
```
XX-XX-XX-XX-XX-XX-XX.wav
```

Where:
1. Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
2. Vocal channel (01 = speech, 02 = song)
3. Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
4. Emotional intensity (01 = normal, 02 = strong)
5. Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
6. Repetition (01 = 1st repetition, 02 = 2nd repetition)
7. Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)

## Model Architecture

Our advanced model architecture includes:
- Wav2Vec 2.0 feature extraction
- Context transformer layers for temporal modeling
- Multi-head attention pooling
- Dual-branch architecture with spectrogram features
- Gender classification as an auxiliary task

## Training Requirements

- Python 3.8+
- PyTorch 1.10+
- torchaudio 0.10+
- Transformers library for Wav2Vec 2.0
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

## Dataset Preparation

1. Download the RAVDESS dataset from the [official website](https://zenodo.org/record/1188976)
2. Extract the audio files to a directory structure like:
   ```
   dataset/
     RAVDESS/
       Actor_01/
         03-01-01-01-01-01-01.wav
         03-01-01-01-01-02-01.wav
         ...
       Actor_02/
         ...
   ```

## Training Process

We use a comprehensive training approach:
1. Data split by actor to ensure generalization
2. Advanced data augmentation techniques
3. Mixed precision training for speed
4. Cosine annealing learning rate scheduling
5. Multi-task learning with gender classification

## Training the Model

Run the training script:

```bash
cd src
./train_ravdess_advanced.sh --dataset_root ../dataset/RAVDESS
```

### Training Options

You can customize training with various parameters:
- `--output_dir`: Where to save the trained model
- `--batch_size`: Batch size (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 3e-5)
- `--device`: Computing device (default: cuda if available)
- `--emotion_subset`: Subset of emotions (basic4, basic6, or all 8)

## Optimal Training Configuration

For best results, we recommend:
- Using 'basic6' emotion subset (neutral, happy, sad, angry, fearful, surprised)
- Train for at least 50 epochs
- Use Adam optimizer with weight decay
- Freeze feature extractor for stability
- Use multi-task learning with gender classification

## Evaluation

After training, evaluate the model:

```bash
./ravdess_evaluate.py --model_path ../models/ravdess_advanced/best_model.pt --dataset_root ../dataset/RAVDESS
```

This will generate:
- Confusion matrix
- ROC curves
- Precision, recall, and F1 scores
- Error analysis report

## Real-Time Inference

To run real-time inference:

```bash
./ravdess_inference.py --model_path ../models/ravdess_advanced/best_model.pt
```

This will capture audio from your microphone and display the predicted emotions in real-time with a visualization window.

## Performance Expectations

With the optimal configuration, you can expect:
- Overall accuracy: 85-92% (depending on emotion subset)
- High F1 scores for anger, happiness, and sadness
- Lower performance on subtler emotions like calm and neutral

## Troubleshooting

Common issues:
- **GPU out of memory**: Reduce batch size or model complexity
- **Audio capture issues**: Check microphone permissions and PyAudio installation
- **Low accuracy**: Ensure dataset is properly structured and try longer training
- **Overfitting**: Add more regularization or data augmentation 