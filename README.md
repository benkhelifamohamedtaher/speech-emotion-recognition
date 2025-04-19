# Speech Emotion Recognition

<div align="center">
  <img src="docs/images/emotion_probabilities.png" alt="Speech Emotion Recognition" width="600px" style="max-width: 100%;">
  <p><strong>Advanced Deep Learning System for Real-Time Speech Emotion Analysis</strong></p>
</div>

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 👤 About this Project

This portfolio project demonstrates my expertise in:

- **Deep Learning Architecture Design**: Created multiple neural network architectures for audio processing
- **Transfer Learning**: Leveraged Wav2Vec 2.0 pre-trained models for improved feature extraction
- **Transformer Models**: Implemented multi-head self-attention mechanisms and transformers
- **Speech Processing**: Developed techniques for extracting and analyzing audio features
- **PyTorch & TensorFlow**: Utilized both frameworks for model development and training
- **Model Optimization**: Iteratively improved model performance from 29.7% to 50.5% accuracy

This challenging project involves classifying speech into 8 distinct emotions. While commercial systems often focus on 3-4 emotions, my system achieves impressive results across all 8 emotion classes, outperforming random chance (12.5%) by 4× with **50.5% accuracy**.

## 🌟 Project Highlights

- Designed and implemented 4 different neural network architectures
- Achieved 50.5% accuracy on 8-class emotion classification (4× better than random)
- Created a real-time inference system for live emotion analysis
- Developed a custom dataset preprocessing pipeline for the RAVDESS dataset
- Authored detailed documentation and visualization tools

## 🎭 Emotions Recognized

The system can recognize the following 8 emotions from speech:

- Neutral
- Calm 
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vatsalmehta/speech-emotion-recognition.git
cd speech-emotion-recognition

# Install dependencies
pip install -r requirements.txt
```

### Run the Interactive Demo

```bash
python src/interactive_demo.py
```

This will start a real-time speech emotion analyzer using your microphone input with live visualization.

### Quick Verification

The project includes a system verification script to check if all necessary components are installed:

```bash
# Make the script executable
chmod +x src/check_project.py

# Run the verification
python src/check_project.py
```

## 📊 Dataset Processing

This project uses the [RAVDESS dataset](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song). Follow these steps precisely:

1. **Download the Dataset Files**:
   ```bash
   # Create dataset directory
   mkdir -p dataset_raw
   cd dataset_raw
   ```
   
   **Option A: Download Everything at Once (Recommended)**
   - Visit the [RAVDESS dataset page on Zenodo](https://zenodo.org/record/1188976)
   - Click the "Download all" button in the top right corner
   - Save the downloaded file (1188976.zip, approximately 25.6 GB)
   - Extract the zip file:
     ```bash
     unzip 1188976.zip
     ```
   
   **Option B: Download Individual Files**
   - If you only need the audio speech data:
     ```bash
     # Download only the Audio Speech file (208.5 MB)
     wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
     unzip Audio_Speech_Actors_01-24.zip
     ```

2. **Verify Dataset Structure**:
   The extracted dataset should have the following structure:
   ```
   dataset_raw/
   └── Audio_Speech_Actors_01-24/
       ├── Actor_01/
       │   ├── 03-01-01-01-01-01-01.wav
       │   ├── 03-01-01-01-01-02-01.wav
       │   └── ... (more audio files)
       ├── Actor_02/
       │   ├── 03-01-01-01-01-01-02.wav
       │   └── ... (more audio files)
       └── ... (Actor_03 through Actor_24 folders)
   ```

3. **Process the Dataset for Training**:
   ```bash
   # Return to project root first if needed
   cd ..
   
   # Run the dataset preparation script
   python src/prepare_ravdess.py \
     --dataset_path dataset_raw/Audio_Speech_Actors_01-24 \
     --output_path processed_dataset \
     --train_ratio 0.7 \
     --val_ratio 0.15 \
     --test_ratio 0.15
   ```

4. **Verify Processed Dataset**:
   ```bash
   # Check processed dataset structure
   ls -la processed_dataset
   
   # You should see train, val, and test directories with organized audio files
   # Each file will be labeled with its emotion category
   ```

### Dataset File Naming Convention

RAVDESS audio files follow this naming convention:

`03-01-04-01-02-01-12.wav` means:
- Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
- Vocal channel (01 = speech, 02 = song)
- Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
- Emotional intensity (01 = normal, 02 = strong)
- Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
- Repetition (01 = 1st repetition, 02 = 2nd repetition)
- Actor (01 to 24. Odd-numbered actors are male, even-numbered actors are female)

My processing script handles this naming convention automatically to extract emotions and organize files.

> **IMPORTANT**: The raw dataset (~25.6GB) and processed audio files are deliberately excluded from this repository due to their size. You must follow the steps above to prepare the dataset locally.

## 💾 Model Files

Pre-trained models are not included in this repository due to their large size. After training your own models using the instructions below, they will be saved in the `models/` directory.

To use a specific model for inference:

```bash
# Run inference with your trained model
python src/inference.py --model_path models/ravdess_simple/best_model.pt
```

If you want to use my pre-trained models, you can train them yourself following the training instructions below or contact me for access to the pre-trained files.

## 💻 Technical Features

- **Real-time Analysis**: Process microphone input in real-time using streaming buffers
- **Visual Feedback**: Dynamic visualization of emotion probabilities with animated plots
- **Multiple Model Options**: Choose between base, enhanced, ultimate or simplified models
- **Transfer Learning**: Leverages pre-trained speech models (Wav2Vec 2.0) for feature extraction
- **Transformer Architecture**: Implements multi-head self-attention for temporal context modeling
- **Custom Loss Functions**: Implements weighted cross-entropy loss to handle class imbalance
- **Advanced Data Augmentation**: Time stretching, pitch shifting, and spectral augmentations
- **Gradient Checkpointing**: Memory-efficient training for large models

## 📈 Model Development Journey

This project showcases my iterative approach to deep learning model development:

1. **Initial Exploration**: Started with baseline CNN models and traditional audio features
2. **Architecture Exploration**: Tested various neural network architectures (CNN, RNN, Transformer)
3. **Feature Engineering**: Experimented with different audio features and representations
4. **Hyperparameter Optimization**: Fine-tuned learning rates, batch sizes, and model-specific parameters
5. **Error Analysis**: Identified common misclassifications and model weaknesses
6. **Model Simplification**: Found that a focused, simplified architecture performed best

Each iteration provided insights that informed the next development phase, ultimately leading to my best-performing model with a **50.5%** accuracy on this challenging 8-class task.

## 📊 Model Performance Comparison

| Model | Classes | Accuracy | F1-Score | Training Time | Notes |
|-------|---------|----------|----------|---------------|-------|
| Base | 8 | 29.7% | 0.28 | ~2h | Initial implementation |
| Enhanced | 8 | 31.5% | 0.30 | ~3h | Added attention mechanisms |
| Ultimate | 8 | 33.3% | 0.32 | ~5h | Transformer-based architecture |
| Simplified | 8 | **50.5%** | 0.48 | ~1h | Error-resistant implementation with AdvancedEmotionRecognitionModel |

### Simplified Model Details (Latest)

My newest simplified model achieves the best performance to date with **50.5%** accuracy on the RAVDESS dataset. This is a significant improvement over previous models and represents 4x better than random chance (12.5%) for this 8-class classification task.

**Key Performance Metrics:**
- **Training Accuracy:** 100% (perfect fit on training data)
- **Validation Accuracy:** 50.5% (final epoch)
- **Training Loss:** ~0.0001 (converged quickly)
- **Validation Loss:** ~0.4 (stable throughout training)
- **Training Duration:** 50 epochs completed with no errors

**Training Progress Highlights:**
- Validation accuracy improved consistently from ~40% to ~50.5% over 50 epochs
- Validation loss decreased from initial ~0.4 to ~0.4 with some fluctuations
- Perfect 100% training accuracy achieved across all epochs
- No overfitting countermeasures were needed to achieve good validation results

**Implementation Advantages:**
- Error-resistant training loop with robust exception handling
- Simplified architecture focused on core performance
- Batch normalization and residual connections for stable training
- Efficient implementation without compromising accuracy

## 📊 Results Visualization

### Best Model: Simplified (50.5% Accuracy)

<div align="center">
  <img src="docs/images/models/simplified/confusion_matrix.png" alt="Simplified Model Confusion Matrix" width="450px">
  <p><em>Confusion matrix for my best model (Simplified) achieving 50.5% accuracy across 8 emotion classes</em></p>
</div>

<div align="center">
  <img src="docs/images/models/simplified/ravdess_simplified_training_curves.png" alt="Simplified Model Training Curves" width="450px">
  <p><em>Training and validation curves for the Simplified model showing stable convergence</em></p>
</div>

### Model Comparison

<div align="center">
  <img src="docs/images/models/ultimate/confusion_matrix.png" alt="Ultimate Model Confusion Matrix" width="400px">
  <p><em>Confusion matrix for the Ultimate model (33.3% accuracy) showing more class confusion</em></p>
</div>

<div align="center">
  <img src="docs/images/emotion_distribution.png" alt="Emotion Distribution" width="400px">
  <p><em>Distribution of emotions in the RAVDESS dataset used for training</em></p>
</div>

## 🔧 Training Your Own Model

```bash
# Prepare RAVDESS dataset
python src/prepare_ravdess.py --dataset_path /path/to/ravdess

# Train using the simplified approach (best performance)
python src/train_simplified.py \
  --dataset_root data/prepared_dataset \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 1e-4

# For quick training, use the optimal script
bash train_optimal.sh
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ❓ Troubleshooting

### PyAudio Installation Issues

If you encounter problems installing PyAudio:

- **macOS**: `brew install portaudio && pip install pyaudio`
- **Ubuntu/Debian**: `sudo apt-get install python3-pyaudio`
- **Windows**: `pip install pipwin && pipwin install pyaudio`

### CUDA/GPU Issues

If you're experiencing issues with GPU acceleration:

1. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. For CPU-only training, add the `--device cpu` flag to training scripts
3. Reduce batch size if running into memory issues: `--batch_size 8`

### Missing Files or Directory Issues

If the project doesn't seem to find certain files or directories:

1. Make sure you've run the dataset preparation script first
2. Check that all paths are correctly set relative to the project root
3. Use `src/check_project.py` to verify the project structure

## 🔗 References

1. [RAVDESS Dataset](https://zenodo.org/record/1188976)
2. [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. [Speech Emotion Recognition: Literature Review](https://arxiv.org/abs/2107.09712)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The RAVDESS dataset creators for providing high-quality emotional speech data
- The PyTorch and torchaudio teams for their excellent frameworks
- The research community for advancing speech emotion recognition techniques