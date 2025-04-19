# %% [markdown]
# # 📊 RAVDESS Dataset Preparation
# 
# This notebook documents the process of preparing the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset for our speech emotion recognition project.

# %% [markdown]
# ## Dataset Overview
# 
# The RAVDESS dataset is a validated multimodal database of emotional speech and song. It contains recordings from 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent.
# 
# **Key dataset characteristics:**
# 
# - **Emotions:** 8 distinct emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
# - **Intensity levels:** Normal and strong intensity (except for neutral emotion)
# - **File format:** High-quality audio recordings (48kHz, 16-bit)
# - **Statements:** Two different statements per emotion
# - **Repetitions:** Two repetitions of each statement
# - **Actors:** 24 professional actors (12 female, 12 male)
# - **Total files:** Approximately 1,440 audio files

# %%
# Import libraries for dataset exploration
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
import IPython.display as ipd

# Set paths
DATASET_PATH = "../dataset_raw/Audio_Speech_Actors_01-24"
OUTPUT_PATH = "../processed_dataset"

# %% [markdown]
# ## File Naming Convention
# 
# Each file in the RAVDESS dataset follows a specific naming convention:
# 
# **Format:** `03-01-04-01-02-01-12.wav`
# 
# The positions represent:
# 1. Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
# 2. Vocal channel (01 = speech, 02 = song)
# 3. Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
# 4. Emotional intensity (01 = normal, 02 = strong)
# 5. Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
# 6. Repetition (01 = 1st repetition, 02 = 2nd repetition)
# 7. Actor (01 to 24. Odd-numbered actors are male, even-numbered actors are female)

# %%
# Define mapping between emotion IDs and names
EMOTION_MAPPING = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to parse RAVDESS filename
def parse_ravdess_filename(filename):
    """Extract metadata from RAVDESS filename format"""
    parts = filename.split('.')[0].split('-')
    
    metadata = {
        'modality': parts[0],
        'vocal_channel': parts[1],
        'emotion_id': parts[2],
        'emotion': EMOTION_MAPPING[parts[2]],
        'intensity': 'normal' if parts[3] == '01' else 'strong',
        'statement': parts[4],
        'repetition': parts[5],
        'actor_id': parts[6],
        'gender': 'male' if int(parts[6]) % 2 == 1 else 'female'
    }
    
    return metadata

# %% [markdown]
# ## Dataset Exploration
# 
# Let's explore the dataset to understand its structure and distribution.

# %%
# Function to collect file information
def explore_dataset(dataset_path):
    """Collect information about all audio files in the dataset"""
    file_data = []
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        print("Please download and extract the RAVDESS dataset first.")
        return pd.DataFrame()
    
    # Collect all audio files
    for actor_dir in sorted(os.listdir(dataset_path)):
        actor_path = os.path.join(dataset_path, actor_dir)
        
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    # Parse metadata from filename
                    metadata = parse_ravdess_filename(filename)
                    
                    # Add file path
                    metadata['file_path'] = os.path.join(actor_path, filename)
                    
                    # Add to data list
                    file_data.append(metadata)
    
    # Create DataFrame
    df = pd.DataFrame(file_data)
    
    return df

# Run exploration (commented out to prevent execution on non-existent paths)
# df = explore_dataset(DATASET_PATH)
# print(f"Total files: {len(df)}")
# df.head()

# %% [markdown]
# ## Distribution Analysis
# 
# Let's analyze the distribution of emotions, intensities, and gender in the dataset.

# %%
# Visualization functions (will execute if you have the dataset)

def plot_emotion_distribution(df):
    """Plot distribution of emotions in the dataset"""
    plt.figure(figsize=(12, 6))
    emotion_counts = df['emotion'].value_counts().reindex(EMOTION_MAPPING.values())
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
    plt.title('Distribution of Emotions in RAVDESS Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../docs/images/emotion_distribution.png')
    plt.show()

def plot_gender_distribution(df):
    """Plot distribution of gender in the dataset"""
    plt.figure(figsize=(8, 6))
    gender_counts = df['gender'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values)
    plt.title('Gender Distribution in RAVDESS Dataset')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_intensity_by_emotion(df):
    """Plot distribution of intensity by emotion"""
    # Filter out neutral emotions (which don't have intensity variation)
    df_intensity = df[df['emotion'] != 'neutral'].copy()
    
    plt.figure(figsize=(12, 6))
    intensity_counts = pd.crosstab(df_intensity['emotion'], df_intensity['intensity'])
    intensity_counts.plot(kind='bar', stacked=True)
    plt.title('Emotion Intensity Distribution in RAVDESS Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Uncomment to run visualizations if dataset is available
# plot_emotion_distribution(df)
# plot_gender_distribution(df)
# plot_intensity_by_emotion(df)

# %% [markdown]
# ## Audio Visualization
# 
# Let's visualize some audio samples from the dataset to understand the characteristics of different emotions.

# %%
# Function to visualize waveform and spectrogram
def visualize_audio(file_path, emotion):
    """Visualize audio waveform and spectrogram for a given file"""
    plt.figure(figsize=(15, 10))
    
    # Load audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f'Waveform: {emotion.title()} Emotion')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title(f'Spectrogram: {emotion.title()} Emotion')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    # Display audio player
    display(ipd.Audio(y, rate=sr))

# Example usage (uncomment if dataset is available)
# Example files for different emotions
# sample_files = df.groupby('emotion').first()['file_path'].tolist()
# for file_path in sample_files[:3]:  # Visualize first 3 emotions
#     metadata = parse_ravdess_filename(os.path.basename(file_path))
#     visualize_audio(file_path, metadata['emotion'])

# %% [markdown]
# ## Data Preparation
# 
# Now, let's prepare the dataset for training by:
# 1. Creating training, validation, and test splits
# 2. Extracting audio features
# 3. Organizing files into an appropriate directory structure

# %%
def prepare_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare RAVDESS dataset for training
    
    Args:
        dataset_path: Path to raw RAVDESS dataset
        output_path: Path to store processed data
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
    """
    # Check ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Collect dataset information
    df = explore_dataset(dataset_path)
    
    if len(df) == 0:
        print("No files found. Exiting.")
        return
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    
    # Create split by actors to prevent data leakage
    actors = sorted(df['actor_id'].unique())
    
    # Shuffle actors with fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(actors)
    
    # Split actors
    n_actors = len(actors)
    n_train = int(n_actors * train_ratio)
    n_val = int(n_actors * val_ratio)
    
    train_actors = actors[:n_train]
    val_actors = actors[n_train:n_train+n_val]
    test_actors = actors[n_train+n_val:]
    
    # Create splits
    train_df = df[df['actor_id'].isin(train_actors)]
    val_df = df[df['actor_id'].isin(val_actors)]
    test_df = df[df['actor_id'].isin(test_actors)]
    
    print(f"Training set: {len(train_df)} files from {len(train_actors)} actors")
    print(f"Validation set: {len(val_df)} files from {len(val_actors)} actors")
    print(f"Test set: {len(test_df)} files from {len(test_actors)} actors")
    
    # Function to process and copy files
    def process_files(files_df, split_name):
        """Process files for a given split"""
        output_dir = os.path.join(output_path, split_name)
        
        # Create emotion directories
        for emotion in EMOTION_MAPPING.values():
            os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)
        
        # Copy files to appropriate directories
        for _, row in files_df.iterrows():
            emotion = row['emotion']
            src_path = row['file_path']
            dst_path = os.path.join(output_dir, emotion, os.path.basename(src_path))
            
            # Copy file (using os.system for simplicity, could use shutil.copy)
            os.system(f'cp "{src_path}" "{dst_path}"')
    
    # Process each split
    process_files(train_df, 'train')
    process_files(val_df, 'val')
    process_files(test_df, 'test')
    
    print("Dataset preparation completed successfully!")

# Example call (commented out to prevent execution)
# prepare_dataset(DATASET_PATH, OUTPUT_PATH)

# %% [markdown]
# ## Data Analysis and Validation
# 
# After preparing the dataset, let's analyze it to validate our splits and ensure the data is ready for model training.

# %%
def validate_prepared_dataset(output_path):
    """Validate prepared dataset structure and distribution"""
    splits = ['train', 'val', 'test']
    emotion_counts = {}
    
    for split in splits:
        split_path = os.path.join(output_path, split)
        emotion_counts[split] = {}
        
        # Count files by emotion
        for emotion in EMOTION_MAPPING.values():
            emotion_dir = os.path.join(split_path, emotion)
            if os.path.exists(emotion_dir):
                count = len(os.listdir(emotion_dir))
                emotion_counts[split][emotion] = count
            else:
                emotion_counts[split][emotion] = 0
    
    # Create DataFrame for analysis
    analysis_data = []
    for split, emotions in emotion_counts.items():
        for emotion, count in emotions.items():
            analysis_data.append({
                'split': split,
                'emotion': emotion,
                'count': count
            })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Plot distribution
    plt.figure(figsize=(15, 8))
    sns.barplot(x='emotion', y='count', hue='split', data=analysis_df)
    plt.title('Emotion Distribution Across Splits')
    plt.xlabel('Emotion')
    plt.ylabel('File Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Check class balance
    total_files = analysis_df.groupby('emotion')['count'].sum()
    print("Total files by emotion:")
    print(total_files)
    
    imbalance = total_files.max() / total_files.min()
    print(f"\nClass imbalance ratio (max/min): {imbalance:.2f}")
    
    return analysis_df

# Example call (commented out to prevent execution)
# validate_prepared_dataset(OUTPUT_PATH)

# %% [markdown]
# ## Sample Audio Feature Extraction
# 
# Let's explore some audio feature extraction techniques that will be useful for our models.

# %%
def extract_features(file_path):
    """Extract common audio features from an audio file"""
    # Load audio file
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract features
    features = {}
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    features['mel_spectrogram'] = librosa.power_to_db(mel_spec)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc'] = mfcc
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma'] = chroma
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['contrast'] = contrast
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr'] = zcr
    
    return features

def visualize_features(features, emotion):
    """Visualize extracted audio features"""
    plt.figure(figsize=(15, 12))
    
    # Plot Mel spectrogram
    plt.subplot(3, 2, 1)
    librosa.display.specshow(features['mel_spectrogram'], x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram ({emotion})')
    
    # Plot MFCC
    plt.subplot(3, 2, 2)
    librosa.display.specshow(features['mfcc'], x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC ({emotion})')
    
    # Plot Chroma
    plt.subplot(3, 2, 3)
    librosa.display.specshow(features['chroma'], x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title(f'Chroma ({emotion})')
    
    # Plot Spectral contrast
    plt.subplot(3, 2, 4)
    librosa.display.specshow(features['contrast'], x_axis='time')
    plt.colorbar()
    plt.title(f'Spectral Contrast ({emotion})')
    
    # Plot Zero crossing rate
    plt.subplot(3, 2, 5)
    plt.plot(features['zcr'].T)
    plt.title(f'Zero Crossing Rate ({emotion})')
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    
    plt.tight_layout()
    plt.show()

# Example usage (commented out to prevent execution)
# Find a sample file for demonstration
# if len(df) > 0:
#     sample_file = df[df['emotion'] == 'angry'].iloc[0]['file_path']
#     features = extract_features(sample_file)
#     visualize_features(features, 'angry')

# %% [markdown]
# ## Feature Comparison Across Emotions
# 
# Let's compare some key audio features across different emotions to understand what makes each emotion distinct.

# %%
def compare_emotion_features(df):
    """Compare average features across emotions"""
    # Select one file per emotion
    sample_files = {}
    for emotion in EMOTION_MAPPING.values():
        if len(df[df['emotion'] == emotion]) > 0:
            sample_files[emotion] = df[df['emotion'] == emotion].iloc[0]['file_path']
    
    # Extract features
    features_by_emotion = {}
    for emotion, file_path in sample_files.items():
        features_by_emotion[emotion] = extract_features(file_path)
    
    # Compare spectrograms
    plt.figure(figsize=(15, 12))
    i = 1
    for emotion, features in features_by_emotion.items():
        plt.subplot(4, 2, i)
        librosa.display.specshow(features['mel_spectrogram'], x_axis='time', y_axis='mel')
        plt.title(f'Mel Spectrogram: {emotion.title()}')
        plt.colorbar(format='%+2.0f dB')
        i += 1
    
    plt.tight_layout()
    plt.savefig('../docs/images/emotion_spectrograms.png')
    plt.show()
    
    # Compare MFCCs
    plt.figure(figsize=(15, 12))
    i = 1
    for emotion, features in features_by_emotion.items():
        plt.subplot(4, 2, i)
        librosa.display.specshow(features['mfcc'], x_axis='time')
        plt.title(f'MFCC: {emotion.title()}')
        plt.colorbar()
        i += 1
    
    plt.tight_layout()
    plt.show()

# Example usage (commented out to prevent execution)
# compare_emotion_features(df)

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored the RAVDESS dataset, including:
# 
# 1. **Dataset Structure**: Understanding the file naming convention and organization
# 2. **Distribution Analysis**: Examining the distribution of emotions, gender, and intensity
# 3. **Audio Visualization**: Visualizing waveforms and spectrograms for different emotions
# 4. **Data Preparation**: Creating train/val/test splits by actor to prevent data leakage
# 5. **Feature Extraction**: Exploring different audio features for emotion recognition
# 
# This preprocessing ensures our dataset is well-organized and ready for model training. In the next notebook, we'll explore data augmentation techniques to enhance our training data.

# %% [markdown]
# ## Next Steps
# 
# - Implement data augmentation techniques specific to audio (pitch shifting, time stretching, etc.)
# - Develop a data loading pipeline optimized for deep learning models
# - Explore feature normalization approaches
# - Investigate model architectures suitable for emotion recognition 