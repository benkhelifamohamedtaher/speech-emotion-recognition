# %% [markdown]
# # 📊 Audio Feature Extraction for Emotion Recognition
# 
# This notebook explores various audio feature extraction techniques for speech emotion recognition.

# %% [markdown]
# ## Introduction
# 
# Speech audio contains rich information about emotions through various acoustic properties:
# 
# - **Pitch**: Higher for excitement/happiness, lower for sadness
# - **Energy/Intensity**: Higher for anger/happiness, lower for sadness
# - **Speaking Rate**: Faster for excitement, slower for sadness
# - **Voice Quality**: Tense for anger, breathy for fear
# 
# We'll extract these properties using various audio features.

# %%
# Import libraries
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import pandas as pd
import seaborn as sns
from pathlib import Path

# Sample settings
SAMPLE_RATE = 22050
DURATION = 3  # seconds
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# %% [markdown]
# ## Basic Audio Features
# 
# Let's explore common audio features used for emotion recognition:

# %%
# Generate a simple audio example if no dataset is available
def generate_sample_audio():
    """Generate a simple audio example with different frequencies"""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Create a signal with some harmonics to simulate speech
    waveform = 0.5 * np.sin(2 * np.pi * 220 * t)  # Fundamental (220 Hz)
    waveform += 0.3 * np.sin(2 * np.pi * 440 * t)  # 1st harmonic
    waveform += 0.2 * np.sin(2 * np.pi * 880 * t)  # 2nd harmonic
    waveform *= np.exp(-t/2)  # Add some decay
    return waveform

# Load a sample audio file or generate one
def get_audio_sample():
    # Try to load from dataset
    sample_path = Path("../processed_dataset/train/angry")
    if sample_path.exists():
        files = list(sample_path.glob("*.wav"))
        if files:
            return librosa.load(files[0], sr=SAMPLE_RATE)[0]
    
    # Generate sample if dataset not available
    print("Using generated audio sample")
    return generate_sample_audio()

# Get audio sample
audio = get_audio_sample()

# Visualize waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=SAMPLE_RATE)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1. Time Domain Features
# 
# Time domain features are extracted directly from the raw waveform.

# %%
def extract_time_domain_features(audio, sr=SAMPLE_RATE):
    """Extract time domain features from an audio signal"""
    features = {}
    
    # Root Mean Square Energy (volume/intensity)
    features['rms'] = librosa.feature.rms(y=audio)[0]
    
    # Zero Crossing Rate (frequency content indicator)
    features['zcr'] = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Envelope (amplitude envelope)
    def get_envelope(y, frame_length=1024):
        return np.array([max(y[i:i+frame_length]) for i in range(0, len(y), frame_length)])
    
    features['envelope'] = get_envelope(audio)
    
    return features

# Extract and visualize time domain features
time_features = extract_time_domain_features(audio)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time_features['rms'])
plt.title("Root Mean Square Energy (Volume/Intensity)")
plt.xlabel("Frames")
plt.ylabel("RMS")

plt.subplot(3, 1, 2)
plt.plot(time_features['zcr'])
plt.title("Zero Crossing Rate (Frequency Content)")
plt.xlabel("Frames")
plt.ylabel("ZCR")

plt.subplot(3, 1, 3)
plt.plot(time_features['envelope'])
plt.title("Amplitude Envelope")
plt.xlabel("Frames")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Frequency Domain Features
# 
# Frequency domain features represent the spectral content of the audio.

# %%
def extract_frequency_domain_features(audio, sr=SAMPLE_RATE):
    """Extract frequency domain features from an audio signal"""
    features = {}
    
    # Short-time Fourier Transform (STFT)
    stft = librosa.stft(audio)
    features['stft_magnitude'] = np.abs(stft)
    features['stft_phase'] = np.angle(stft)
    features['stft_db'] = librosa.amplitude_to_db(features['stft_magnitude'], ref=np.max)
    
    # Spectral Centroid (brightness of sound)
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    
    # Spectral Bandwidth (width of the spectrum)
    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # Spectral Rolloff (frequency below which 85% of energy is contained)
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    return features

# Extract and visualize frequency domain features
freq_features = extract_frequency_domain_features(audio)

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
librosa.display.specshow(freq_features['stft_db'], sr=SAMPLE_RATE, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram (STFT)")

plt.subplot(3, 1, 2)
times = librosa.times_like(freq_features['spectral_centroid'])
plt.plot(times, freq_features['spectral_centroid'])
plt.title("Spectral Centroid (Brightness)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.subplot(3, 1, 3)
plt.plot(times, freq_features['spectral_bandwidth'])
plt.title("Spectral Bandwidth (Spread)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Mel Spectrogram and MFCCs
# 
# Mel-based features are particularly useful for speech processing as they approximate human auditory perception.

# %%
def extract_mel_features(audio, sr=SAMPLE_RATE):
    """Extract mel-based features from an audio signal"""
    features = {}
    
    # Mel Spectrogram (power spectrogram mapped to mel scale)
    features['mel_spec'] = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    features['mel_spec_db'] = librosa.power_to_db(features['mel_spec'], ref=np.max)
    
    # MFCCs (Mel-frequency cepstral coefficients)
    features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Delta and Delta-Delta MFCCs (velocity and acceleration)
    features['mfcc_delta'] = librosa.feature.delta(features['mfcc'])
    features['mfcc_delta2'] = librosa.feature.delta(features['mfcc'], order=2)
    
    return features

# Extract and visualize mel-based features
mel_features = extract_mel_features(audio)

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
librosa.display.specshow(mel_features['mel_spec_db'], sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")

plt.subplot(3, 1, 2)
librosa.display.specshow(mel_features['mfcc'], sr=SAMPLE_RATE, x_axis='time')
plt.colorbar()
plt.title("MFCC (Mel-frequency cepstral coefficients)")

plt.subplot(3, 1, 3)
librosa.display.specshow(mel_features['mfcc_delta'], sr=SAMPLE_RATE, x_axis='time')
plt.colorbar()
plt.title("Delta MFCC (Velocity)")

plt.tight_layout()
plt.show()

# Save the mel spectrogram for documentation
plt.figure(figsize=(10, 6))
librosa.display.specshow(mel_features['mel_spec_db'], sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.savefig('../docs/images/mel_spectrogram.png')
plt.close()

# %% [markdown]
# ## 4. Voice and Pitch Features
# 
# Pitch and voice quality features are particularly important for emotion recognition.

# %%
def extract_pitch_features(audio, sr=SAMPLE_RATE):
    """Extract pitch-related features from an audio signal"""
    features = {}
    
    # Fundamental frequency (F0) estimation using PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                fmin=librosa.note_to_hz('C2'), 
                                                fmax=librosa.note_to_hz('C7'),
                                                sr=sr)
    features['f0'] = f0  # Pitch
    features['voiced_flag'] = voiced_flag  # Whether frame is voiced
    
    # Harmonic-Percussive Source Separation
    harmonic, percussive = librosa.effects.hpss(audio)
    features['harmonic'] = harmonic
    features['percussive'] = percussive
    
    return features

# Extract and visualize pitch features
pitch_features = extract_pitch_features(audio)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
times = librosa.times_like(pitch_features['f0'])
plt.plot(times, pitch_features['f0'])
plt.title("Fundamental Frequency (F0/Pitch)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.subplot(3, 1, 2)
librosa.display.waveshow(pitch_features['harmonic'], sr=SAMPLE_RATE)
plt.title("Harmonic Component")
plt.xlabel("Time (s)")

plt.subplot(3, 1, 3)
librosa.display.waveshow(pitch_features['percussive'], sr=SAMPLE_RATE)
plt.title("Percussive Component")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Feature Extraction with PyTorch Audio
# 
# Let's implement the same feature extraction using PyTorch Audio for seamless integration with deep learning pipelines.

# %%
# Convert audio to PyTorch tensor
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Define feature extractors
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=13,
    melkwargs={'n_fft': 1024, 'hop_length': 512, 'n_mels': 128}
)

# Extract features
pt_mel_spec = mel_spec_transform(audio_tensor)
pt_mfcc = mfcc_transform(audio_tensor)

# Convert to log scale
pt_mel_spec_db = torchaudio.transforms.AmplitudeToDB()(pt_mel_spec)

# Visualize PyTorch-extracted features
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.imshow(pt_mel_spec_db[0].numpy(), origin='lower', aspect='auto')
plt.colorbar()
plt.title("PyTorch Mel Spectrogram")

plt.subplot(2, 1, 2)
plt.imshow(pt_mfcc[0].numpy(), origin='lower', aspect='auto')
plt.colorbar()
plt.title("PyTorch MFCC")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Feature Extraction Pipeline for Emotion Recognition
# 
# Now, let's create a complete feature extraction pipeline for emotion recognition.

# %%
class EmotionFeatureExtractor:
    """A complete feature extraction pipeline for emotion recognition"""
    
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Initialize PyTorch transforms
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': n_mels}
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def extract_features(self, waveform):
        """Extract all features from a waveform"""
        # Ensure waveform is a torch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extract features
        mel_spec = self.mel_spec_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        mfcc = self.mfcc_transform(waveform)
        
        return {
            'waveform': waveform,
            'mel_spectrogram': mel_spec,
            'mel_spectrogram_db': mel_spec_db,
            'mfcc': mfcc
        }
    
    def __call__(self, waveform):
        """Extract features and return the primary feature"""
        features = self.extract_features(waveform)
        # Return mel spectrogram in dB scale as primary feature
        return features['mel_spectrogram_db']

# %% [markdown]
# ## Feature Comparison Across Emotions
# 
# Let's compare features across different emotions to understand how they vary.

# %%
# This code would extract features from actual emotion samples if available
"""
# Define paths for emotion samples
emotion_samples = {}
for emotion in EMOTIONS:
    path = Path(f"../processed_dataset/train/{emotion}")
    if path.exists():
        files = list(path.glob("*.wav"))
        if files:
            emotion_samples[emotion] = files[0]

# Extract features for each emotion
if emotion_samples:
    # Create feature extractor
    feature_extractor = EmotionFeatureExtractor()
    
    # Extract and plot mel spectrograms
    plt.figure(figsize=(15, 12))
    for i, (emotion, file_path) in enumerate(emotion_samples.items()):
        # Load audio
        waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Extract features
        features = feature_extractor.extract_features(waveform)
        mel_spec_db = features['mel_spectrogram_db'][0].numpy()
        
        # Plot
        plt.subplot(4, 2, i+1)
        plt.imshow(mel_spec_db, origin='lower', aspect='auto')
        plt.title(f"{emotion.capitalize()} Emotion")
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('../docs/images/emotion_mel_spectrograms.png')
    plt.show()
"""

# %% [markdown]
# ## Feature Statistics for Emotion Classification
# 
# For traditional machine learning approaches, we often extract statistical features from these audio representations.

# %%
def extract_statistical_features(features_dict):
    """Extract statistical features from audio feature representations"""
    stats = {}
    
    # Process each feature
    for feature_name, feature_data in features_dict.items():
        if feature_name == 'waveform':
            continue  # Skip raw waveform
        
        # Convert to numpy if needed
        if isinstance(feature_data, torch.Tensor):
            feature_data = feature_data.squeeze().numpy()
        
        # For 2D features (spectrograms, MFCCs)
        if feature_data.ndim == 2:
            # Global statistics
            stats[f"{feature_name}_mean"] = np.mean(feature_data)
            stats[f"{feature_name}_std"] = np.std(feature_data)
            stats[f"{feature_name}_min"] = np.min(feature_data)
            stats[f"{feature_name}_max"] = np.max(feature_data)
            stats[f"{feature_name}_range"] = np.max(feature_data) - np.min(feature_data)
            
            # Frame-wise statistics (across frequency bins)
            frame_means = np.mean(feature_data, axis=0)
            stats[f"{feature_name}_frame_mean_std"] = np.std(frame_means)
            stats[f"{feature_name}_frame_mean_range"] = np.max(frame_means) - np.min(frame_means)
            
            # Frequency-wise statistics (across time)
            freq_means = np.mean(feature_data, axis=1)
            stats[f"{feature_name}_freq_mean_std"] = np.std(freq_means)
    
    return stats

# Example: extract statistical features from our sample
feature_extractor = EmotionFeatureExtractor()
features = feature_extractor.extract_features(audio)
stats = extract_statistical_features(features)

# Display some of the statistical features
pd.Series(stats).sort_index().head(10)

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we explored various audio feature extraction techniques for speech emotion recognition:
# 
# 1. **Time Domain Features**: RMS energy, zero crossing rate, envelope
# 2. **Frequency Domain Features**: Spectrogram, spectral centroid, bandwidth, rolloff
# 3. **Mel-based Features**: Mel spectrogram, MFCCs and their deltas
# 4. **Pitch Features**: Fundamental frequency, harmonic-percussive separation
# 
# We also implemented a complete feature extraction pipeline using PyTorch Audio, which can be easily integrated into deep learning models.
# 
# For our speech emotion recognition models, we'll primarily use **Mel spectrograms** as input features, as they:
# 
# - Capture frequency content in a way that approximates human perception
# - Preserve temporal dynamics important for emotion recognition
# - Can be processed effectively by CNNs and transformer architectures
# - Balance information density with computational efficiency
# 
# In the next notebook, we'll explore the base model architecture for speech emotion recognition. 