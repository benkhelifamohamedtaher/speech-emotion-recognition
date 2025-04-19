# %% [markdown]
# # 🔊 Audio Data Augmentation for Emotion Recognition
# 
# This notebook explores various data augmentation techniques for audio signals, specifically tailored for speech emotion recognition. Data augmentation is crucial for improving model generalization and performance, especially for deep learning models.

# %% [markdown]
# ## Introduction
# 
# Data augmentation helps to:
# 
# 1. **Increase dataset size**: Generate more training examples from existing data
# 2. **Improve generalization**: Make the model robust to variations in input
# 3. **Reduce overfitting**: Prevent the model from memorizing training examples
# 4. **Handle class imbalance**: Generate more samples for underrepresented classes
# 
# For speech emotion recognition, we need to be careful about which augmentations to apply, as some transformations may alter emotional content (e.g., heavy pitch shifting might change perceived emotion).

# %%
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import torch
import torchaudio
import random
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# %% [markdown]
# ## Basic Audio Augmentation Techniques
# 
# Let's implement and visualize various audio augmentation techniques:

# %%
# Load a sample audio file (assuming we have the RAVDESS dataset)
def load_sample_audio(emotion='angry'):
    """Load a sample audio file for the given emotion"""
    # This is a placeholder function. In real use, you would load from your dataset
    try:
        # Try to load from processed dataset
        emotion_path = f"../processed_dataset/train/{emotion}"
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
            if files:
                file_path = os.path.join(emotion_path, files[0])
                waveform, sample_rate = librosa.load(file_path, sr=22050)
                return waveform, sample_rate, file_path
    except:
        pass
    
    # If no file found, generate dummy audio
    print("No sample file found, generating dummy audio")
    sample_rate = 22050
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return waveform, sample_rate, "dummy_audio.wav"

# Function to visualize waveform and spectrogram
def visualize_audio(waveform, sample_rate, title="Audio"):
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title(f'{title} - Waveform')
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sample_rate)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Spectrogram')
    
    plt.tight_layout()
    plt.show()
    
    # Play audio
    return ipd.Audio(waveform, rate=sample_rate)

# %% [markdown]
# ## 1. Time Domain Augmentations
# 
# These augmentations modify the signal in the time domain.

# %%
# 1.1 Time Shifting
def time_shift(waveform, sample_rate, shift_pct=0.2):
    """Shift the audio in time (left or right) by a random amount"""
    shift = int(sample_rate * shift_pct * np.random.uniform(-1, 1))
    if shift > 0:
        # Shift right: insert silence at the beginning
        shifted = np.pad(waveform, (shift, 0), mode='constant')[:len(waveform)]
    else:
        # Shift left: insert silence at the end
        shifted = np.pad(waveform, (0, -shift), mode='constant')[:-shift if shift else None]
    return shifted

# 1.2 Time Stretching
def time_stretch(waveform, rate=1.2):
    """Stretch the audio by a rate factor"""
    return librosa.effects.time_stretch(waveform, rate=rate)

# 1.3 Adding Noise
def add_noise(waveform, noise_level=0.005):
    """Add random noise to the audio signal"""
    noise = np.random.normal(0, noise_level, len(waveform))
    noisy_signal = waveform + noise
    return noisy_signal

# 1.4 Clipping
def clip_audio(waveform, clip_factor=0.8):
    """Clip the audio signal"""
    threshold = clip_factor * np.max(np.abs(waveform))
    return np.clip(waveform, -threshold, threshold)

# %% [markdown]
# ## 2. Frequency Domain Augmentations
# 
# These augmentations modify the signal in the frequency domain.

# %%
# 2.1 Pitch Shifting
def pitch_shift(waveform, sample_rate, n_steps=2):
    """Shift the pitch up or down by n_steps"""
    return librosa.effects.pitch_shift(waveform, sr=sample_rate, n_steps=n_steps)

# 2.2 Frequency Masking (for spectrograms)
def frequency_mask(spectrogram, max_mask_width=10, num_masks=1):
    """Apply frequency masking to a spectrogram"""
    spec = spectrogram.copy()
    freq_dim = spec.shape[0]
    
    for _ in range(num_masks):
        f_width = np.random.randint(1, max_mask_width)
        f_start = np.random.randint(0, freq_dim - f_width)
        spec[f_start:f_start + f_width, :] = 0
    
    return spec

# 2.3. Time Masking (for spectrograms)
def time_mask(spectrogram, max_mask_width=20, num_masks=1):
    """Apply time masking to a spectrogram"""
    spec = spectrogram.copy()
    time_dim = spec.shape[1]
    
    for _ in range(num_masks):
        t_width = np.random.randint(1, max_mask_width)
        t_start = np.random.randint(0, time_dim - t_width)
        spec[:, t_start:t_start + t_width] = 0
    
    return spec

# %% [markdown]
# ## 3. Combined Augmentations
# 
# More complex augmentations combine multiple techniques.

# %%
# 3.1 Room Reverb Simulation
def add_reverb(waveform, sample_rate, reverberance=50, damping=50, room_scale=50):
    """
    Simulate room reverb effect. This is a simplified version.
    For more accurate reverb, consider using libraries like pyroomacoustics.
    """
    # Convert parameters to values between 0 and 1
    reverberance = reverberance / 100
    damping = damping / 100
    room_scale = room_scale / 100
    
    # Simple reverb simulation using convolution with decaying impulse response
    impulse_length = int(sample_rate * room_scale * 0.5)
    impulse = np.exp(-np.arange(impulse_length) / (impulse_length * damping))
    reverb = np.convolve(waveform, impulse, mode='full')[:len(waveform)]
    
    # Mix original and reverbed signal
    mixture = (1 - reverberance) * waveform + reverberance * reverb
    return mixture / np.max(np.abs(mixture))  # Normalize

# 3.2 Filtering (Bandpass, Highpass, Lowpass)
def apply_filter(waveform, sample_rate, filter_type='bandpass', cutoff_low=500, cutoff_high=3000):
    """Apply a filter to the audio"""
    from scipy import signal
    
    nyquist = sample_rate / 2
    if filter_type == 'lowpass':
        b, a = signal.butter(4, cutoff_high / nyquist, btype='lowpass')
    elif filter_type == 'highpass':
        b, a = signal.butter(4, cutoff_low / nyquist, btype='highpass')
    else:  # bandpass
        b, a = signal.butter(4, [cutoff_low / nyquist, cutoff_high / nyquist], btype='band')
    
    return signal.filtfilt(b, a, waveform)

# %% [markdown]
# ## Demonstration of Augmentation Techniques
# 
# Let's visualize and listen to various augmentations applied to a sample audio.

# %%
# Load a sample audio file
waveform, sample_rate, _ = load_sample_audio(emotion='angry')

# Visualize and play the original audio
print("Original Audio:")
visualize_audio(waveform, sample_rate, "Original")

# Apply and visualize different augmentations

# 1. Time Shifting
print("\nTime Shifted Audio:")
shifted = time_shift(waveform, sample_rate, shift_pct=0.2)
visualize_audio(shifted, sample_rate, "Time Shifted")

# 2. Time Stretching
print("\nTime Stretched Audio (Slower):")
stretched = time_stretch(waveform, rate=0.8)  # slower
visualize_audio(stretched, sample_rate, "Time Stretched (Slower)")

print("\nTime Stretched Audio (Faster):")
stretched_fast = time_stretch(waveform, rate=1.2)  # faster
visualize_audio(stretched_fast, sample_rate, "Time Stretched (Faster)")

# 3. Adding Noise
print("\nNoisy Audio:")
noisy = add_noise(waveform, noise_level=0.01)
visualize_audio(noisy, sample_rate, "Noisy")

# 4. Pitch Shifting
print("\nPitch Shifted Audio (Higher):")
pitched_up = pitch_shift(waveform, sample_rate, n_steps=2)
visualize_audio(pitched_up, sample_rate, "Pitch Shifted (Higher)")

print("\nPitch Shifted Audio (Lower):")
pitched_down = pitch_shift(waveform, sample_rate, n_steps=-2)
visualize_audio(pitched_down, sample_rate, "Pitch Shifted (Lower)")

# 5. Reverb
print("\nReverb Audio:")
reverb = add_reverb(waveform, sample_rate, reverberance=30, damping=50, room_scale=60)
visualize_audio(reverb, sample_rate, "Reverb")

# 6. Filtered Audio
print("\nBandpass Filtered Audio:")
filtered = apply_filter(waveform, sample_rate, filter_type='bandpass', cutoff_low=500, cutoff_high=3000)
visualize_audio(filtered, sample_rate, "Bandpass Filtered")

# %% [markdown]
# ## Spectrogram Augmentations
# 
# Now let's visualize how masking augmentations affect the spectrogram representation.

# %%
# Compute Mel spectrogram for original audio
mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Apply frequency masking
masked_freq = frequency_mask(mel_spec_db, max_mask_width=20, num_masks=2)

# Apply time masking
masked_time = time_mask(mel_spec_db, max_mask_width=40, num_masks=2)

# Apply both
masked_both = time_mask(frequency_mask(mel_spec_db, max_mask_width=20, num_masks=2), 
                        max_mask_width=40, num_masks=2)

# Visualize the spectrograms
plt.figure(figsize=(12, 12))

plt.subplot(4, 1, 1)
librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Mel Spectrogram')

plt.subplot(4, 1, 2)
librosa.display.specshow(masked_freq, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Frequency Masked Mel Spectrogram')

plt.subplot(4, 1, 3)
librosa.display.specshow(masked_time, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Time Masked Mel Spectrogram')

plt.subplot(4, 1, 4)
librosa.display.specshow(masked_both, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Time and Frequency Masked Mel Spectrogram')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Implementation with PyTorch Audio Transforms
# 
# Let's implement the same augmentations using PyTorch audio transforms, which are optimized for GPU processing and integrate well with PyTorch data pipelines.

# %%
# Convert to PyTorch tensor
waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

# Initialize transforms
time_shift_transform = torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * sample_rate))
pitch_shift_transform = lambda x: torchaudio.functional.pitch_shift(x.unsqueeze(0), sample_rate, 2).squeeze(0)
time_stretch_transform = lambda x: torchaudio.functional.speed(x.unsqueeze(0), 0.8).squeeze(0)

# Apply time masking directly to Mel spectrogram
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512
)
time_mask_transform = torchaudio.transforms.TimeMasking(time_mask_param=80)
freq_mask_transform = torchaudio.transforms.FrequencyMasking(freq_mask_param=40)

# Compute a mel spectrogram for demonstration
mel_spec_tensor = mel_spec_transform(waveform_tensor)
mel_spec_db_tensor = torchaudio.transforms.AmplitudeToDB()(mel_spec_tensor)

# Apply transformations
mel_spec_time_masked = time_mask_transform(mel_spec_db_tensor)
mel_spec_freq_masked = freq_mask_transform(mel_spec_db_tensor)
mel_spec_both_masked = freq_mask_transform(time_mask_transform(mel_spec_db_tensor))

# Visualize transformations
plt.figure(figsize=(12, 12))

plt.subplot(4, 1, 1)
plt.imshow(mel_spec_db_tensor.numpy()[0], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Original Mel Spectrogram (PyTorch)')

plt.subplot(4, 1, 2)
plt.imshow(mel_spec_time_masked.numpy()[0], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Time Masked Mel Spectrogram (PyTorch)')

plt.subplot(4, 1, 3)
plt.imshow(mel_spec_freq_masked.numpy()[0], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Frequency Masked Mel Spectrogram (PyTorch)')

plt.subplot(4, 1, 4)
plt.imshow(mel_spec_both_masked.numpy()[0], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Time and Frequency Masked Mel Spectrogram (PyTorch)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Creating an Augmentation Pipeline for Training
# 
# Let's create a class that can be used in a PyTorch data pipeline to dynamically augment audio samples during training.

# %%
class AudioAugmenter:
    """
    Apply a series of audio augmentations with configurable probabilities.
    """
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
        # Define augmentation techniques and their probabilities
        self.augmentations = {
            'time_shift': {'prob': 0.5, 'params': {'shift_pct': 0.2}},
            'add_noise': {'prob': 0.3, 'params': {'noise_level': 0.005}},
            'pitch_shift': {'prob': 0.4, 'params': {'n_steps': 2}},
            'time_stretch': {'prob': 0.4, 'params': {'rate': 1.2}},
            'reverb': {'prob': 0.3, 'params': {'reverberance': 30, 'damping': 50, 'room_scale': 50}}
        }
    
    def __call__(self, waveform):
        """Apply random augmentations to the waveform"""
        # Convert from PyTorch tensor if needed
        if isinstance(waveform, torch.Tensor):
            is_tensor = True
            waveform = waveform.numpy() if waveform.ndim == 1 else waveform[0].numpy()
        else:
            is_tensor = False
        
        # Apply augmentations randomly based on their probabilities
        for aug_name, aug_config in self.augmentations.items():
            if random.random() < aug_config['prob']:
                if aug_name == 'time_shift':
                    waveform = time_shift(waveform, self.sample_rate, **aug_config['params'])
                elif aug_name == 'add_noise':
                    waveform = add_noise(waveform, **aug_config['params'])
                elif aug_name == 'pitch_shift':
                    waveform = pitch_shift(waveform, self.sample_rate, **aug_config['params'])
                elif aug_name == 'time_stretch':
                    waveform = time_stretch(waveform, **aug_config['params'])
                elif aug_name == 'reverb':
                    waveform = add_reverb(waveform, self.sample_rate, **aug_config['params'])
        
        # Convert back to tensor if the input was a tensor
        if is_tensor:
            waveform = torch.tensor(waveform, dtype=torch.float32)
        
        return waveform

# %% [markdown]
# ## Demonstration of the Augmentation Pipeline
# 
# Let's visualize several augmented versions of the same audio file to see the variety we can create.

# %%
# Create an augmenter instance
augmenter = AudioAugmenter(sample_rate=sample_rate)

# Create multiple augmented versions of the same audio
plt.figure(figsize=(15, 10))

# Original audio
plt.subplot(4, 1, 1)
librosa.display.waveshow(waveform, sr=sample_rate)
plt.title('Original Audio')

# Generate three different augmented versions
for i in range(3):
    augmented = augmenter(waveform)
    plt.subplot(4, 1, i+2)
    librosa.display.waveshow(augmented, sr=sample_rate)
    plt.title(f'Augmented Version {i+1}')

plt.tight_layout()
plt.show()

# Let's save an example spectrogram image of augmented audio for docs
mel_spec_augmented = librosa.feature.melspectrogram(y=augmenter(waveform), sr=sample_rate, n_mels=128)
mel_spec_db_augmented = librosa.power_to_db(mel_spec_augmented, ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(mel_spec_db_augmented, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram of Augmented Audio')
plt.tight_layout()
plt.savefig('../docs/images/augmented_spectrogram.png')
plt.show()

# %% [markdown]
# ## Integration with PyTorch Dataset
# 
# Finally, let's see how to integrate our augmentation pipeline into a PyTorch Dataset for easy use during training.

# %%
class AugmentedAudioDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that applies augmentations to audio samples.
    """
    def __init__(self, root_dir, transform=None, sample_rate=22050, augment=True):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.transform = transform
        self.augment = augment
        
        # Create augmenter if augmentation is enabled
        self.augmenter = AudioAugmenter(sample_rate=sample_rate) if augment else None
        
        # Find all audio files recursively
        self.files = []
        self.labels = []
        
        # Here we assume each subfolder name is an emotion label
        for emotion_dir in self.root_dir.glob('*'):
            if emotion_dir.is_dir():
                emotion = emotion_dir.name
                for audio_file in emotion_dir.glob('*.wav'):
                    self.files.append(audio_file)
                    self.labels.append(emotion)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio file
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Apply augmentation if enabled
        if self.augment and self.augmenter:
            waveform = self.augmenter(waveform)
        
        # Convert to tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Apply additional transforms if any (e.g., mel spectrogram)
        if self.transform:
            waveform = self.transform(waveform)
        
        # Convert label to index (you would need a proper label mapping)
        label_idx = {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 
                     'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7}.get(label, 0)
        
        return waveform, label_idx

# Example usage (commented out as it requires the dataset)
"""
# Define transforms
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_mels=128, n_fft=1024, hop_length=512
)

# Create dataset
dataset = AugmentedAudioDataset(
    root_dir='../processed_dataset/train',
    transform=transform,
    sample_rate=22050,
    augment=True
)

# Check the dataset
if len(dataset) > 0:
    # Get a sample
    waveform, label = dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Label: {label}")
    
    # Create a dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get a batch
    batch_waveforms, batch_labels = next(iter(dataloader))
    print(f"Batch shape: {batch_waveforms.shape}")
    print(f"Batch labels: {batch_labels}")
"""

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we explored various audio augmentation techniques for speech emotion recognition:
# 
# 1. **Time Domain Augmentations**: Time shifting, stretching, adding noise, clipping
# 2. **Frequency Domain Augmentations**: Pitch shifting, frequency masking, time masking
# 3. **Combined Augmentations**: Room reverb simulation, filtering
# 
# We implemented these techniques using both standard audio processing libraries (librosa) and PyTorch audio transforms, and created a comprehensive augmentation pipeline that can be integrated into a PyTorch dataset.
# 
# Audio augmentation is especially important for emotion recognition because:
# 
# - It helps models become robust to variations in speaking rate, pitch, and background noise
# - It creates a more diverse training set, improving generalization
# - It mitigates the limited size of emotion datasets, which are often smaller than other speech datasets
# 
# In the next notebook, we'll explore how these augmented features are processed by our model architectures. 