#!/usr/bin/env python3
"""
Optimized Dataset for Speech Emotion Recognition
Designed for high performance and accuracy with the RAVDESS dataset
"""

import os
import torch
import numpy as np
import random
import torchaudio
import librosa
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Define emotion mappings
EMOTION_DICT = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

SIMPLIFIED_EMOTIONS = {
    'neutral': 0,  # neutral, calm
    'happy': 1,    # happy, surprised
    'sad': 2,      # sad, fearful
    'angry': 3     # angry, disgust
}

def extract_emotion_from_filename(filename):
    """Extract emotion label from RAVDESS filename"""
    # Format: 03-01-05-01-01-01-01.wav
    # Third element (05) is the emotion id
    basename = os.path.basename(filename)
    parts = basename.split('-')
    if len(parts) >= 3:
        emotion_id = parts[2]
        return emotion_id
    return '01'  # Default to neutral if unknown format

def map_to_simplified_emotion(emotion_id):
    """Map RAVDESS emotion ID to simplified 4-class format"""
    if emotion_id in ['01', '02']:  # neutral, calm
        return 'neutral'
    elif emotion_id in ['03', '08']:  # happy, surprised
        return 'happy'
    elif emotion_id in ['04', '06']:  # sad, fearful
        return 'sad'
    elif emotion_id in ['05', '07']:  # angry, disgust
        return 'angry'
    return 'neutral'  # Default

class OptimizedEmotionDataset(Dataset):
    """
    Optimized dataset for emotion recognition with RAVDESS
    Supports both preprocessed and on-the-fly feature extraction
    """
    def __init__(
        self,
        data_dir,
        split="train",
        emotion_type="full",
        sample_rate=16000,
        duration=3.0,
        augment=False,
        transform=None,
        cache_features=True,
        feature_type="waveform",  # Can be "waveform", "melspec", or "mfcc"
        use_preprocessed=True
    ):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing the dataset
            split: "train", "val", or "test"
            emotion_type: "full" (8 emotions) or "simplified" (4 emotions)
            sample_rate: Audio sample rate
            duration: Duration in seconds
            augment: Whether to apply data augmentation
            transform: Additional transforms to apply
            cache_features: Whether to cache features in memory
            feature_type: Type of features to extract
            use_preprocessed: Whether to use preprocessed files
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.emotion_type = emotion_type
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.augment = augment and split == "train"  # Only augment training data
        self.transform = transform
        self.cache_features = cache_features
        self.feature_type = feature_type
        self.use_preprocessed = use_preprocessed
        
        # Determine the data directory based on emotion type and split
        self.data_path = self.data_dir / emotion_type / split
        
        if not self.data_path.exists():
            raise ValueError(f"Data directory {self.data_path} does not exist")
        
        # Get all audio files
        self.file_paths = []
        for ext in ['.npy', '.wav']:
            self.file_paths.extend(list(self.data_path.glob(f"*{ext}")))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No audio files found in {self.data_path}")
        
        # Sort for reproducibility
        self.file_paths.sort()
        
        # Set up label mapping based on emotion type
        if emotion_type == "full":
            self.num_classes = 8
            self.label_map = {emotion: i for i, emotion in enumerate(EMOTION_DICT.values())}
        else:  # simplified
            self.num_classes = 4
            self.label_map = SIMPLIFIED_EMOTIONS
        
        # Initialize cache
        self.feature_cache = {} if cache_features else None
        
        # Set up feature transformations
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={
                'n_fft': 1024,
                'hop_length': 512,
                'n_mels': 128
            }
        )
        
        logger.info(f"Initialized {emotion_type} dataset for {split} split with {len(self.file_paths)} samples")
    
    def __len__(self):
        return len(self.file_paths)
    
    def _load_audio(self, file_path):
        """Load audio file, handling different formats"""
        if file_path.suffix == '.npy':
            # Load preprocessed file
            waveform = torch.from_numpy(np.load(file_path)).float()
            
            # Add channel dimension if needed
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
                
            return waveform
        else:
            # Load raw audio file
            try:
                waveform, sr = torchaudio.load(file_path)
                
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Normalize
                if waveform.abs().max() > 0:
                    waveform = waveform / waveform.abs().max()
                
                # Adjust length
                if waveform.shape[1] < self.target_length:
                    # Pad
                    padding = self.target_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                else:
                    # Trim (use middle section)
                    excess = waveform.shape[1] - self.target_length
                    start = excess // 2
                    waveform = waveform[:, start:start+self.target_length]
                
                return waveform
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                # Return silent audio
                return torch.zeros(1, self.target_length)
    
    def _extract_label(self, file_path):
        """Extract emotion label from filename"""
        if self.emotion_type == "full":
            emotion_id = extract_emotion_from_filename(str(file_path))
            emotion_name = EMOTION_DICT.get(emotion_id, "neutral")
            return self.label_map.get(emotion_name, 0)
        else:  # simplified
            emotion_id = extract_emotion_from_filename(str(file_path))
            simplified_emotion = map_to_simplified_emotion(emotion_id)
            return self.label_map.get(simplified_emotion, 0)
    
    def _apply_augmentation(self, waveform):
        """Apply data augmentation to waveform"""
        if not self.augment or random.random() > 0.5:
            return waveform
            
        # Convert to numpy for augmentation
        audio_np = waveform.numpy().squeeze()
        
        # Choose random augmentation
        aug_type = random.choice(['noise', 'pitch', 'speed', 'stretch'])
        
        if aug_type == 'noise':
            # Add noise
            noise_level = random.uniform(0.001, 0.005)
            noise = np.random.randn(len(audio_np)) * noise_level
            audio_np = audio_np + noise
        
        elif aug_type == 'pitch':
            # Pitch shift
            n_steps = random.uniform(-2, 2)
            audio_np = librosa.effects.pitch_shift(audio_np, sr=self.sample_rate, n_steps=n_steps)
        
        elif aug_type == 'speed':
            # Speed change
            speed_factor = random.uniform(0.9, 1.1)
            audio_np = librosa.effects.time_stretch(audio_np, rate=speed_factor)
            
            # Ensure correct length
            if len(audio_np) < self.target_length:
                audio_np = np.pad(audio_np, (0, self.target_length - len(audio_np)))
            else:
                audio_np = audio_np[:self.target_length]
        
        elif aug_type == 'stretch':
            # Time stretch
            stretch_factor = random.uniform(0.9, 1.1)
            audio_np = librosa.effects.time_stretch(audio_np, rate=stretch_factor)
            
            # Ensure correct length
            if len(audio_np) < self.target_length:
                audio_np = np.pad(audio_np, (0, self.target_length - len(audio_np)))
            else:
                audio_np = audio_np[:self.target_length]
        
        # Normalize
        if np.abs(audio_np).max() > 0:
            audio_np = audio_np / np.abs(audio_np).max()
        
        # Convert back to tensor
        return torch.from_numpy(audio_np).float().unsqueeze(0)
    
    def _extract_features(self, waveform):
        """Extract features based on the specified feature type"""
        if self.feature_type == "waveform":
            return waveform
        
        elif self.feature_type == "melspec":
            # Apply mel spectrogram
            melspec = self.melspec_transform(waveform)
            # Convert to log mel spectrogram
            melspec = torch.log(melspec + 1e-9)
            return melspec
        
        elif self.feature_type == "mfcc":
            # Apply MFCC
            mfcc = self.mfcc_transform(waveform)
            return mfcc
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        file_path = self.file_paths[idx]
        
        # Check cache first
        if self.cache_features is not None and file_path in self.feature_cache:
            features, label = self.feature_cache[file_path]
            return features, label
        
        # Load audio
        waveform = self._load_audio(file_path)
        
        # Apply augmentation if enabled
        if self.augment:
            waveform = self._apply_augmentation(waveform)
        
        # Extract features
        features = self._extract_features(waveform)
        
        # Extract label
        label = self._extract_label(file_path)
        
        # Apply additional transforms if specified
        if self.transform is not None:
            features = self.transform(features)
        
        # Cache features if enabled
        if self.cache_features is not None:
            self.feature_cache[file_path] = (features, label)
        
        return features, label

def create_dataloaders(
    data_dir,
    batch_size=32,
    sample_rate=16000,
    duration=3.0,
    emotion_type="full",
    augment=True,
    feature_type="waveform",
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    use_preprocessed=True
):
    """Create train, validation, and test dataloaders"""
    # Common transforms for all datasets
    common_kwargs = {
        "sample_rate": sample_rate,
        "duration": duration,
        "emotion_type": emotion_type,
        "feature_type": feature_type,
        "use_preprocessed": use_preprocessed
    }
    
    # Create datasets
    train_dataset = OptimizedEmotionDataset(
        data_dir=data_dir,
        split="train",
        augment=augment,
        cache_features=True,
        **common_kwargs
    )
    
    val_dataset = OptimizedEmotionDataset(
        data_dir=data_dir,
        split="val",
        augment=False,
        cache_features=True,
        **common_kwargs
    )
    
    test_dataset = OptimizedEmotionDataset(
        data_dir=data_dir,
        split="test",
        augment=False,
        cache_features=True,
        **common_kwargs
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset
    import matplotlib.pyplot as plt
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset
    dataset = OptimizedEmotionDataset(
        data_dir="./processed_dataset",
        split="train",
        emotion_type="simplified",
        augment=True,
        feature_type="melspec"
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Get a sample
    features, label = dataset[0]
    print(f"Feature shape: {features.shape}")
    print(f"Label: {label}")
    
    # Visualize mel spectrogram
    if dataset.feature_type == "melspec":
        plt.figure(figsize=(10, 4))
        plt.imshow(features[0].numpy(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - Emotion: {label}')
        plt.tight_layout()
        plt.savefig("sample_melspec.png")
        print(f"Saved sample mel spectrogram to sample_melspec.png") 