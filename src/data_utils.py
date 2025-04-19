import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa


class AudioAugmentation:
    """Audio augmentation techniques for data augmentation"""
    
    @staticmethod
    def add_noise(waveform, noise_level=0.005):
        """Add random noise to waveform"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    @staticmethod
    def time_shift(waveform, shift_limit=0.1):
        """Shift waveform in time"""
        shift = int(random.random() * shift_limit * waveform.shape[-1])
        return torch.roll(waveform, shift, dims=-1)
    
    @staticmethod
    def time_stretch(waveform, sample_rate, rate_min=0.9, rate_max=1.1):
        """Time stretch waveform"""
        rate = random.uniform(rate_min, rate_max)
        waveform_np = waveform.numpy().squeeze()
        waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
        return torch.from_numpy(waveform_stretched).unsqueeze(0)
    
    @staticmethod
    def pitch_shift(waveform, sample_rate, shift_min=-2, shift_max=2):
        """Pitch shift waveform"""
        shift = random.uniform(shift_min, shift_max)
        waveform_np = waveform.numpy().squeeze()
        waveform_shifted = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=shift)
        return torch.from_numpy(waveform_shifted).unsqueeze(0)


class EmotionSpeechDataset(Dataset):
    """Dataset for speech emotion recognition"""
    
    def __init__(self, root_dir, split='train', target_sr=16000, transform=None, max_length=None):
        """
        Args:
            root_dir (str): Dataset root directory
            split (str): 'train' or 'test'
            target_sr (int): Target sample rate
            transform (callable, optional): Optional transform to be applied on waveforms
            max_length (int, optional): Max length of waveform in samples
        """
        self.root_dir = os.path.join(root_dir, split)
        self.target_sr = target_sr
        self.transform = transform
        self.max_length = max_length
        
        self.emotion_labels = ["angry", "happy", "sad", "neutral"]
        self.emotion_to_id = {label: i for i, label in enumerate(self.emotion_labels)}
        
        self.samples = []
        
        # Load all audio files and their labels
        for emotion in self.emotion_labels:
            emotion_dir = os.path.join(self.root_dir, emotion)
            if os.path.exists(emotion_dir):
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        self.samples.append({
                            'path': os.path.join(emotion_dir, file_name),
                            'emotion': emotion,
                            'label': self.emotion_to_id[emotion]
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(sample['path'])
        
        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure exact length of waveform by padding or trimming
        if self.max_length is not None:
            if waveform.shape[1] > self.max_length:
                # Trim from the center
                excess = waveform.shape[1] - self.max_length
                start = excess // 2
                waveform = waveform[:, start:start+self.max_length]
            elif waveform.shape[1] < self.max_length:
                # Pad with zeros to exact length
                padding = self.max_length - waveform.shape[1]
                pad_left = padding // 2
                pad_right = padding - pad_left
                waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
                
            # Double-check the length is exactly as expected
            assert waveform.shape[1] == self.max_length, f"Waveform length {waveform.shape[1]} doesn't match required length {self.max_length}"
        
        # Apply transforms if any
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        # Normalize waveform
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        
        # Ensure waveform has correct shape [1, sequence_length] without extra dimensions
        if waveform.dim() > 2:
            waveform = waveform.squeeze(0)  # Remove any extra dimensions
        
        # Ensure shape is [1, sequence_length]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Final shape check
        assert waveform.dim() == 2 and waveform.shape[0] == 1, f"Waveform has incorrect shape: {waveform.shape}, expected [1, {self.max_length}]"
        
        return {
            'waveform': waveform,
            'label': sample['label'],
            'emotion': sample['emotion']
        }


# Define a pickle-friendly transform class
class AudioTransform:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        
    def __call__(self, waveform):
        augmentations = [
            AudioAugmentation.add_noise,
            lambda x: AudioAugmentation.time_shift(x),
            lambda x: AudioAugmentation.time_stretch(x, self.target_sr),
            lambda x: AudioAugmentation.pitch_shift(x, self.target_sr)
        ]
        
        # Apply random augmentation with 50% probability
        if random.random() < 0.5:
            aug_fn = random.choice(augmentations)
            waveform = aug_fn(waveform)
        
        return waveform


def create_dataloaders(dataset_root, batch_size=16, target_sr=16000, max_length=48000, 
                      apply_augmentation=True, num_workers=4):
    """
    Create train and test dataloaders
    
    Args:
        dataset_root (str): Dataset root directory
        batch_size (int): Batch size
        target_sr (int): Target sample rate
        max_length (int): Max audio length in samples (3 seconds at 16kHz)
        apply_augmentation (bool): Whether to apply augmentation to training data
        num_workers (int): Number of worker threads for dataloading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms for training with augmentation
    train_transform = None
    if apply_augmentation:
        train_transform = AudioTransform(target_sr=target_sr)
    
    # Create datasets
    train_dataset = EmotionSpeechDataset(
        root_dir=dataset_root,
        split='train',
        target_sr=target_sr,
        transform=train_transform,
        max_length=max_length
    )
    
    test_dataset = EmotionSpeechDataset(
        root_dir=dataset_root,
        split='test',
        target_sr=target_sr,
        transform=None,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader 