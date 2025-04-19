#!/usr/bin/env python3
"""
RAVDESS Raw Dataset Loader
Dataset class for loading the raw RAVDESS dataset files from the original structure
with proper parsing of filename identifiers for emotion, intensity, etc.
"""

import os
import glob
import random
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAVDESSRawDataset(Dataset):
    """
    Dataset class for the raw RAVDESS dataset that handles the file naming convention:
    
    Filename format: XX-XX-XX-XX-XX-XX-XX.wav where positions indicate:
    1. Modality (01=AV, 02=video only, 03=audio only).
    2. Vocal channel (01=speech, 02=song).
    3. Emotion (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised).
    4. Emotional intensity (01=normal, 02=strong).
    5. Statement (01="Kids are talking", 02="Dogs are sitting").
    6. Repetition (01=1st repetition, 02=2nd repetition).
    7. Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    """
    
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Maps emotion IDs to integer indices for classification
    EMOTION_ID_TO_IDX = {
        '01': 0,  # neutral
        '02': 1,  # calm
        '03': 2,  # happy
        '04': 3,  # sad
        '05': 4,  # angry
        '06': 5,  # fearful
        '07': 6,  # disgust
        '08': 7   # surprised
    }
    
    def __init__(self, root_dir, split='train', sample_rate=16000, max_duration=5.0, transforms=None, 
                 audio_only=True, speech_only=True, cache_waveforms=False, subset=None):
        """
        Initialize the RAVDESS dataset.
        
        Args:
            root_dir (str): Root directory of the RAVDESS dataset
            split (str): 'train', 'val', or 'test' split
            sample_rate (int): Target sample rate for audio
            max_duration (float): Maximum duration in seconds
            transforms (callable, optional): Optional transform to be applied on a sample
            audio_only (bool): If True, only include audio-only files
            speech_only (bool): If True, only include speech files (not song)
            cache_waveforms (bool): If True, cache waveforms in memory (faster but memory-intensive)
            subset (str, optional): Optional subset of emotions to use ('basic4', 'basic6', or None for all)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_sample_length = int(max_duration * sample_rate)
        self.transforms = transforms
        self.audio_only = audio_only
        self.speech_only = speech_only
        self.cache_waveforms = cache_waveforms
        self.subset = subset
        
        # Define emotion subsets
        self.emotion_subsets = {
            'basic4': ['01', '03', '04', '05'],  # neutral, happy, sad, angry
            'basic6': ['01', '03', '04', '05', '06', '08']  # neutral, happy, sad, angry, fearful, surprised
        }
        
        # Get all audio files matching the criteria
        logger.info(f"Loading files from {root_dir}")
        
        # Look for audio files in the expected structures
        self.audio_files = []
        
        # First try standard RAVDESS structure with audio-specific folders
        audio_folders = []
        if audio_only:
            if speech_only:
                audio_folders.append("Audio_Speech_Actors_01-24")
            else:
                audio_folders.extend(["Audio_Speech_Actors_01-24", "Audio_Song_Actors_01-24"])
        
        # Find files in the expected audio folders
        for folder in audio_folders:
            folder_path = self.root_dir / folder
            if folder_path.exists():
                # Look for files in Actor_XX subdirectories
                actor_dirs = sorted(folder_path.glob("Actor_*"))
                for actor_dir in actor_dirs:
                    if actor_dir.is_dir():
                        # Get all .wav files in the actor directory
                        actor_files = list(actor_dir.glob("*.wav"))
                        self.audio_files.extend(actor_files)
                        logger.debug(f"Found {len(actor_files)} files in {actor_dir}")
        
        # If no files found, try recursive search with modality filtering
        if not self.audio_files:
            logger.warning(f"No files found in standard audio folders. Trying recursive search.")
            # Find all wav files recursively
            for wav_file in self.root_dir.glob("**/*.wav"):
                # Parse filename to check if it meets our criteria
                parts = self._parse_filename(wav_file.name)
                if not parts:
                    continue
                
                # Filter by modality
                if audio_only and parts[0] != '03':  # 03 = audio-only
                    continue
                
                # Filter by vocal channel
                if speech_only and parts[1] != '01':  # 01 = speech
                    continue
                
                self.audio_files.append(wav_file)
        
        logger.info(f"Found total of {len(self.audio_files)} audio files")
        
        # Filter by subset if needed
        if self.subset in self.emotion_subsets:
            allowed_emotions = self.emotion_subsets[self.subset]
            self.audio_files = [f for f in self.audio_files if self._parse_filename(f.name)[2] in allowed_emotions]
            logger.info(f"After subset filtering: {len(self.audio_files)} files")
        
        # Split the dataset
        self._create_splits()
        
        # Cache for waveforms
        self.waveform_cache = {} if cache_waveforms else None
        
        logger.info(f"RAVDESS {split} dataset: {len(self.file_paths)} files")

    def _parse_filename(self, filename):
        """Parse the RAVDESS filename to extract metadata"""
        try:
            parts = filename.split('.')[0].split('-')
            if len(parts) == 7:
                return parts
        except Exception as e:
            logger.warning(f"Error parsing filename {filename}: {e}")
        return None

    def _create_splits(self):
        """Split the dataset into train, val, and test sets based on actor ID"""
        # Group files by actor
        actors = {}
        for file_path in self.audio_files:
            parts = self._parse_filename(file_path.name)
            if not parts:
                continue
                
            actor_id = parts[6]
            if actor_id not in actors:
                actors[actor_id] = []
            actors[actor_id].append(file_path)
        
        # Sort actor IDs to ensure reproducibility
        actor_ids = sorted(actors.keys())
        logger.info(f"Found {len(actor_ids)} actors: {', '.join(actor_ids)}")
        
        # Split actors into train (70%), val (15%), test (15%)
        n_actors = len(actor_ids)
        n_train = int(0.7 * n_actors)
        n_val = int(0.15 * n_actors)
        
        random.seed(42)  # For reproducibility
        random.shuffle(actor_ids)
        
        train_actors = actor_ids[:n_train]
        val_actors = actor_ids[n_train:n_train+n_val]
        test_actors = actor_ids[n_train+n_val:]
        
        logger.info(f"Train actors: {', '.join(train_actors)}")
        logger.info(f"Val actors: {', '.join(val_actors)}")
        logger.info(f"Test actors: {', '.join(test_actors)}")
        
        if self.split == 'train':
            self.file_paths = [f for actor in train_actors for f in actors[actor]]
        elif self.split == 'val':
            self.file_paths = [f for actor in val_actors for f in actors[actor]]
        else:  # test
            self.file_paths = [f for actor in test_actors for f in actors[actor]]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Parse filename for metadata
        parts = self._parse_filename(file_path.name)
        if not parts:
            # Default values if parsing fails
            modality = '03'
            vocal_channel = '01'
            emotion_id = '01'
            intensity = '01'
            statement = '01'
            repetition = '01'
            actor_id = '01'
        else:
            modality = parts[0]  # 01=AV, 02=video, 03=audio
            vocal_channel = parts[1]  # 01=speech, 02=song
            emotion_id = parts[2]  # Emotion ID
            intensity = parts[3]  # 01=normal, 02=strong
            statement = parts[4]  # 01="Kids...", 02="Dogs..."
            repetition = parts[5]  # 01=1st, 02=2nd
            actor_id = parts[6]  # 01-24
        
        try:
            # Check cache first if enabled
            if self.cache_waveforms and file_path in self.waveform_cache:
                waveform = self.waveform_cache[file_path]
            else:
                # Load audio
                waveform, orig_sample_rate = torchaudio.load(file_path)
                
                # Resample if needed
                if orig_sample_rate != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, orig_sample_rate, self.sample_rate
                    )
                
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Normalize audio
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                
                # Cache if enabled
                if self.cache_waveforms:
                    self.waveform_cache[file_path] = waveform
            
            # Pad or truncate to max_sample_length
            if waveform.shape[1] < self.max_sample_length:
                # Pad with zeros
                padding = self.max_sample_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            else:
                # Truncate
                waveform = waveform[:, :self.max_sample_length]
            
            # Apply transforms if any
            if self.transforms:
                waveform = self.transforms(waveform)
            
            # Convert emotion ID to index
            emotion_idx = self.EMOTION_ID_TO_IDX.get(emotion_id, 0)
            
            # Create metadata dictionary
            metadata = {
                'emotion_id': emotion_id,
                'emotion_name': self.EMOTIONS.get(emotion_id, 'unknown'),
                'intensity': intensity,
                'actor_id': actor_id,
                'gender': 'male' if int(actor_id) % 2 == 1 else 'female',
                'file_path': str(file_path)
            }
            
            return {
                'waveform': waveform,
                'emotion': emotion_idx,
                'metadata': metadata
            }
        
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            
            # Return dummy data
            waveform = torch.zeros((1, self.max_sample_length))
            return {
                'waveform': waveform,
                'emotion': 0,  # Default to neutral
                'metadata': {'error': str(e), 'file_path': str(file_path)}
            }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader
        """
        waveforms = torch.stack([item['waveform'] for item in batch])
        emotions = torch.tensor([item['emotion'] for item in batch], dtype=torch.long)
        metadata = [item['metadata'] for item in batch]
        
        return {
            'waveform': waveforms,
            'emotion': emotions,
            'metadata': metadata
        }
    
    @staticmethod
    def get_emotion_mapping():
        """Return mapping of emotion indices to names"""
        idx_to_emotion = {v: k for k, v in RAVDESSRawDataset.EMOTION_ID_TO_IDX.items()}
        idx_to_name = {idx: RAVDESSRawDataset.EMOTIONS[emotion_id] for idx, emotion_id in idx_to_emotion.items()}
        return idx_to_name

    @staticmethod
    def get_num_classes(subset=None):
        """Return the number of emotion classes based on subset"""
        if subset == 'basic4':
            return 4
        elif subset == 'basic6':
            return 6
        else:
            return 8  # All emotions

if __name__ == "__main__":
    # Test the dataset
    dataset = RAVDESSRawDataset(
        root_dir="dataset_raw/RAVDESS",
        split="train",
        sample_rate=16000,
        max_duration=5.0,
        audio_only=True,
        speech_only=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Print a sample
        sample = dataset[0]
        print(f"Sample waveform shape: {sample['waveform'].shape}")
        print(f"Sample emotion: {sample['emotion']} ({dataset.EMOTIONS.get(list(dataset.EMOTION_ID_TO_IDX.keys())[list(dataset.EMOTION_ID_TO_IDX.values()).index(sample['emotion'])], 'unknown')})")
        print(f"Sample metadata: {sample['metadata']}")
        
        # Calculate class distribution
        emotions = [dataset[i]['emotion'] for i in range(len(dataset))]
        unique, counts = np.unique(emotions, return_counts=True)
        print("\nClass distribution:")
        for e, c in zip(unique, counts):
            emotion_name = dataset.EMOTIONS.get(
                list(dataset.EMOTION_ID_TO_IDX.keys())[list(dataset.EMOTION_ID_TO_IDX.values()).index(e)], 
                'unknown'
            )
            print(f"  {emotion_name}: {c} samples") 