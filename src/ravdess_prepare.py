#!/usr/bin/env python3
"""
RAVDESS Dataset Preparation Script
Downloads and prepares the RAVDESS dataset for emotion recognition training
"""

import os
import sys
import argparse
import shutil
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# RAVDESS dataset URLs
RAVDESS_URLS = {
    'speech': 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip',
    'song': 'https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip',
}

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_dataset(output_dir, include_song=False):
    """Download the RAVDESS dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download speech data
    speech_zip = output_dir / 'Audio_Speech_Actors_01-24.zip'
    if not speech_zip.exists():
        print("Downloading speech data...")
        download_file(RAVDESS_URLS['speech'], speech_zip)
    else:
        print("Speech data already downloaded.")
    
    # Download song data if requested
    if include_song:
        song_zip = output_dir / 'Audio_Song_Actors_01-24.zip'
        if not song_zip.exists():
            print("Downloading song data...")
            download_file(RAVDESS_URLS['song'], song_zip)
        else:
            print("Song data already downloaded.")


def extract_dataset(output_dir, include_song=False):
    """Extract the RAVDESS dataset"""
    output_dir = Path(output_dir)
    ravdess_dir = output_dir / 'RAVDESS'
    ravdess_dir.mkdir(exist_ok=True)
    
    # Extract speech data
    speech_zip = output_dir / 'Audio_Speech_Actors_01-24.zip'
    if speech_zip.exists():
        print("Extracting speech data...")
        with zipfile.ZipFile(speech_zip, 'r') as zip_ref:
            zip_ref.extractall(ravdess_dir)
    
    # Extract song data if requested
    if include_song:
        song_zip = output_dir / 'Audio_Song_Actors_01-24.zip'
        if song_zip.exists():
            print("Extracting song data...")
            with zipfile.ZipFile(song_zip, 'r') as zip_ref:
                zip_ref.extractall(ravdess_dir)


def reorganize_dataset(output_dir):
    """Reorganize the dataset into a standard structure"""
    output_dir = Path(output_dir)
    ravdess_dir = output_dir / 'RAVDESS'
    
    # Check if reorganization is needed
    if (ravdess_dir / 'Actor_01').exists():
        print("Dataset already organized.")
        return
    
    print("Reorganizing dataset...")
    # First, find all extracted directories
    extracted_dirs = [d for d in ravdess_dir.iterdir() if d.is_dir()]
    
    for extracted_dir in extracted_dirs:
        # Move all audio files to actor directories
        for audio_file in extracted_dir.glob('*.wav'):
            # Parse file name to get actor ID
            parts = audio_file.name.split('-')
            actor_id = parts[6]
            actor_dir = ravdess_dir / f'Actor_{actor_id}'
            actor_dir.mkdir(exist_ok=True)
            
            # Move file to actor directory
            shutil.move(str(audio_file), str(actor_dir / audio_file.name))
        
        # Remove the empty directory
        if not any(extracted_dir.iterdir()):
            extracted_dir.rmdir()


def analyze_dataset(dataset_dir):
    """Analyze the dataset and print statistics"""
    dataset_dir = Path(dataset_dir)
    ravdess_dir = dataset_dir / 'RAVDESS'
    
    # Check if the directory exists
    if not ravdess_dir.exists():
        print("RAVDESS directory not found.")
        return
    
    # Count actors
    actors = [d for d in ravdess_dir.iterdir() if d.is_dir() and d.name.startswith('Actor_')]
    
    # Count files by emotion
    emotion_counts = {
        '01': 0,  # neutral
        '02': 0,  # calm
        '03': 0,  # happy
        '04': 0,  # sad
        '05': 0,  # angry
        '06': 0,  # fearful
        '07': 0,  # disgust
        '08': 0   # surprised
    }
    
    # Count files by modality
    modality_counts = {
        '01': 0,  # full-AV
        '02': 0,  # video-only
        '03': 0   # audio-only
    }
    
    # Count files by vocal channel
    channel_counts = {
        '01': 0,  # speech
        '02': 0   # song
    }
    
    # Total audio files
    total_files = 0
    
    # Process all audio files
    for actor_dir in actors:
        for audio_file in actor_dir.glob('*.wav'):
            parts = audio_file.name.split('-')
            
            # Count by modality
            modality = parts[0]
            if modality in modality_counts:
                modality_counts[modality] += 1
            
            # Count by vocal channel
            channel = parts[1]
            if channel in channel_counts:
                channel_counts[channel] += 1
            
            # Count by emotion
            emotion = parts[2]
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            
            total_files += 1
    
    # Print statistics
    print("\nRAVDESS Dataset Statistics:")
    print("=" * 30)
    print(f"Total actors: {len(actors)}")
    print(f"Total audio files: {total_files}")
    
    print("\nFiles by emotion:")
    emotion_names = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    for emotion_id, count in emotion_counts.items():
        if count > 0:
            print(f"  {emotion_names[emotion_id]}: {count}")
    
    print("\nFiles by modality:")
    modality_names = {
        '01': 'full-AV',
        '02': 'video-only',
        '03': 'audio-only'
    }
    for modality_id, count in modality_counts.items():
        if count > 0:
            print(f"  {modality_names[modality_id]}: {count}")
    
    print("\nFiles by vocal channel:")
    channel_names = {
        '01': 'speech',
        '02': 'song'
    }
    for channel_id, count in channel_counts.items():
        if count > 0:
            print(f"  {channel_names[channel_id]}: {count}")
    
    print("\nDataset is ready for training!")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare the RAVDESS dataset")
    parser.add_argument('--output_dir', type=str, default='../dataset',
                        help='Output directory for the dataset')
    parser.add_argument('--include_song', action='store_true',
                        help='Include song data in addition to speech data')
    parser.add_argument('--download_only', action='store_true',
                        help='Only download the dataset without extracting')
    
    args = parser.parse_args()
    
    # Download dataset
    download_dataset(args.output_dir, args.include_song)
    
    # Extract and reorganize if requested
    if not args.download_only:
        extract_dataset(args.output_dir, args.include_song)
        reorganize_dataset(args.output_dir)
        analyze_dataset(args.output_dir)
    

if __name__ == "__main__":
    main() 