#!/usr/bin/env python3
"""
Script to process the RAVDESS dataset for speech emotion recognition training
"""

import os
import argparse
import shutil
import random
from tqdm import tqdm
import pandas as pd


def map_ravdess_emotion(emotion_code):
    """Map RAVDESS emotion code to project emotion categories"""
    # RAVDESS emotion codes:
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 
    # 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    
    # Our project categories: angry, happy, sad, neutral
    emotion_map = {
        '01': 'neutral',  # neutral -> neutral
        '02': 'neutral',  # calm -> neutral
        '03': 'happy',    # happy -> happy 
        '04': 'sad',      # sad -> sad
        '05': 'angry',    # angry -> angry
        '06': None,       # fearful -> not used
        '07': None,       # disgust -> not used
        '08': None        # surprised -> not used
    }
    
    return emotion_map.get(emotion_code)


def parse_ravdess_filename(filename):
    """Parse RAVDESS filename to extract metadata"""
    # Filename format: 03-01-05-01-01-01-01.wav
    # Format: Modality-Vocal channel-Emotion-Intensity-Statement-Repetition-Actor.wav
    
    parts = os.path.splitext(filename)[0].split('-')
    
    if len(parts) != 7:
        return None
        
    metadata = {
        'modality': parts[0],       # 01=AV, 02=video only, 03=audio only
        'vocal_channel': parts[1],  # 01=speech, 02=song
        'emotion_code': parts[2],   # 01=neutral, 02=calm, etc.
        'intensity': parts[3],      # 01=normal, 02=strong
        'statement': parts[4],      # 01="Kids...", 02="Dogs..."
        'repetition': parts[5],     # 01=1st repetition, 02=2nd repetition
        'actor_id': parts[6],       # 01 to 24 (odd=male, even=female)
        'gender': 'male' if int(parts[6]) % 2 == 1 else 'female'
    }
    
    # Map the emotion code to our categories
    metadata['emotion'] = map_ravdess_emotion(metadata['emotion_code'])
    
    return metadata


def process_ravdess(input_dir, output_dir, test_ratio=0.2, speech_only=True, seed=42):
    """Process RAVDESS dataset for training"""
    random.seed(seed)
    
    # Create output directory structure
    for split in ['train', 'test']:
        for emotion in ['angry', 'happy', 'sad', 'neutral']:
            os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)
    
    # Collect all audio files
    all_files = []
    actor_dirs = [d for d in os.listdir(input_dir) if d.startswith('Actor_')]
    
    # Process each actor directory
    for actor_dir in actor_dirs:
        actor_path = os.path.join(input_dir, actor_dir)
        if os.path.isdir(actor_path):
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(actor_path, filename)
                    metadata = parse_ravdess_filename(filename)
                    
                    if metadata and metadata['emotion'] is not None:
                        # Skip song files if speech_only is True
                        if speech_only and metadata['vocal_channel'] != '01':
                            continue
                            
                        all_files.append({
                            'path': file_path,
                            'metadata': metadata,
                            'emotion': metadata['emotion']
                        })
    
    # Group files by emotion
    emotion_groups = {}
    for file_info in all_files:
        emotion = file_info['emotion']
        if emotion not in emotion_groups:
            emotion_groups[emotion] = []
        emotion_groups[emotion].append(file_info)
    
    # Create a DataFrame for analysis
    df_rows = []
    for file_info in all_files:
        metadata = file_info['metadata']
        df_rows.append({
            'filename': os.path.basename(file_info['path']),
            'emotion': metadata['emotion'],
            'emotion_code': metadata['emotion_code'],
            'gender': metadata['gender'],
            'actor_id': metadata['actor_id'],
            'intensity': metadata['intensity'],
            'modality': metadata['modality'],
            'vocal_channel': metadata['vocal_channel']
        })
    
    df = pd.DataFrame(df_rows)
    
    # Save dataset statistics
    os.makedirs(os.path.join(output_dir, 'stats'), exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'stats', 'dataset_stats.csv'), index=False)
    
    # Generate summary statistics
    summary = {
        'total_files': len(df),
        'emotion_counts': df['emotion'].value_counts().to_dict(),
        'gender_counts': df['gender'].value_counts().to_dict(),
        'intensity_counts': df['intensity'].value_counts().to_dict()
    }
    
    with open(os.path.join(output_dir, 'stats', 'summary.txt'), 'w') as f:
        f.write("RAVDESS Dataset Summary\n")
        f.write("======================\n\n")
        f.write(f"Total files: {summary['total_files']}\n\n")
        
        f.write("Emotion counts:\n")
        for emotion, count in summary['emotion_counts'].items():
            f.write(f"  {emotion}: {count}\n")
        f.write("\n")
        
        f.write("Gender counts:\n")
        for gender, count in summary['gender_counts'].items():
            f.write(f"  {gender}: {count}\n")
        f.write("\n")
        
        f.write("Intensity counts:\n")
        for intensity, count in summary['intensity_counts'].items():
            intensity_label = "normal" if intensity == "01" else "strong"
            f.write(f"  {intensity_label}: {count}\n")
    
    # Split and copy files
    for emotion, files in emotion_groups.items():
        random.shuffle(files)
        split_idx = int(len(files) * (1 - test_ratio))
        
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Copy training files
        for file_info in tqdm(train_files, desc=f"Copying {emotion} (train)"):
            src = file_info['path']
            dst = os.path.join(output_dir, 'train', emotion, os.path.basename(src))
            shutil.copy2(src, dst)
        
        # Copy testing files
        for file_info in tqdm(test_files, desc=f"Copying {emotion} (test)"):
            src = file_info['path']
            dst = os.path.join(output_dir, 'test', emotion, os.path.basename(src))
            shutil.copy2(src, dst)
    
    print(f"RAVDESS dataset processed and saved to {output_dir}")
    print(f"Statistics saved to {os.path.join(output_dir, 'stats')}")


def main():
    parser = argparse.ArgumentParser(description='Process RAVDESS dataset for speech emotion recognition')
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory with RAVDESS audio files (Audio_Speech_Actors_01-24)')
    parser.add_argument('--output_dir', type=str, default='./processed_dataset',
                        help='Output directory for processed dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test set (default: 0.2)')
    parser.add_argument('--speech_only', action='store_true',
                        help='Use only speech files (exclude song files)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    process_ravdess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        speech_only=args.speech_only,
        seed=args.seed
    )


if __name__ == '__main__':
    main() 