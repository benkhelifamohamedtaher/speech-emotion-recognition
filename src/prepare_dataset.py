import os
import argparse
import shutil
import pandas as pd
import random
from tqdm import tqdm


def create_dataset_structure(output_dir):
    """
    Create the dataset directory structure
    
    Args:
        output_dir: Output directory
    """
    # Create directories
    for split in ['train', 'test']:
        for emotion in ['angry', 'happy', 'sad', 'neutral']:
            os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)
    
    print(f"Created dataset structure at {output_dir}")


def split_dataset(input_dir, output_dir, test_ratio=0.2, seed=42):
    """
    Split dataset into train and test sets
    
    Args:
        input_dir: Input directory with emotion-labeled audio files
        output_dir: Output directory for split dataset
        test_ratio: Ratio of test set
        seed: Random seed
    """
    random.seed(seed)
    
    # Create dataset structure
    create_dataset_structure(output_dir)
    
    # List all emotion directories
    emotion_dirs = [d for d in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, d))]
    
    for emotion in emotion_dirs:
        # Map to target emotion category
        target_emotion = map_emotion_category(emotion)
        if target_emotion is None:
            print(f"Skipping unknown emotion category: {emotion}")
            continue
        
        # Get all audio files
        audio_files = [f for f in os.listdir(os.path.join(input_dir, emotion))
                      if f.endswith('.wav')]
        
        # Shuffle files
        random.shuffle(audio_files)
        
        # Split into train and test
        split_idx = int(len(audio_files) * (1 - test_ratio))
        train_files = audio_files[:split_idx]
        test_files = audio_files[split_idx:]
        
        # Copy files to train set
        for f in tqdm(train_files, desc=f"Copying {target_emotion} (train)"):
            src = os.path.join(input_dir, emotion, f)
            dst = os.path.join(output_dir, 'train', target_emotion, f)
            shutil.copy2(src, dst)
        
        # Copy files to test set
        for f in tqdm(test_files, desc=f"Copying {target_emotion} (test)"):
            src = os.path.join(input_dir, emotion, f)
            dst = os.path.join(output_dir, 'test', target_emotion, f)
            shutil.copy2(src, dst)
    
    print("Dataset split completed")


def map_emotion_category(source_emotion):
    """
    Map source emotion category to target emotion category
    
    Args:
        source_emotion: Source emotion category
        
    Returns:
        Mapped emotion category or None if not mappable
    """
    # Convert to lowercase
    source_emotion = source_emotion.lower()
    
    # Direct mappings
    if source_emotion in ['angry', 'anger', 'rage']:
        return 'angry'
    elif source_emotion in ['happy', 'happiness', 'joy', 'excited']:
        return 'happy'
    elif source_emotion in ['sad', 'sadness', 'depressed']:
        return 'sad'
    elif source_emotion in ['neutral', 'normal', 'calm']:
        return 'neutral'
    else:
        return None


def process_metadata_csv(csv_file, audio_dir, output_dir, test_ratio=0.2, seed=42):
    """
    Process dataset from metadata CSV file
    
    Args:
        csv_file: CSV file with metadata (filename, emotion)
        audio_dir: Directory with audio files
        output_dir: Output directory for split dataset
        test_ratio: Ratio of test set
        seed: Random seed
    """
    random.seed(seed)
    
    # Create dataset structure
    create_dataset_structure(output_dir)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Check required columns
    if 'filename' not in df.columns or 'emotion' not in df.columns:
        raise ValueError("CSV file must have 'filename' and 'emotion' columns")
    
    # Group by emotion
    emotion_groups = df.groupby('emotion')
    
    for emotion, group in emotion_groups:
        # Map to target emotion category
        target_emotion = map_emotion_category(emotion)
        if target_emotion is None:
            print(f"Skipping unknown emotion category: {emotion}")
            continue
        
        # Get all files
        files = group['filename'].tolist()
        
        # Shuffle files
        random.shuffle(files)
        
        # Split into train and test
        split_idx = int(len(files) * (1 - test_ratio))
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Copy files to train set
        for f in tqdm(train_files, desc=f"Copying {target_emotion} (train)"):
            # Check if file exists
            src = os.path.join(audio_dir, f)
            if not os.path.exists(src):
                print(f"Warning: File not found: {src}")
                continue
                
            dst = os.path.join(output_dir, 'train', target_emotion, os.path.basename(f))
            shutil.copy2(src, dst)
        
        # Copy files to test set
        for f in tqdm(test_files, desc=f"Copying {target_emotion} (test)"):
            # Check if file exists
            src = os.path.join(audio_dir, f)
            if not os.path.exists(src):
                print(f"Warning: File not found: {src}")
                continue
                
            dst = os.path.join(output_dir, 'test', target_emotion, os.path.basename(f))
            shutil.copy2(src, dst)
    
    print("Dataset preparation completed")


def main():
    parser = argparse.ArgumentParser(description='Prepare Speech Emotion Recognition Dataset')
    
    parser.add_argument('--input_dir', type=str, help='Input directory with emotion-labeled audio files')
    parser.add_argument('--csv_file', type=str, help='CSV file with metadata (filename, emotion)')
    parser.add_argument('--audio_dir', type=str, help='Directory with audio files (used with csv_file)')
    parser.add_argument('--output_dir', type=str, default='./dataset', help='Output directory for dataset')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of test set')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--create_structure', action='store_true', help='Only create dataset structure')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_dataset_structure(args.output_dir)
    elif args.csv_file:
        if not args.audio_dir:
            raise ValueError("--audio_dir is required when using --csv_file")
        process_metadata_csv(args.csv_file, args.audio_dir, args.output_dir, args.test_ratio, args.seed)
    elif args.input_dir:
        split_dataset(args.input_dir, args.output_dir, args.test_ratio, args.seed)
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 