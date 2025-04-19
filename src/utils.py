import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torchaudio


def load_audio(file_path, target_sr=16000):
    """
    Load audio file with librosa
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    waveform, sr = librosa.load(file_path, sr=target_sr)
    return waveform, sr


def load_audio_torchaudio(file_path, target_sr=16000):
    """
    Load audio file with torchaudio
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    waveform, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    """
    Plot waveform
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        title: Plot title
        ax: Matplotlib axis
        
    Returns:
        matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    
    time = np.arange(0, len(waveform)) / sr
    ax.plot(time, waveform)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim([0, len(waveform) / sr])
    
    return ax


def plot_spectrogram(waveform, sr, n_fft=1024, hop_length=512, title="Spectrogram", ax=None):
    """
    Plot spectrogram
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        title: Plot title
        ax: Matplotlib axis
        
    Returns:
        matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    # Plot spectrogram
    img = librosa.display.specshow(
        D, 
        sr=sr, 
        x_axis='time', 
        y_axis='log', 
        ax=ax,
        hop_length=hop_length
    )
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    
    return ax


def plot_mel_spectrogram(waveform, sr, n_fft=1024, hop_length=512, n_mels=80, 
                         title="Mel Spectrogram", ax=None):
    """
    Plot mel spectrogram
    
    Args:
        waveform: Audio waveform
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bands
        title: Plot title
        ax: Matplotlib axis
        
    Returns:
        matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot mel spectrogram
    img = librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel', 
        ax=ax,
        hop_length=hop_length
    )
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    
    return ax


def visualize_audio(file_path, target_sr=16000):
    """
    Visualize audio file with waveform, spectrogram, and mel spectrogram
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
    """
    # Load audio
    waveform, sr = load_audio(file_path, target_sr)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot waveform
    plot_waveform(waveform, sr, title="Waveform", ax=axes[0])
    
    # Plot spectrogram
    plot_spectrogram(waveform, sr, title="Spectrogram", ax=axes[1])
    
    # Plot mel spectrogram
    plot_mel_spectrogram(waveform, sr, title="Mel Spectrogram", ax=axes[2])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def analyze_dataset(dataset_root):
    """
    Analyze dataset and print statistics
    
    Args:
        dataset_root: Dataset root directory
    """
    print(f"Analyzing dataset: {dataset_root}")
    
    # Get emotions
    emotions = ["angry", "happy", "sad", "neutral"]
    
    # Stats to collect
    total_files = 0
    emotion_counts = {emotion: 0 for emotion in emotions}
    durations = []
    
    # Process train and test sets
    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            continue
        
        print(f"\n{split.upper()} Set:")
        split_files = 0
        split_emotion_counts = {emotion: 0 for emotion in emotions}
        
        for emotion in emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Emotion directory not found: {emotion_dir}")
                continue
            
            # Count files
            files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            num_files = len(files)
            split_files += num_files
            split_emotion_counts[emotion] = num_files
            emotion_counts[emotion] += num_files
            
            # Sample some files for duration analysis
            sample_size = min(num_files, 5)
            for i in range(sample_size):
                file_path = os.path.join(emotion_dir, files[i])
                waveform, sr = load_audio(file_path)
                duration = len(waveform) / sr
                durations.append(duration)
        
        total_files += split_files
        
        # Print split statistics
        print(f"Total files: {split_files}")
        for emotion, count in split_emotion_counts.items():
            percentage = (count / split_files) * 100 if split_files > 0 else 0
            print(f"  {emotion}: {count} files ({percentage:.1f}%)")
    
    # Print overall statistics
    print("\nOVERALL STATISTICS:")
    print(f"Total files: {total_files}")
    for emotion, count in emotion_counts.items():
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"  {emotion}: {count} files ({percentage:.1f}%)")
    
    if durations:
        print(f"\nAudio Durations:")
        print(f"  Mean: {np.mean(durations):.2f}s")
        print(f"  Min: {np.min(durations):.2f}s")
        print(f"  Max: {np.max(durations):.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Visualization and Analysis")
    parser.add_argument('--file', type=str, help='Path to audio file for visualization')
    parser.add_argument('--dataset', type=str, help='Path to dataset root for analysis')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate')
    
    args = parser.parse_args()
    
    if args.file:
        visualize_audio(args.file, args.sample_rate)
    
    if args.dataset:
        analyze_dataset(args.dataset) 