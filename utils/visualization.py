#!/usr/bin/env python3
"""
Visualization utilities for speech emotion recognition results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(confusion_matrix, class_names, title='Confusion Matrix', 
                          normalize=True, save_path=None, figsize=(10, 8)):
    """
    Generate a confusion matrix visualization.
    
    Args:
        confusion_matrix: The confusion matrix as numpy array
        class_names: List of class names for axis labels
        title: Plot title
        normalize: Whether to normalize values (default: True)
        save_path: Path to save the figure (optional)
        figsize: Figure size as tuple (width, height)
    """
    # Create figure and axis
    plt.figure(figsize=figsize)
    
    # Normalize if required
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create heatmap
    sns.set(font_scale=1.2)
    heatmap = sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, 
        vmax=1 if normalize else None,
        square=True,
        cbar=True
    )
    
    # Set labels and title
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.title(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt


def plot_emotion_probabilities(probabilities, emotions, title='Emotion Probabilities', 
                               figsize=(10, 6), save_path=None, highlight_max=True):
    """
    Plot emotion probabilities as a horizontal bar chart.
    
    Args:
        probabilities: Array-like of probability values
        emotions: List of emotion names
        title: Plot title
        figsize: Figure size as tuple (width, height)
        save_path: Path to save the figure (optional)
        highlight_max: Whether to highlight the max probability
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to array if not already
    probs = np.array(probabilities)
    
    # Set colors
    base_color = '#3498db'  # Blue
    highlight_color = '#e74c3c'  # Red
    
    # Get colors for each bar
    colors = [base_color] * len(probs)
    if highlight_max:
        max_idx = np.argmax(probs)
        colors[max_idx] = highlight_color
    
    # Create horizontal bar chart
    y_pos = np.arange(len(emotions))
    ax.barh(y_pos, probs, align='center', color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title(title)
    
    # Add value labels
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    # Add grid lines
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability chart saved to {save_path}")
    
    return plt


def create_emotion_animation(emotion_history, emotions, update_interval=100):
    """
    Create an animation for real-time emotion probability display
    
    Args:
        emotion_history: List to store historical probabilities
        emotions: List of emotion names
        update_interval: Refresh interval in ms
    
    Returns:
        Figure and animation objects
    """
    import matplotlib.animation as animation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update_plot(frame):
        ax.clear()
        if not emotion_history:
            # No data yet
            return []
        
        # Get the latest probabilities
        latest_probs = emotion_history[-1]
        
        # Set colors
        base_color = '#3498db'  # Blue
        highlight_color = '#e74c3c'  # Red
        
        # Get colors for each bar
        colors = [base_color] * len(latest_probs)
        max_idx = np.argmax(latest_probs)
        colors[max_idx] = highlight_color
        
        # Create horizontal bar chart
        y_pos = np.arange(len(emotions))
        bars = ax.barh(y_pos, latest_probs, align='center', color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(emotions)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Probability')
        ax.set_title('Real-time Emotion Recognition')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(latest_probs):
            ax.text(max(v + 0.01, 0.05), i, f'{v:.2f}', va='center')
        
        # Add grid lines
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        return bars
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update_plot, interval=update_interval, blit=False
    )
    
    plt.tight_layout()
    return fig, ani


def save_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Save training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Example usage
    # Generate sample confusion matrix
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    cm = np.array([
        [10, 2, 0, 1, 0, 0, 0, 0],
        [1, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 7, 1, 1, 0, 0, 1],
        [1, 0, 1, 8, 0, 1, 0, 0],
        [0, 0, 1, 0, 9, 0, 1, 0],
        [0, 0, 0, 1, 0, 7, 0, 1],
        [0, 0, 0, 0, 1, 0, 6, 0],
        [0, 0, 1, 0, 0, 2, 0, 7]
    ])
    
    # Create output directory
    Path('docs/images').mkdir(exist_ok=True, parents=True)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        cm, emotions, 
        title='Speech Emotion Recognition Confusion Matrix',
        save_path='docs/images/confusion_matrix.png'
    )
    
    # Plot and save probabilities example
    probs = [0.7, 0.05, 0.1, 0.02, 0.05, 0.03, 0.01, 0.04]
    plot_emotion_probabilities(
        probs, emotions,
        title='Example Emotion Probabilities',
        save_path='docs/images/emotion_probabilities.png'
    ) 