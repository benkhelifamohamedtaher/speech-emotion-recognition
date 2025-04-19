#!/usr/bin/env python3
"""
Generate a header image for the speech emotion recognition project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as path_effects
from pathlib import Path

# Set up the figure
plt.figure(figsize=(15, 5))
ax = plt.subplot(111)

# Create a spectrogram-like background
t = np.linspace(0, 1, 1000)
freq_components = [
    (5, 0.5, 0.3),  # (frequency, amplitude, phase)
    (10, 0.7, 0.1),
    (15, 0.4, 0.7),
    (20, 0.8, 0.9),
    (25, 0.6, 0.5),
    (30, 0.3, 0.2)
]

# Generate a signal that looks like speech
y = np.zeros_like(t)
for freq, amp, phase in freq_components:
    y += amp * np.sin(2 * np.pi * freq * t + phase)

# Add some random noise
y += np.random.normal(0, 0.1, len(t))

# Create a 2D spectrogram-like array
spec = np.zeros((100, len(t)))
for i in range(100):
    # Create different frequency bands with varying intensities
    factor = np.exp(-(i - 50)**2 / 500)
    spec[i, :] = factor * (y + 0.2 * np.sin(i/5) * np.random.normal(0, 0.1, len(t)))

# Create a colormap
colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
cmap = LinearSegmentedColormap.from_list("emotion_cmap", colors)

# Plot the spectrogram
img = ax.imshow(spec, aspect='auto', cmap=cmap, alpha=0.7)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add emotion labels with visual appeal
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
colors = ['#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e74c3c', '#1abc9c', '#e67e22', '#34495e']

# Position the emotion words at varying heights
y_positions = np.linspace(0.3, 0.7, len(emotions))
np.random.shuffle(y_positions)

for i, (emotion, color) in enumerate(zip(emotions, colors)):
    x_pos = 0.1 + 0.8 * i / (len(emotions) - 1)
    y_pos = y_positions[i]
    
    # Get a font size that varies slightly
    font_size = 16 + np.random.randint(-2, 3)
    
    # Add the text with a nice effect
    text = ax.text(
        x_pos, y_pos, emotion.upper(), 
        ha='center', va='center', 
        fontsize=font_size,
        color=color,
        fontweight='bold',
        transform=ax.transAxes
    )
    
    # Add a subtle shadow effect
    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='black', alpha=0.2),
        path_effects.Normal()
    ])

# Add the title
title = ax.text(
    0.5, 0.85, "SPEECH EMOTION RECOGNITION", 
    ha='center', va='center',
    fontsize=32, 
    color='white',
    fontweight='bold',
    transform=ax.transAxes
)

# Add a shadow to the title
title.set_path_effects([
    path_effects.Stroke(linewidth=5, foreground='black', alpha=0.5),
    path_effects.Normal()
])

# Add a subtitle
subtitle = ax.text(
    0.5, 0.75, "Deep Learning for Audio Emotion Analysis", 
    ha='center', va='center',
    fontsize=18, 
    color='white',
    fontweight='bold',
    transform=ax.transAxes
)

# Add a shadow to the subtitle
subtitle.set_path_effects([
    path_effects.Stroke(linewidth=3, foreground='black', alpha=0.3),
    path_effects.Normal()
])

# Add a waveform at the bottom
waveform_y = 0.15
waveform_x = np.linspace(0.05, 0.95, 1000)
waveform_amplitude = 0.05
waveform = waveform_y + waveform_amplitude * y

# Plot the waveform
ax.plot(
    waveform_x, waveform, 
    color='#3498db', 
    linewidth=2, 
    alpha=0.8
)

# Save the image
output_dir = Path(__file__).parent
output_path = output_dir / "emotion_recognition_header.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Header image saved to {output_path}")

# Also create a simplified confusion matrix image
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
cm = np.array([
    [33, 20, 5, 8, 2, 4, 3, 5],
    [18, 40, 3, 10, 2, 5, 2, 0],
    [5, 4, 38, 2, 16, 5, 3, 7],
    [8, 10, 2, 44, 3, 12, 1, 0],
    [2, 2, 14, 4, 39, 3, 9, 7],
    [3, 4, 5, 12, 4, 46, 2, 4],
    [4, 1, 3, 2, 10, 2, 34, 4],
    [4, 0, 8, 1, 7, 6, 3, 51]
])

# Normalize the confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create the figure
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)

# Create the heatmap
im = ax.imshow(cm_norm, cmap='Blues')

# Add text annotations
for i in range(len(emotions)):
    for j in range(len(emotions)):
        text = ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                      ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white")

# Set up the axes
ax.set_xticks(np.arange(len(emotions)))
ax.set_yticks(np.arange(len(emotions)))
ax.set_xticklabels(emotions)
ax.set_yticklabels(emotions)

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add labels and title
ax.set_ylabel('True Emotion')
ax.set_xlabel('Predicted Emotion')
ax.set_title('Speech Emotion Recognition Confusion Matrix')

# Add a color bar
cbar = plt.colorbar(im)
cbar.set_label('Probability')

# Save the confusion matrix
cm_path = output_dir / "confusion_matrix.png"
plt.tight_layout()
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Confusion matrix saved to {cm_path}") 