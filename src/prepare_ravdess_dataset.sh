#!/bin/bash
# Prepare RAVDESS dataset for training

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Preparing RAVDESS Dataset for Training =====${NC}"
echo

# Use the correct Python interpreter
PYTHON_CMD="python3"

# Default parameters
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../dataset/RAVDESS_prepared"
SAMPLE_RATE=16000
TRAIN_SPLIT=0.8
VAL_SPLIT=0.1
TEST_SPLIT=0.1

# Create the output directory
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/train
mkdir -p $OUTPUT_DIR/val
mkdir -p $OUTPUT_DIR/test

echo -e "${GREEN}Creating train, validation, and test splits for RAVDESS dataset...${NC}"

# Run the Python script to prepare the dataset
$PYTHON_CMD - <<EOF
import os
import sys
import random
import shutil
from pathlib import Path
import numpy as np

# Set parameters
dataset_root = "$DATASET_ROOT"
output_dir = "$OUTPUT_DIR"
train_split = $TRAIN_SPLIT
val_split = $VAL_SPLIT
test_split = $TEST_SPLIT

# Create output directories
Path(output_dir).mkdir(exist_ok=True, parents=True)
Path(os.path.join(output_dir, "train")).mkdir(exist_ok=True)
Path(os.path.join(output_dir, "val")).mkdir(exist_ok=True)
Path(os.path.join(output_dir, "test")).mkdir(exist_ok=True)

# Get all actor directories
actor_dirs = [d for d in os.listdir(dataset_root) if d.startswith("Actor_")]
actor_dirs.sort()

print(f"Found {len(actor_dirs)} actors in the dataset.")

# Initialize counters
total_files = 0
train_files = 0
val_files = 0
test_files = 0

# Function to extract emotion from filename
def get_emotion(filename):
    parts = filename.split('-')
    emotion_code = int(parts[2])
    # RAVDESS emotion mapping:
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 
    # 06 = fearful, 07 = disgust, 08 = surprised
    emotion_names = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotion_names[emotion_code]

# Process each actor
for actor_dir in actor_dirs:
    actor_path = os.path.join(dataset_root, actor_dir)
    audio_files = [f for f in os.listdir(actor_path) if f.endswith(".wav")]
    
    # Skip empty directories
    if not audio_files:
        continue
    
    total_files += len(audio_files)
    
    # Group by emotion to ensure balanced splits
    emotion_groups = {}
    for file in audio_files:
        emotion = get_emotion(file)
        if emotion not in emotion_groups:
            emotion_groups[emotion] = []
        emotion_groups[emotion].append(file)
    
    # Distribute files maintaining emotion balance
    for emotion, files in emotion_groups.items():
        random.shuffle(files)
        n_files = len(files)
        
        train_end = int(n_files * train_split)
        val_end = train_end + int(n_files * val_split)
        
        # Ensure at least one file in each split if possible
        if n_files >= 3:
            if train_end == 0:
                train_end = 1
            if val_end == train_end:
                val_end = train_end + 1
        
        # Split files
        train_files_emotion = files[:train_end]
        val_files_emotion = files[train_end:val_end]
        test_files_emotion = files[val_end:]
        
        # Copy files to respective directories
        for file in train_files_emotion:
            src = os.path.join(actor_path, file)
            dst = os.path.join(output_dir, "train", f"{actor_dir}_{file}")
            shutil.copy2(src, dst)
            train_files += 1
        
        for file in val_files_emotion:
            src = os.path.join(actor_path, file)
            dst = os.path.join(output_dir, "val", f"{actor_dir}_{file}")
            shutil.copy2(src, dst)
            val_files += 1
        
        for file in test_files_emotion:
            src = os.path.join(actor_path, file)
            dst = os.path.join(output_dir, "test", f"{actor_dir}_{file}")
            shutil.copy2(src, dst)
            test_files += 1

print(f"Dataset preparation complete!")
print(f"Total files: {total_files}")
print(f"Training files: {train_files} ({train_files/total_files*100:.1f}%)")
print(f"Validation files: {val_files} ({val_files/total_files*100:.1f}%)")
print(f"Testing files: {test_files} ({test_files/total_files*100:.1f}%)")

# Create metadata files
with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
    f.write(f"Total files: {total_files}\n")
    f.write(f"Training files: {train_files} ({train_files/total_files*100:.1f}%)\n")
    f.write(f"Validation files: {val_files} ({val_files/total_files*100:.1f}%)\n")
    f.write(f"Testing files: {test_files} ({test_files/total_files*100:.1f}%)\n")
EOF

# Check if the script was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Dataset preparation complete!${NC}"
    echo -e "${GREEN}The prepared dataset is available at ${OUTPUT_DIR}${NC}"
    echo -e "${YELLOW}To use this dataset for training, run:${NC}"
    echo -e "${BLUE}./run_ravdess_high_accuracy.sh --dataset_root ${OUTPUT_DIR}${NC}"
else
    echo -e "${RED}Dataset preparation failed!${NC}"
    exit 1
fi 