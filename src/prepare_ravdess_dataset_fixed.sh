#!/bin/bash
# Prepare RAVDESS dataset for training in the proper structure

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Preparing RAVDESS Dataset for Training (Fixed Structure) =====${NC}"
echo

# Use the correct Python interpreter
PYTHON_CMD="python3"

# Default parameters
DATASET_ROOT="../dataset/RAVDESS"
OUTPUT_DIR="../dataset/RAVDESS_fixed"
SAMPLE_RATE=16000

# Create the output directory
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/train
mkdir -p $OUTPUT_DIR/val
mkdir -p $OUTPUT_DIR/test

echo -e "${GREEN}Creating train, validation, and test splits for RAVDESS dataset...${NC}"

# Run the Python script to prepare the dataset with proper actor splits
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

# Create output directories
Path(output_dir).mkdir(exist_ok=True, parents=True)
Path(os.path.join(output_dir, "train")).mkdir(exist_ok=True)
Path(os.path.join(output_dir, "val")).mkdir(exist_ok=True)
Path(os.path.join(output_dir, "test")).mkdir(exist_ok=True)

# Get all actor directories
actor_dirs = [d for d in os.listdir(dataset_root) if d.startswith("Actor_")]
actor_dirs.sort()

print(f"Found {len(actor_dirs)} actors in the dataset.")

# Split actors: 70% train, 15% val, 15% test
n_actors = len(actor_dirs)
n_train = int(0.7 * n_actors)
n_val = int(0.15 * n_actors)

# Set seed for reproducibility
random.seed(42)
random.shuffle(actor_dirs)

train_actors = actor_dirs[:n_train]
val_actors = actor_dirs[n_train:n_train+n_val]
test_actors = actor_dirs[n_train+n_val:]

# Create a mapping to show which actors are in which split
split_mapping = {
    'train': train_actors,
    'val': val_actors,
    'test': test_actors
}

# Print mapping
print("\nActor split mapping:")
for split, actors in split_mapping.items():
    print(f"{split}: {', '.join(actors)}")

# Clone the directory structure but with actor split
for split, actors in split_mapping.items():
    for actor in actors:
        src_dir = os.path.join(dataset_root, actor)
        dst_dir = os.path.join(output_dir, split, actor)
        Path(dst_dir).mkdir(exist_ok=True, parents=True)
        
        # Copy all .wav files from source to destination
        for wav_file in Path(src_dir).glob("*.wav"):
            shutil.copy2(wav_file, dst_dir)
            
# Count files in each split
train_files = sum([len(list(Path(os.path.join(output_dir, "train", actor)).glob("*.wav"))) for actor in train_actors])
val_files = sum([len(list(Path(os.path.join(output_dir, "val", actor)).glob("*.wav"))) for actor in val_actors])
test_files = sum([len(list(Path(os.path.join(output_dir, "test", actor)).glob("*.wav"))) for actor in test_actors])
total_files = train_files + val_files + test_files

print(f"\nDataset preparation complete!")
print(f"Total files: {total_files}")
print(f"Training files: {train_files} ({train_files/total_files*100:.1f}%)")
print(f"Validation files: {val_files} ({val_files/total_files*100:.1f}%)")
print(f"Testing files: {test_files} ({test_files/total_files*100:.1f}%)")

# Create metadata files
with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
    f.write(f"Total files: {total_files}\n")
    f.write(f"Training files: {train_files} ({train_files/total_files*100:.1f}%)\n")
    f.write(f"Validation files: {val_files} ({val_files/total_files*100:.1f}%)\n")
    f.write(f"Testing files: {test_files} ({test_files/total_files*100:.1f}%)\n\n")
    f.write("Actor split mapping:\n")
    for split, actors in split_mapping.items():
        f.write(f"{split}: {', '.join(actors)}\n")
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