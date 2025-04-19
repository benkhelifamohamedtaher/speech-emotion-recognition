#!/bin/bash
# Script to set up the RAVDESS dataset structure

# Set text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Setting up RAVDESS Dataset =====${NC}"

SOURCE_DIR="dataset/Audio_Speech_Actors_01-24"
TARGET_DIR="dataset/RAVDESS"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Count number of WAV files in source directory
NUM_FILES=$(find "$SOURCE_DIR" -name "*.wav" | wc -l)
echo -e "${GREEN}Found $NUM_FILES audio files in source directory${NC}"

# Copy files from each Actor directory
for ACTOR_DIR in "$SOURCE_DIR"/Actor_*; do
    # Extract actor number
    ACTOR_NUM=$(basename "$ACTOR_DIR" | sed 's/Actor_//')
    
    # Check if actor number is two digits
    if [ ${#ACTOR_NUM} -eq 2 ]; then
        # Remove leading zero if present
        ACTOR_NUM=$(echo "$ACTOR_NUM" | sed 's/^0//')
    fi
    
    echo -e "${YELLOW}Processing Actor $ACTOR_NUM...${NC}"
    
    # Create target actor directory
    TARGET_ACTOR_DIR="$TARGET_DIR/Actor_$ACTOR_NUM"
    mkdir -p "$TARGET_ACTOR_DIR"
    
    # Copy all WAV files
    find "$ACTOR_DIR" -name "*.wav" -exec cp {} "$TARGET_ACTOR_DIR/" \;
    
    # Count files copied
    FILES_COPIED=$(find "$TARGET_ACTOR_DIR" -name "*.wav" | wc -l)
    echo -e "${GREEN}  Copied $FILES_COPIED files to $TARGET_ACTOR_DIR${NC}"
done

# Verify total files copied
TOTAL_COPIED=$(find "$TARGET_DIR" -name "*.wav" | wc -l)
echo -e "${BLUE}Total files copied: $TOTAL_COPIED / $NUM_FILES${NC}"

if [ "$TOTAL_COPIED" -eq "$NUM_FILES" ]; then
    echo -e "${GREEN}RAVDESS dataset setup complete!${NC}"
else
    echo -e "${RED}Warning: Not all files were copied ($TOTAL_COPIED/$NUM_FILES)${NC}"
fi

echo -e "${YELLOW}The dataset is now ready for training.${NC}" 