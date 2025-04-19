#!/bin/bash
# Cleanup script to remove large data directories before pushing to GitHub

# Set text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting cleanup of large directories before GitHub push...${NC}"

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
  echo -e "${YELLOW}Initializing Git repository...${NC}"
  git init
  echo -e "${GREEN}Git repository initialized.${NC}"
fi

# Create backup directory for important files (if needed)
if [ ! -d "backup_config" ]; then
  mkdir -p backup_config
  echo -e "${GREEN}Created backup directory.${NC}"
fi

# Large directories to be excluded from Git
LARGE_DIRS=(
  "dataset"
  "dataset_raw"
  "processed_dataset"
  "models"
  "outputs"
  "tensorboard_test"
)

# Check if directories exist and notify user about their sizes
echo -e "${YELLOW}Checking sizes of large directories:${NC}"
for dir in "${LARGE_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    size=$(du -sh "$dir" | cut -f1)
    echo -e "  - ${dir}: ${size}"
  fi
done

# Save a list of large directories as a reference
echo "The following directories were excluded from Git due to their size:" > backup_config/excluded_directories.txt
for dir in "${LARGE_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    size=$(du -sh "$dir" | cut -f1)
    echo "  - ${dir}: ${size}" >> backup_config/excluded_directories.txt
  fi
done

# Make sure .gitignore includes all large directories
echo -e "${YELLOW}Checking .gitignore file...${NC}"
if [ ! -f ".gitignore" ]; then
  echo -e "${RED}No .gitignore file found. Creating one...${NC}"
  touch .gitignore
fi

# Add directories to .gitignore if not already there
for dir in "${LARGE_DIRS[@]}"; do
  if ! grep -q "^${dir}/$" .gitignore; then
    echo "${dir}/" >> .gitignore
    echo -e "  - Added ${dir}/ to .gitignore"
  fi
done

echo -e "${GREEN}Cleanup complete! You can now safely push to GitHub.${NC}"
echo -e "${YELLOW}Note: The large directories still exist on your local machine but won't be pushed to GitHub.${NC}"
echo -e "${YELLOW}To push to GitHub, run:${NC}"
echo -e "  git add ."
echo -e "  git commit -m \"Initial commit\""
echo -e "  git push origin main" 