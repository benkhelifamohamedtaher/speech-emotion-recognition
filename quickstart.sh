#!/bin/bash
# Quickstart script for Real-Time Speech Emotion Recognition

# Set text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Real-Time Speech Emotion Recognition Quickstart =====${NC}"
echo

# Check if Python 3 is installed
if command -v python3 &>/dev/null; then
    echo -e "${GREEN}✓ Python 3 is installed${NC}"
    PYTHON="python3"
else
    if command -v python &>/dev/null; then
        # Check Python version
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
        if [ "$PYTHON_VERSION" -eq 3 ]; then
            echo -e "${GREEN}✓ Python 3 is installed${NC}"
            PYTHON="python"
        else
            echo -e "${RED}✗ Python 3 is required${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Python is not installed. Please install Python 3.8 or newer.${NC}"
        exit 1
    fi
fi

# Make verification script executable
chmod +x src/check_project.py

# Check project structure
echo -e "\n${YELLOW}Checking project structure...${NC}"
$PYTHON src/check_project.py

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "\n${YELLOW}Would you like to install the required dependencies? (y/n)${NC}"
    read -r install_deps
    if [[ $install_deps == "y" || $install_deps == "Y" ]]; then
        echo -e "\n${YELLOW}Installing dependencies...${NC}"
        $PYTHON -m pip install -r requirements.txt
        
        # Check if PyAudio installation was successful
        if $PYTHON -c "import pyaudio" &>/dev/null; then
            echo -e "${GREEN}✓ PyAudio installed successfully${NC}"
        else
            echo -e "${YELLOW}⚠ PyAudio installation might have failed.${NC}"
            echo -e "   On macOS, try: ${BLUE}brew install portaudio && pip install pyaudio${NC}"
            echo -e "   On Ubuntu/Debian, try: ${BLUE}sudo apt-get install python3-pyaudio${NC}"
            echo -e "   On Windows, try: ${BLUE}pip install pipwin && pipwin install pyaudio${NC}"
        fi
    fi
fi

# Ask if the user wants to prepare the dataset
echo -e "\n${YELLOW}Would you like to prepare the RAVDESS dataset? (y/n)${NC}"
read -r prepare_dataset
if [[ $prepare_dataset == "y" || $prepare_dataset == "Y" ]]; then
    if [ -f "src/prepare_ravdess.py" ]; then
        echo -e "\n${YELLOW}Preparing dataset...${NC}"
        $PYTHON src/prepare_ravdess.py
    else
        echo -e "${RED}✗ Dataset preparation script not found${NC}"
    fi
fi

# Ask if the user wants to train a model
echo -e "\n${YELLOW}Would you like to train a model? (y/n)${NC}"
read -r train_model
if [[ $train_model == "y" || $train_model == "Y" ]]; then
    if [ -f "src/train_optimal_model.sh" ]; then
        echo -e "\n${YELLOW}Training model...${NC}"
        bash src/train_optimal_model.sh
    else
        # Try alternative training scripts
        if [ -f "src/train_ravdess.sh" ]; then
            echo -e "\n${YELLOW}Training model using train_ravdess.sh...${NC}"
            bash src/train_ravdess.sh
        else
            echo -e "${RED}✗ Training script not found${NC}"
        fi
    fi
fi

# Ask if the user wants to run inference
echo -e "\n${YELLOW}Would you like to run real-time inference? (y/n)${NC}"
read -r run_inference
if [[ $run_inference == "y" || $run_inference == "Y" ]]; then
    if [ -f "src/optimal_inference.py" ]; then
        echo -e "\n${YELLOW}Running inference...${NC}"
        $PYTHON src/optimal_inference.py
    else
        # Try alternative inference scripts
        if [ -f "src/inference.py" ]; then
            echo -e "\n${YELLOW}Running inference using inference.py...${NC}"
            $PYTHON src/inference.py
        elif [ -f "src/simple_console_inference.py" ]; then
            echo -e "\n${YELLOW}Running simple console inference...${NC}"
            $PYTHON src/simple_console_inference.py
        else
            echo -e "${RED}✗ Inference script not found${NC}"
        fi
    fi
fi

echo -e "\n${GREEN}Quickstart completed!${NC}"
echo -e "For more information, please read the README.md file." 