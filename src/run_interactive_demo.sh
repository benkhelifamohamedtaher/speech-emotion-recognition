#!/bin/bash
# run_interactive_demo.sh - Script to run the interactive emotion recognition demo

# Terminal colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set defaults
MODEL_TYPE="enhanced"
DEVICE="cpu"
VISUALIZE=true
SAMPLE_RATE=16000

# Print banner
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}    Real-Time Speech Emotion Recognition Demo    ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      MODEL_TYPE="$2"
      shift
      shift
      ;;
    --device)
      DEVICE="$2"
      shift
      shift
      ;;
    --no-visualize)
      VISUALIZE=false
      shift
      ;;
    --sample-rate)
      SAMPLE_RATE="$2"
      shift
      shift
      ;;
    --help|-h)
      echo -e "Usage: $0 [options]"
      echo -e "Options:"
      echo -e "  --model TYPE        Model type: 'base', 'enhanced', or 'ensemble' (default: enhanced)"
      echo -e "  --device DEVICE     Computation device: 'cpu' or 'cuda' (default: cpu)"
      echo -e "  --no-visualize      Run without visualization"
      echo -e "  --sample-rate RATE  Audio sample rate (default: 16000)"
      echo -e "  --help, -h          Show this help message"
      exit 0
      ;;
    *)
      echo -e "${YELLOW}Warning: Unknown option '$key'${NC}"
      shift
      ;;
  esac
done

# Set model paths based on model type
case $MODEL_TYPE in
  base)
    MODEL_PATH="../models/base_model/model_final.pt"
    CONFIG_PATH="../models/base_model/config.json"
    echo -e "${GREEN}Using base model${NC}"
    ;;
  enhanced)
    MODEL_PATH="../models/enhanced_model/model_final.pt"
    CONFIG_PATH="../models/enhanced_model/config.json"
    echo -e "${GREEN}Using enhanced model${NC}"
    ;;
  ensemble)
    MODEL_PATH="../models/ensemble_model/model_final.pt"
    CONFIG_PATH="../models/ensemble_model/config.json"
    echo -e "${GREEN}Using ensemble model${NC}"
    ;;
  *)
    echo -e "${YELLOW}Unknown model type '$MODEL_TYPE', defaulting to enhanced model${NC}"
    MODEL_PATH="../models/enhanced_model/model_final.pt"
    CONFIG_PATH="../models/enhanced_model/config.json"
    ;;
esac

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo -e "${YELLOW}Warning: Model file not found at $MODEL_PATH${NC}"
  echo -e "${YELLOW}You may need to train a model first using:${NC}"
  echo -e "${YELLOW}  bash train_ravdess.sh${NC}"
  echo -e "${YELLOW}Continuing anyway in case you're using a custom model path...${NC}"
fi

# Set visualization flag
if [ "$VISUALIZE" = true ]; then
  VISUALIZE_ARG="--visualize"
  echo -e "${GREEN}Visualization: Enabled${NC}"
else
  VISUALIZE_ARG=""
  echo -e "${GREEN}Visualization: Disabled${NC}"
fi

# Set PYTHONPATH to include current directory and parent
export PYTHONPATH="$(pwd):$(pwd)/..:$PYTHONPATH"

# Build the command
CMD="python interactive_demo.py --model_path $MODEL_PATH --config_path $CONFIG_PATH --device $DEVICE --sample_rate $SAMPLE_RATE $VISUALIZE_ARG"

echo -e "${BLUE}Starting demo with:${NC}"
echo -e "  ${GREEN}Model:${NC} $MODEL_PATH"
echo -e "  ${GREEN}Device:${NC} $DEVICE"
echo -e "  ${GREEN}Sample Rate:${NC} $SAMPLE_RATE"
echo -e "${BLUE}=======================================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to exit the demo${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Run the command
echo "$CMD"
eval "$CMD"

# Check if command execution was successful
if [ $? -eq 0 ]; then
  echo -e "${GREEN}Demo completed successfully.${NC}"
else
  echo -e "${YELLOW}Demo exited with an error. See above for details.${NC}"
fi 