#!/bin/bash
# Run the system check script

# Make the script executable if it isn't already
chmod +x check_project.py

# Run the check script
./check_project.py

# Store the exit code
exit_code=$?

# If the exit code is not 0, provide additional guidance
if [ $exit_code -ne 0 ]; then
  echo ""
  echo "For detailed installation instructions, please refer to the README.md file."
  echo "You can install all required dependencies using: pip install -r requirements.txt"
fi

exit $exit_code 