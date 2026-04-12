#!/bin/bash
# Start the SloughGPT API server
# Uses system Python which has torch installed

cd /Users/mac/sloughGPT

echo "Starting SloughGPT API server..."
echo "Using Python: $(which python3)"

# Run the API server
python3 apps/api/server/main.py
