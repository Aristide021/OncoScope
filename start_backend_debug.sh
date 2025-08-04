#!/bin/bash
# Start backend with debug logging

echo "Starting OncoScope backend with debug logging..."
echo "This will show all AI model calls"
echo ""

cd "$(dirname "$0")"

# Clear Python cache
echo "Clearing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Set environment variables for debugging
export PYTHONUNBUFFERED=1
export LOG_LEVEL=DEBUG

# Start backend
echo "Starting backend..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level debug --reload