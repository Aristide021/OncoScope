#!/bin/bash

# OncoScope Demo Launcher
# Starts both the API server and Gradio interface

echo "🧬 OncoScope - Molecular Tumor Board in a Box"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Warning: Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Check if oncoscope-cancer model exists
if ! ollama list | grep -q "oncoscope-cancer"; then
    echo "⚠️  Warning: oncoscope-cancer model not found in Ollama"
    echo "Please run: ollama create oncoscope-cancer -f ai/fine_tuning/oncoscope-gemma-3n-gguf/Modelfile"
fi

# Start the API server in background
echo ""
echo "Starting OncoScope API server..."
uvicorn backend.main:app --host localhost --port 8000 --reload > api_server.log 2>&1 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to initialize..."
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server running at http://localhost:8000"
else
    echo "❌ API server failed to start. Check api_server.log"
    exit 1
fi

# Start Gradio interface
echo ""
echo "Starting Gradio interface..."
echo "============================================"
python gradio_app.py

# Cleanup on exit
trap "kill $API_PID 2>/dev/null" EXIT