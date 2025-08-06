#!/bin/bash
# Simple test script for OncoScope desktop app

echo "🧬 Testing OncoScope Desktop App"
echo "==============================="

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f "python.*backend" 2>/dev/null
pkill -f "electron" 2>/dev/null
sleep 2

# Start backend
echo "Starting backend..."
cd "$(dirname "$0")"
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend
echo "Waiting for backend to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    echo "Waiting... ($i/10)"
    sleep 1
done

# Check if backend is actually running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend failed to start!"
    exit 1
fi

# Show backend status
echo ""
echo "Backend status:"
curl -s http://localhost:8000/health | python3 -m json.tool

# Start Electron
echo ""
echo "Starting Electron app..."
echo "The desktop app window should open now."
echo "Press Ctrl+C to stop everything."
echo ""

cd frontend
SKIP_BACKEND=true npm start

# Cleanup
echo "Stopping backend..."
kill $BACKEND_PID 2>/dev/null