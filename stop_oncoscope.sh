#!/bin/bash
# OncoScope Stop Script

echo "🛑 Stopping OncoScope Services..."

# Kill Python processes (backend and gradio)
echo "Stopping Python services..."
pkill -f "python.*backend.main" 2>/dev/null
pkill -f "python.*gradio_app" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null

# Kill Electron app
echo "Stopping Electron app..."
pkill -f "electron.*oncoscope" 2>/dev/null
pkill -f "Electron" 2>/dev/null
pkill -f "electron" 2>/dev/null

# Check what's still running on common ports
echo ""
echo "Checking ports..."
lsof -i :8000 2>/dev/null && echo "⚠️  Port 8000 still in use" || echo "✅ Port 8000 is free"
lsof -i :7860 2>/dev/null && echo "⚠️  Port 7860 still in use" || echo "✅ Port 7860 is free"

echo ""
echo "All OncoScope services stopped."