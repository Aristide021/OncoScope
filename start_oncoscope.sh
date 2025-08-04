#!/bin/bash
# OncoScope Startup Script

echo "🧬 OncoScope Startup Menu"
echo "========================"
echo "1) Start Backend API only"
echo "2) Start Electron Desktop App (with backend)"
echo "3) Start Gradio Web Interface"
echo "4) Start Gradio Demo (no backend needed)"
echo "5) Start All Services"
echo "6) Exit"
echo ""
read -p "Select option (1-6): " choice

case $choice in
    1)
        echo "Starting Backend API..."
        cd "$(dirname "$0")"
        python3 -m backend.main
        ;;
    2)
        echo "Starting Electron Desktop App..."
        # Start backend first
        cd "$(dirname "$0")"
        echo "Starting backend..."
        python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
        BACKEND_PID=$!
        echo "Backend PID: $BACKEND_PID"
        
        # Wait for backend to be ready
        echo "Waiting for backend to start..."
        sleep 3
        
        # Check if backend is running
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "Backend is ready!"
        else
            echo "Backend failed to start. Check the logs."
            kill $BACKEND_PID 2>/dev/null
            exit 1
        fi
        
        # Start Electron with backend already running
        cd frontend
        SKIP_BACKEND=true npm start
        
        # Kill backend when Electron exits
        kill $BACKEND_PID 2>/dev/null
        ;;
    3)
        echo "Starting Gradio Web Interface..."
        echo "Make sure backend is running on port 8000!"
        cd "$(dirname "$0")"
        python3 gradio_app.py
        ;;
    4)
        echo "Starting Gradio Demo..."
        cd "$(dirname "$0")"
        python3 gradio_app_demo.py
        ;;
    5)
        echo "Starting all services..."
        cd "$(dirname "$0")"
        # Start backend in background
        python3 -m backend.main &
        BACKEND_PID=$!
        echo "Backend PID: $BACKEND_PID"
        
        # Wait for backend to start
        echo "Waiting for backend to start..."
        sleep 5
        
        # Start Electron app
        cd frontend
        npm start
        
        # When Electron closes, kill backend
        kill $BACKEND_PID 2>/dev/null
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac