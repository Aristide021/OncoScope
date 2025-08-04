# How to Run OncoScope Apps

## Quick Start

Use the provided scripts:
```bash
./start_oncoscope.sh  # Interactive menu to start apps
./stop_oncoscope.sh   # Stop all running services
```

## Manual Commands

### 1. Backend API (Required for Electron and Gradio Web)
```bash
# Start backend
python3 -m backend.main

# The API will run at http://localhost:8000
# Check health: http://localhost:8000/health
# View docs: http://localhost:8000/docs
```

### 2. Electron Desktop App
```bash
# First time setup (if not done)
cd frontend
npm install

# Run the app
npm start

# Or run without auto-starting backend (if backend already running)
SKIP_BACKEND=true npm start
```

### 3. Gradio Web Interface
```bash
# Requires backend to be running first!
python3 gradio_app.py

# Opens at http://localhost:7860
```

### 4. Gradio Demo (Standalone)
```bash
# No backend required - uses mock data
python3 gradio_app_demo.py

# Opens at http://localhost:7860
```

## Stopping Services

### Stop specific service:
```bash
# Stop backend
pkill -f "python.*backend.main"

# Stop Electron app
# Just close the window or Cmd+Q (Mac) / Alt+F4 (Windows)

# Stop Gradio
# Press Ctrl+C in the terminal where it's running
```

### Stop everything:
```bash
./stop_oncoscope.sh
```

## Common Issues

1. **Port 8000 already in use**
   ```bash
   # Find what's using it
   lsof -i :8000
   
   # Kill it
   kill -9 <PID>
   ```

2. **Electron app closes immediately**
   - The backend might not be running
   - Check if port 8000 is accessible

3. **Gradio web app shows connection error**
   - Make sure backend is running on port 8000
   - Check: `curl http://localhost:8000/health`

## Development Tips

- Run backend with auto-reload:
  ```bash
  cd backend
  uvicorn main:app --reload --host 127.0.0.1 --port 8000
  ```

- Run Electron in dev mode:
  ```bash
  cd frontend
  npm run dev
  ```