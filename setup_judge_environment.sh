#!/bin/bash
# OncoScope Judge Environment Setup Script for Linux/WSL
# This script creates a Python 3.11 virtual environment with all required dependencies

echo "=========================================="
echo "OncoScope Fine-tuning Environment Setup"
echo "=========================================="

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "ERROR: Python 3.11 is required but not found."
    echo "Please install Python 3.11:"
    echo "Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo "CentOS/RHEL: sudo dnf install python3.11 python3.11-venv python3.11-devel"
    exit 1
fi

echo "Python 3.11 found: $(python3.11 --version)"

# Create virtual environment
echo ""
echo "Creating Python 3.11 virtual environment..."
python3.11 -m venv oncoscope_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source oncoscope_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyTorch with CUDA"
    exit 1
fi

# Install Unsloth
echo ""
echo "Installing Unsloth..."
python -m pip install unsloth
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Unsloth"
    exit 1
fi

# Install timm for Gemma 3N vision support
echo ""
echo "Installing timm for vision support..."
python -m pip install timm
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install timm"
    exit 1
fi

# Install additional requirements
echo ""
echo "Installing additional requirements..."
python -m pip install datasets>=3.4.1 transformers>=4.51.3 accelerate>=0.34.1 peft>=0.7.1 trl>=0.7.9

# Test installation
echo ""
echo "Testing installation..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -c "import unsloth; print('Unsloth installed successfully')"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "1. Run: source oncoscope_env/bin/activate"
echo "2. Navigate to the fine_tuning directory"
echo "3. Run: python train_cancer_model.py"
echo ""
echo "Environment is ready for OncoScope fine-tuning!"