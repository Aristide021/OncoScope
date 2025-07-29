@echo off
REM OncoScope Judge Environment Setup Script for Windows
REM This script creates a Python 3.11 virtual environment with all required dependencies

echo ==========================================
echo OncoScope Fine-tuning Environment Setup
echo ==========================================

REM Check if Python 3.11 is available
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 is required but not found.
    echo Please install Python 3.11 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python 3.11 found: 
py -3.11 --version

REM Create virtual environment
echo.
echo Creating Python 3.11 virtual environment...
py -3.11 -m venv oncoscope_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call oncoscope_env\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch with CUDA
    pause
    exit /b 1
)

REM Install Unsloth for Windows
echo.
echo Installing Unsloth for Windows...
python -m pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
if errorlevel 1 (
    echo ERROR: Failed to install Unsloth
    pause
    exit /b 1
)

REM Install timm for Gemma 3N vision support
echo.
echo Installing timm for vision support...
python -m pip install timm
if errorlevel 1 (
    echo ERROR: Failed to install timm
    pause
    exit /b 1
)

REM Install additional requirements
echo.
echo Installing additional requirements...
python -m pip install datasets>=3.4.1 transformers>=4.51.3 accelerate>=0.34.1 peft>=0.7.1 trl>=0.7.9

REM Test installation
echo.
echo Testing installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -c "import unsloth; print('Unsloth installed successfully')"

echo.
echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo.
echo To use this environment:
echo 1. Run: oncoscope_env\Scripts\activate.bat
echo 2. Navigate to the fine_tuning directory
echo 3. Run: python train_cancer_model.py
echo.
echo Environment is ready for OncoScope fine-tuning!
pause