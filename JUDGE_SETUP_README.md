# OncoScope Fine-tuning Environment Setup for Judges

This guide provides step-by-step instructions for judges to set up the OncoScope fine-tuning environment on both Windows and Linux systems.

## System Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 4070 or better recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)

## Quick Setup

### Windows Setup

1. **Download and install Python 3.11**:
   - Go to https://www.python.org/downloads/
   - Download Python 3.11.x (NOT 3.12 or 3.13)
   - During installation, check "Add Python to PATH"

2. **Run the setup script**:
   ```cmd
   cd C:\path\to\OncoScope
   setup_judge_environment.bat
   ```

3. **Activate the environment**:
   ```cmd
   oncoscope_env\Scripts\activate.bat
   ```

### Linux/WSL Setup

1. **Install Python 3.11**:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev
   
   # CentOS/RHEL
   sudo dnf install python3.11 python3.11-venv python3.11-devel
   ```

2. **Run the setup script**:
   ```bash
   cd /path/to/OncoScope
   chmod +x setup_judge_environment.sh
   ./setup_judge_environment.sh
   ```

3. **Activate the environment**:
   ```bash
   source oncoscope_env/bin/activate
   ```

## Running Fine-tuning

After setup is complete:

1. **Navigate to fine-tuning directory**:
   ```bash
   cd ai/fine_tuning
   ```

2. **Run the training script**:
   ```bash
   python train_cancer_model.py --training_data cancer_training_data.json --output_dir ./oncoscope_model --epochs 1
   ```

## Training Parameters

The fine-tuning script supports several parameters:

- `--training_data`: Path to training data (default: cancer_training_data.json)
- `--model_name`: Base model to use (default: unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit)
- `--output_dir`: Output directory for trained model (default: ./oncoscope_model)
- `--epochs`: Number of training epochs (default: 1)
- `--max_seq_length`: Maximum sequence length (default: 2048)

## Expected Output

The training process will:

1. Load the Gemma 3N model with Unsloth optimization
2. Process 5,988 cancer genomics training examples
3. Apply LoRA fine-tuning with 4-bit quantization
4. Save the model in multiple formats:
   - LoRA adapters
   - Merged model for deployment
   - GGUF format for Ollama integration

## Troubleshooting

### Common Issues

1. **"Python 3.11 not found"**:
   - Ensure Python 3.11 is installed and in PATH
   - Use `py -3.11 --version` (Windows) or `python3.11 --version` (Linux) to verify

2. **"CUDA not available"**:
   - Install NVIDIA drivers
   - Verify GPU with `nvidia-smi`
   - Reinstall PyTorch with CUDA support

3. **Memory errors**:
   - Close other applications
   - Reduce batch size in training script
   - Use smaller model if necessary

4. **Import errors**:
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

### Windows-Specific Issues

- **Multiprocessing errors**: The script is configured with `num_proc=1` for Windows compatibility
- **Path issues**: Use forward slashes or escape backslashes in paths
- **Antivirus**: Whitelist the Python environment folder

### GPU Memory Requirements

- **Gemma 3N E4B**: ~6-7GB VRAM (recommended)
- **Gemma 3N E2B**: ~4-5GB VRAM (fallback)
- **Training overhead**: Additional 1-2GB during training

## Support

If you encounter issues during setup or training:

1. Check the error logs in the terminal
2. Verify all requirements are met
3. Try the fallback model: `--model_name "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"`
4. Contact technical support with full error messages

## Environment Details

The setup script installs:

- **PyTorch 2.7.1** with CUDA 12.8 support
- **Unsloth 2025.7.11** for Windows/Linux
- **Transformers 4.54.1** with Gemma 3N support
- **TRL, PEFT, Accelerate** for fine-tuning
- **Datasets, Tokenizers** for data processing
- **TIMM** for vision components (disabled for text-only training)

This ensures reproducible results across different judge environments.