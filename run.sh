#!/bin/bash
# OncoScope - Privacy-First Cancer Genomics Analysis Platform
# Production deployment script for competition and clinical environments

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ONCOSCOPE_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.8"
VENV_PATH="${ONCOSCOPE_HOME}/venv"
OLLAMA_MODEL="oncoscope-cancer"
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-0.0.0.0}"

# Logging
LOG_DIR="${ONCOSCOPE_HOME}/logs"
LOG_FILE="${LOG_DIR}/oncoscope-startup.log"

# Create log directory
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

print_header() {
    echo -e "${BLUE}"
    echo "TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW"
    echo "Q                         OncoScope                           Q"
    echo "Q            Privacy-First Cancer Genomics Analysis           Q"
    echo "Q                        Version 1.0.0                        Q"
    echo "ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]"
    echo -e "${NC}"
}

check_system_requirements() {
    log "${BLUE}[INFO]${NC} Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log "${RED}[ERROR]${NC} Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log "${GREEN}[OK]${NC} Python ${PYTHON_VER} detected"
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "${MEMORY_GB}" -lt 4 ]; then
            log "${YELLOW}[WARNING]${NC} Less than 4GB RAM detected. OncoScope may run slowly."
        else
            log "${GREEN}[OK]${NC} ${MEMORY_GB}GB RAM available"
        fi
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h "${ONCOSCOPE_HOME}" | awk 'NR==2{print $4}')
    log "${GREEN}[OK]${NC} ${DISK_SPACE} disk space available"
}

setup_python_environment() {
    log "${BLUE}[INFO]${NC} Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "${VENV_PATH}" ]; then
        log "${BLUE}[INFO]${NC} Creating virtual environment..."
        python3 -m venv "${VENV_PATH}"
    fi
    
    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [ -f "${ONCOSCOPE_HOME}/requirements.txt" ]; then
        log "${BLUE}[INFO]${NC} Installing Python dependencies..."
        pip install -r "${ONCOSCOPE_HOME}/requirements.txt"
    fi
    
    log "${GREEN}[OK]${NC} Python environment ready"
}

check_ollama() {
    log "${BLUE}[INFO]${NC} Checking Ollama installation..."
    
    if ! command -v ollama &> /dev/null; then
        log "${YELLOW}[WARNING]${NC} Ollama not found. AI features will be limited."
        log "${BLUE}[INFO]${NC} Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
        return 1
    fi
    
    # Check if Ollama is running
    if ! pgrep -f ollama > /dev/null; then
        log "${BLUE}[INFO]${NC} Starting Ollama server..."
        ollama serve &
        sleep 5
    fi
    
    # Check if OncoScope model is available
    if ollama list | grep -q "${OLLAMA_MODEL}"; then
        log "${GREEN}[OK]${NC} OncoScope AI model ready"
    else
        log "${YELLOW}[WARNING]${NC} OncoScope AI model not found. Using fallback analysis."
        log "${BLUE}[INFO]${NC} To install: ollama pull ${OLLAMA_MODEL}"
    fi
    
    return 0
}

validate_data() {
    log "${BLUE}[INFO]${NC} Validating OncoScope data files..."
    
    DATA_DIR="${ONCOSCOPE_HOME}/data"
    REQUIRED_FILES=(
        "cosmic_mutations.json"
        "drug_associations.json"
        "cancer_types.json"
    )
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "${DATA_DIR}/${file}" ]; then
            SIZE=$(stat -c%s "${DATA_DIR}/${file}" 2>/dev/null || stat -f%z "${DATA_DIR}/${file}" 2>/dev/null || echo "0")
            if [ "${SIZE}" -gt 1000 ]; then
                log "${GREEN}[OK]${NC} ${file} (${SIZE} bytes)"
            else
                log "${YELLOW}[WARNING]${NC} ${file} seems small (${SIZE} bytes)"
            fi
        else
            log "${RED}[ERROR]${NC} Missing required data file: ${file}"
            exit 1
        fi
    done
}

start_oncoscope() {
    log "${BLUE}[INFO]${NC} Starting OncoScope server..."
    
    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"
    
    # Set environment variables
    export PYTHONPATH="${ONCOSCOPE_HOME}:${PYTHONPATH:-}"
    export ONCOSCOPE_DATA_DIR="${ONCOSCOPE_HOME}/data"
    export API_HOST="${API_HOST}"
    export API_PORT="${API_PORT}"
    
    # Start the server
    cd "${ONCOSCOPE_HOME}"
    
    log "${GREEN}[STARTING]${NC} OncoScope server on http://${API_HOST}:${API_PORT}"
    log "${BLUE}[INFO]${NC} Frontend available at: ${ONCOSCOPE_HOME}/frontend/index.html"
    log "${BLUE}[INFO]${NC} API documentation: http://${API_HOST}:${API_PORT}/docs"
    log "${BLUE}[INFO]${NC} Logs: ${LOG_FILE}"
    
    # Run the application
    python -m backend.main
}

cleanup() {
    log "${BLUE}[INFO]${NC} Shutting down OncoScope..."
    # Kill any remaining processes
    pkill -f "oncoscope" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    log "${GREEN}[OK]${NC} OncoScope stopped"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    print_header
    
    log "${BLUE}[INFO]${NC} Starting OncoScope initialization..."
    log "${BLUE}[INFO]${NC} Working directory: ${ONCOSCOPE_HOME}"
    log "${BLUE}[INFO]${NC} Log file: ${LOG_FILE}"
    
    # Run startup checks
    check_system_requirements
    setup_python_environment
    check_ollama
    validate_data
    
    log "${GREEN}[SUCCESS]${NC} All system checks passed!"
    log "${BLUE}[INFO]${NC} OncoScope is ready for cancer genomics analysis"
    
    # Start the application
    start_oncoscope
}

# Help function
show_help() {
    echo "OncoScope - Privacy-First Cancer Genomics Analysis Platform"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --validate Run validation checks only"
    echo "  -p, --port     Set API port (default: 8000)"
    echo "  --host         Set API host (default: 0.0.0.0)"
    echo ""
    echo "Environment Variables:"
    echo "  API_HOST       Server host (default: 0.0.0.0)"
    echo "  API_PORT       Server port (default: 8000)"
    echo "  DEBUG_MODE     Enable debug mode (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start OncoScope with default settings"
    echo "  $0 -p 8080          # Start on port 8080"
    echo "  $0 --validate       # Run validation checks only"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--validate)
            print_header
            check_system_requirements
            setup_python_environment
            check_ollama
            validate_data
            log "${GREEN}[SUCCESS]${NC} All validation checks passed!"
            exit 0
            ;;
        -p|--port)
            API_PORT="$2"
            shift 2
            ;;
        --host)
            API_HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main "$@"