# OncoScope Environment Configuration

Simple configuration guide.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit if needed** (defaults work for most cases):
   ```bash
   nano .env  # optional customization
   ```

3. **Start OncoScope:**
   ```bash
   python -m oncoscope.backend.main
   ```

## Configuration Overview

OncoScope uses local-only configuration suitable for competition evaluation:

### 🖥️ Local Server
```bash
API_HOST=localhost        # Local development only
API_PORT=8000            # Default port
DEBUG_MODE=true          # Enable for development
```

### 🤖 Local AI Model
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=oncoscope-cancer
MODEL_TEMPERATURE=0.1    # Deterministic responses
```

### 💾 Local Database
```bash
DATABASE_URL=sqlite:///./oncoscope.db  # Local SQLite file
```

### 🧬 OncoScope Data Features
```bash
# Premium mutation analysis features
USE_PREMIUM_CLINVAR=true     # Expert panel variants
USE_TARGETED_COSMIC=true     # Validated COSMIC mutations
USE_FDA_VERIFIED_DRUGS=true  # FDA-approved therapies
MIN_PATHOGENICITY_SCORE=0.7  # Quality threshold
```

### ⚡ Performance
```bash
MAX_CONCURRENT_ANALYSES=3  # Parallel processing limit
CACHE_ENABLED=true        # Enable result caching
```

## Competition Features

OncoScope is designed for **offline evaluation** with:

- ✅ **No external APIs required** - All data is bundled
- ✅ **Local AI models** - Runs entirely on Ollama
- ✅ **SQLite database** - No server setup needed
- ✅ **Self-contained** - Works without internet
- ✅ **Premium datasets** - 41 validated mutations, FDA drugs, expert panel data

## Files Structure

```
OncoScope/
├── .env.example          # Template configuration
├── .env                  # Your local settings
├── oncoscope/
│   ├── data/            # Bundled datasets (no downloads needed)
│   ├── backend/         # API server
│   └── frontend/        # Web interface
└── logs/                # Application logs
```

## Validation

Test your configuration:

```bash
# Verify environment loading
python -c "from oncoscope.backend.config import settings; print('✅ Config OK')"

# Run system validation
python oncoscope/scripts/validate_installation.py

# Start the application
python -m oncoscope.backend.main
```

## Troubleshooting

### Common Issues

**Config loading fails:**
```bash
pip install pydantic-settings python-dotenv
```

**Ollama not found:**
```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

**Port already in use:**
```bash
# Change port in .env
API_PORT=8001
```

## Competition Notes

- **No internet required** - All datasets are pre-bundled
- **No API keys needed** - Completely self-contained
- **Local evaluation** - Designed for competition judging environment
- **Premium quality** - Uses expert-validated cancer genomics data

