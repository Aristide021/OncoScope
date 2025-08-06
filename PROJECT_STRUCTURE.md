# OncoScope Project Structure

## 📁 Directory Overview

```
OncoScope/
├── 📱 frontend/               # Electron desktop application
│   ├── index.html            # Main application UI
│   ├── renderer.js           # Frontend logic
│   ├── styles.css            # Application styling
│   └── main.js               # Electron main process
│
├── 🔧 backend/                # FastAPI backend server
│   ├── main.py               # API endpoints
│   ├── mutation_analyzer.py  # Core analysis engine
│   ├── clustering_engine.py  # Mutation clustering
│   ├── models.py             # Pydantic data models
│   └── database.py           # Database operations
│
├── 🧠 ai/                     # AI model integration
│   ├── inference/            # Ollama integration
│   │   ├── ollama_client.py  # Gemma 3n interface
│   │   └── prompts.py        # AI prompt engineering
│   └── fine_tuning/          # Model fine-tuning
│       ├── train_gemma.py    # Training script
│       └── datasets/         # COSMIC training data
│
├── 📊 data/                   # Database files
│   ├── cosmic_mutations.json # Mutation database
│   ├── drug_associations.json# Drug-target mappings
│   └── cancer_types.json     # Cancer classifications
│
├── 📐 diagrams/               # Architecture diagrams
│   ├── *.mmd                 # Mermaid source files
│   ├── *.svg                 # Vector graphics
│   └── *.png                 # Raster images
│
├── 🧪 tests/                  # Test suite
│   ├── test_mutations.py     # Mutation analysis tests
│   ├── test_api.py           # API endpoint tests
│   └── test_ai.py            # AI integration tests
│
├── 📜 scripts/                # Utility scripts
│   ├── setup_ollama.py       # Model installation
│   ├── download_drug_data.py # Data preparation
│   └── validate_installation.py # System check
│
└── 📋 docs/                   # Documentation
    ├── API.md                # API reference
    ├── DEPLOYMENT.md         # Deployment guide
    └── TROUBLESHOOTING.md    # Common issues
```

## 🔑 Key Files

### Core Application
- `backend/mutation_analyzer.py` - Heart of the analysis engine
- `ai/inference/ollama_client.py` - Gemma 3n integration
- `frontend/renderer.js` - UI interaction logic

### Configuration
- `backend/config.py` - Application settings
- `package.json` - Node.js dependencies
- `requirements.txt` - Python dependencies

### AI Model
- `ai/fine_tuning/oncoscope-gemma-3n-gguf/` - Fine-tuned model
- `ai/inference/prompts.py` - Prompt engineering

## 🚀 Quick Navigation

- **Want to see the AI magic?** → `ai/inference/ollama_client.py`
- **Curious about analysis logic?** → `backend/mutation_analyzer.py`
- **Interested in the UI?** → `frontend/index.html` & `renderer.js`
- **Looking for fine-tuning?** → `ai/fine_tuning/train_gemma.py`

## 💡 Architecture Highlights

1. **Modular Design**: Clean separation of concerns
2. **Async First**: Non-blocking I/O throughout
3. **Type Safety**: Pydantic models everywhere
4. **Privacy by Design**: No external data transmission
5. **Offline Capable**: All processing happens locally