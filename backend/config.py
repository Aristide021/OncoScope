"""
OncoScope Configuration Module
"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Application
    app_name: str = "OncoScope"
    app_version: str = "1.0.0"
    debug_mode: bool = True
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "oncoscope-cancer"
    ollama_timeout: int = 300  # Increased to 5 minutes for complex analyses
    
    # Database
    database_url: str = "sqlite:///./oncoscope.db"
    
    # Security
    secret_key: str = "development-secret-key-change-in-production"
    cors_origins: List[str] = ["http://localhost:3000", "file://", "http://localhost", "*"]
    
    # AI Model Parameters
    model_temperature: float = 0.1
    model_top_p: float = 0.9
    model_max_tokens: int = 1024
    
    # Paths
    data_dir: Path = Path(__file__).parent.parent / "data"
    upload_dir: Path = Path(__file__).parent.parent / "uploads"
    
    # Data Processing (OncoScope Expert Features)
    use_expert_clinvar: bool = True
    use_targeted_cosmic: bool = True
    use_fda_verified_drugs: bool = True
    min_pathogenicity_score: float = 0.7
    
    # Performance Settings
    max_concurrent_analyses: int = 3
    cache_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file_path: str = "logs/oncoscope.log"
    
    # Development Flags
    dev_mode: bool = True
    enable_telemetry: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True, parents=True)
settings.upload_dir.mkdir(exist_ok=True, parents=True)

# Create logs directory
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True, parents=True)