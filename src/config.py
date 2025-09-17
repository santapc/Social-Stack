"""
Central configuration for Digital-MoSJE project.
"""

import os
from pathlib import Path
try:
    import tomllib  
except ImportError:
    import tomli as tomllib  



# Load configuration from config.toml
CONFIG_FILE = Path(__file__).parent.parent / "config.toml"

with open('config.toml', 'rb') as f:
    config = tomllib.load(f)
    

# Model names
groq_models = config["models"]["groq_models"]
gpt_models = config["models"]["gpt_models"]
ollama_models = config["models"]["ollama_models"]
LLM_MODELS = groq_models + gpt_models + ollama_models

# Dataset names
DATASET_NAMES = config["datasets"]["names"]

# External tool paths (override with env vars if set)
TESSERACT_PATH = os.getenv('TESSERACT_PATH', config["paths"]["tesseract"])
POPPLER_PATH = os.getenv('POPPLER_PATH', config["paths"]["poppler"])

# Digilocker configuration
REDIRECT_URL = config["digilocker"]["redirect_url"]

