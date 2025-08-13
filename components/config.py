"""
Central configuration for Digital-MoSJE project.
"""

import os
from dotenv import load_dotenv
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ─── Load environment variables ───
load_dotenv()

# ─── LLM Models Setup ───
LLM_MODELS = []
GROQ_MODELS = []
OPENAI_MODELS = []

try:
    offline_models_list = ollama.list()
    OFFLINE_MODELS = [model['model'] for model in offline_models_list['models']]
except:
    OFFLINE_MODELS = []

if 'OPENAI_API_KEY' in os.environ:
    OPENAI_MODELS.extend([
        "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"
    ])
if 'GROQ_API_KEY' in os.environ:
    GROQ_MODELS.extend([
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen3-32b"
    ])

LLM_MODELS.extend(GROQ_MODELS)
LLM_MODELS.extend(OPENAI_MODELS)
#LLM_MODELS.extend(OFFLINE_MODELS)

# ─── External Tool Paths ───
TESSERACT_PATH = os.getenv('TESSERACT_PATH', r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
POPPLER_PATH = os.getenv('POPPLER_PATH', r"C:\\poppler-24.08.0\\Library\\bin")

# ─── Qdrant Client Setup ───
required_env_vars = ['qdrant_link', 'qdrant_api']
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

QDRANT_URL = os.environ.get('qdrant_link')
QDRANT_API = os.environ.get('qdrant_api')

client = QdrantClient(
    url=os.environ['qdrant_link'],
    api_key=os.environ['qdrant_api']
)

collections = client.get_collections().collections

if not collections:
    client.create_collection(
        collection_name="my_dataset_v1",
        vectors_config=VectorParams(
            size=768,  # Adjust according to your embedding size
            distance=Distance.COSINE
        )
    )
    collections = client.get_collections().collections

# ───── Dataset Names ─────
DATASET_NAMES = [col.name for col in collections]
