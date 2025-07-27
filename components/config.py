"""
Central configuration for Digital-MoSJE project.
"""

import ollama
import os

LLM_MODELS = []

GROQ_MODELS = []
OPENAI_MODELS = []

offline_models_list = ollama.list()
OFFLINE_MODELS = [model['model'] for model in offline_models_list['models']]

if 'OPENAI_API_KEY' in os.environ:
    OPENAI_MODELS.extend(["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"])
if 'GROQ_API_KEY' in os.environ:
    GROQ_MODELS.extend(["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"])

LLM_MODELS.extend(GROQ_MODELS)
LLM_MODELS.extend(OPENAI_MODELS)
LLM_MODELS.extend(OFFLINE_MODELS)

# Dataset names
DATASET_NAMES = [
    'indexing_endpoint_test_v1',
    'all_mosje_simple_v2',
    'all_myschemes_simple_v2',
    'all_myschemes_simple_v1'
]

# External tool paths (override with env vars if set)
TESSERACT_PATH = os.getenv('TESSERACT_PATH', r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
POPPLER_PATH = os.getenv('POPPLER_PATH', r"C:\\poppler-24.08.0\\Library\\bin") 