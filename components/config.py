"""
Central configuration for Digital-MoSJE project.
"""

# Model names

groq_models=["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct", "deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"]
gpt_models=["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
offline_models=["gemma3:27b","qwq:32b","mistral-small3.1:24b","deepseek-r1:32b","deepseek-r1:8b"]
LLM_MODELS = groq_models + gpt_models + offline_models

# Dataset names
DATASET_NAMES = [
    'indexing_endpoint_test_v1',
    'all_mosje_simple_v2',
    'all_myschemes_simple_v2',
    'all_myschemes_simple_v1'
]

# External tool paths (override with env vars if set)
import os
TESSERACT_PATH = os.getenv('TESSERACT_PATH', r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
POPPLER_PATH = os.getenv('POPPLER_PATH', r"C:\\poppler-24.08.0\\Library\\bin") 