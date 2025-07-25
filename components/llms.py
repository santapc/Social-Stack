from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

HPC_URL = os.environ.get('HPC_URL')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def get_hpc_llm(model: str = "llama3.2:3b", temperature: int = 1, top_p: int = 1, timeout: int = 120, headers: dict = {}):
    llm = ChatOllama(
        base_url=HPC_URL,
        model=model,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        headers=headers, 
    )

    return llm

def get_hpc_llm_openai(model: str = "gpt-4o-mini", temperature: float = 1.0, top_p: float = 1.0, timeout: int = 120, headers: dict = {}):
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        default_headers=headers
    )
