import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_embeddings():
    """
    Retrieves HuggingFace embeddings model, cached for performance.

    Returns:
        HuggingFaceEmbeddings: The embeddings model instance.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
