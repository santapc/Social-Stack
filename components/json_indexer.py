"""
JsonIndexer: Indexes all JSON files in a directory into a Qdrant vectorstore.
- page_content contains a readable summary, not raw JSON.
- metadata contains only selected summary fields, with type/category fixed and validated.

Usage:
    from components.json_indexer import JsonIndexer
    indexer = JsonIndexer(collection_name="my_collection", embeddings=my_embeddings)
    indexer.process_json_files("scheme_data/cache")
    indexer.create_vectorstore()
"""
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing import List, Dict, Optional
from tqdm import tqdm
import os
import json
from dotenv import load_dotenv
import warnings
import pickle

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

qdrant_link = os.environ.get("qdrant_link")
qdrant_api = os.environ.get("qdrant_api")

def load_bin_list(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return []

STATE_NAMES = load_bin_list("assets/state_names.bin")
MINISTRY_NAMES = load_bin_list("assets/ministry_names.bin")

def normalize_name(name: str) -> str:
    """Normalize a name for comparison: lowercase, replace '&' with 'and', strip, remove extra spaces and punctuation."""
    import re
    if not isinstance(name, str):
        return ""
    name = name.lower().replace("&", "and")
    name = re.sub(r'[^a-z0-9 ]+', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    return name

def find_best_match(name: str, options: list) -> str:
    """Find the best match for name in options using normalization and fuzzy matching."""
    from difflib import get_close_matches
    norm_name = normalize_name(name)
    # Normalize all options (from bin file) the same way
    norm_options = {normalize_name(opt): opt for opt in options}
    if norm_name in norm_options:
        return norm_options[norm_name]
    close = get_close_matches(norm_name, norm_options.keys(), n=1, cutoff=0.8)
    if close:
        return norm_options[close[0]]
    for norm_opt, orig_opt in norm_options.items():
        if norm_name in norm_opt or norm_opt in norm_name:
            return orig_opt
    return ""

def extract_category_and_type(data: dict) -> (str, str):
    """
    For 'central', category is nodal_department.
    For 'state', category is the first part of 'level' split by '/'.
    Returns (type_val, category_val)
    """
    level = data.get("level", "").lower()
    nodal_department = data.get("nodal_department", "")
    if "state" in level:
        type_val = "State"
        category_val = data.get("state", "")
    else:
        type_val = "Central"
        category_val = nodal_department
    return type_val, category_val

class JsonIndexer:
    def __init__(self, collection_name: str, embeddings):
        self.qdrant_url = qdrant_link
        self.qdrant_api_key = qdrant_api
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.qdrant_client = self.initialize_qdrant_client()
        self.vectorstore = self.connect_vectorstore()
        self.documents = []

    def initialize_qdrant_client(self) -> QdrantClient:
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_LINK and QDRANT_API must be set in environment variables.")
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=30)

    def connect_vectorstore(self):
        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    def process_json_files(self, folder_path: str) -> List[Document]:
        self.documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(data, list) and data:
                        data = data[0]
                    elif not isinstance(data, dict):
                        continue

                # --- Build readable text summary with all keys ---
                summary_lines = []
                for key, value in data.items():
                    # Format lists and dicts as readable text
                    if isinstance(value, list):
                        if value and isinstance(value[0], dict):
                            value_str = "\n    " + "\n    ".join(
                                [", ".join(f"{k}: {v}" for k, v in item.items()) for item in value]
                            )
                        else:
                            value_str = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
                    else:
                        value_str = str(value)
                    summary_lines.append(f"{key.replace('_', ' ').capitalize()}: {value_str}")
                page_content = "\n".join(summary_lines)

                # --- Fix metadata fields using new logic ---
                type_val, category_raw = extract_category_and_type(data)
                if type_val.lower() == "state":
                    category_val = find_best_match(category_raw, STATE_NAMES)
                else:
                    category_val = find_best_match(category_raw, MINISTRY_NAMES)
                if not category_val:
                    category_val = category_raw

                scheme_id = data.get("slug", "")

                metadata = {
                    "scheme_id": scheme_id,
                    "name": data.get("scheme_name", ""),
                    "type": type_val,
                    "category": category_val,
                    "source_url": data.get("source_url", f"https://www.myscheme.gov.in/schemes/{scheme_id}")
                }

                doc = Document(page_content=page_content, metadata=metadata)
                self.documents.append(doc)

        return self.documents

    def ensure_collection_exists(self, vector_size: int):
        from qdrant_client.http import models as rest
        collections = self.qdrant_client.get_collections().collections
        if self.collection_name not in [c.name for c in collections]:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=vector_size,
                    distance=rest.Distance.COSINE,
                ),
            )

    def create_vectorstore(self, batch_size: int = 50):
        if not self.documents:
            raise ValueError("No documents to index. Run process_json_files first.")

        # Ensure the collection exists before uploading
        try:
            vector_size = self.embeddings.client.model.get_sentence_embedding_dimension()
        except AttributeError:
            vector_size = len(self.embeddings.embed_query("test"))
        self.ensure_collection_exists(vector_size)

        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

        for i in tqdm(range(0, len(self.documents), batch_size), desc="Uploading to Qdrant"):
            batch = self.documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)

        return self.vectorstore

    def insert_into_vectorstore(self, new_documents: List[Document]):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run create_vectorstore first.")
        existing_ids = set()
        scroll_filter = None
        offset = None
        while True:
            scroll_result, next_offset = self.vectorstore.scroll(
                collection_name=self.vectorstore.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                with_payload=True,
                offset=offset
            )
            for record in scroll_result:
                meta = record.payload.get('metadata', {})
                scheme_id = meta.get('scheme_id')
                if scheme_id:
                    existing_ids.add(scheme_id)
            if not next_offset:
                break
            offset = next_offset

        filtered_documents = []
        for doc in new_documents:
            scheme_id = doc.metadata.get('scheme_id')
            if scheme_id and scheme_id not in existing_ids:
                filtered_documents.append(doc)
                existing_ids.add(scheme_id)

        if filtered_documents:
            for i in range(0, len(filtered_documents), 50):
                self.vectorstore.add_documents(filtered_documents[i:i + 50])
            self.documents.extend(filtered_documents)

        return self.vectorstore

    def delete_vectorstore(self):
        if self.qdrant_client:
            self.qdrant_client.delete_collection(self.collection_name)
            self.vectorstore = None
            self.documents = []
        return True

    def write_documents_to_json(self, output_path: str):
        """
        Write all indexed documents to a JSON file.
        Each document will be represented as a dict with 'page_content' and 'metadata'.
        """
        export_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in self.documents
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(export_data)} documents to {output_path}")


if __name__ == "__main__":
    indexer = JsonIndexer(
        collection_name="mosje_schemes",
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    indexer.process_json_files("scheme_data/cache")
    indexer.create_vectorstore()
    # indexer.delete_vectorstore()
    # Example usage for writing to JSON:
    # indexer.write_documents_to_json("exported_schemes.json")