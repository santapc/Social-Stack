from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from typing import List, Dict, Tuple,Optional
import os
import uuid
import json
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

QDRANT_LINK = os.environ.get("QDRANT_LINK")
QDRANT_API = os.environ.get("QDRANT_API")

class SimpleIndexer:
    def __init__(self, collection_name, embeddings):
        self.qdrant_url = QDRANT_LINK
        self.qdrant_api_key = qdrant_api
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.qdrant_client = self.initialize_qdrant_client()
        self.vectorstore = self.connect_vectorstore()
        self.bm25 = None
        self.documents = []
        
    def initialize_qdrant_client(self) -> QdrantClient:
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_LINK and QDRANT_API must be set in environment variables.")
        return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key,timeout=30)
    
    def connect_vectorstore(self):
        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

    # def process_text_files(self, folder_path):
    #     self.documents = []
    #     for filename in tqdm(os.listdir(folder_path), desc="Processing files", unit="file"):
    #         if filename.endswith(".txt"):
    #             file_path = os.path.join(folder_path, filename)
    #             loader = TextLoader(file_path)
    #             doc = loader.load()[0]
    #             doc.metadata["scheme_id"] = os.path.splitext(filename)[0]
    #             doc.metadata["scheme_name"] = doc.page_content.split('\n')[0]
    #             self.documents.append(doc)
    #     return self.documents
    
    def process_json_files(self, folder_path):
        self.documents = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        data = data[0]
                    else:
                        continue  # skip if not a non-empty list

                metadata = {
                    "scheme_id": os.path.splitext(filename)[0],
                    "name": data.get("name", ""),
                    "type": data.get("type", ""),
                    "category": data.get("category", ""),
                    "source_url": data.get("source_url", f"https://www.myscheme.gov.in/schemes/{os.path.splitext(filename)[0]}")
                }

                content_fields = {k: v for k, v in data.items() if not k == "scheme_id"}
                page_content = "\n".join(v for k, v in content_fields.items())

                doc = Document(page_content=page_content, metadata=metadata)
                self.documents.append(doc)

        return self.documents

    def create_vectorstore(self):
        if not self.documents:
            raise ValueError("No documents to index. Run process_text_files first.")
        self.vectorstore = Qdrant.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    def insert_into_vectorstore(self, new_documents):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run create_vectorstore first.")
        
        existing_scheme_ids = set()
        
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
                scheme_id = record.payload.get('metadata', {}).get('scheme_id')
                if scheme_id:
                    existing_scheme_ids.add(scheme_id)
            
            
            if not next_offset:
                break
            offset = next_offset

        
        filtered_documents = []
        for doc in new_documents:
            scheme_id = doc.metadata.get('scheme_id')
            if scheme_id and scheme_id not in existing_scheme_ids:
                filtered_documents.append(doc)
                existing_scheme_ids.add(scheme_id)
        
        
        if filtered_documents:
            self.vectorstore.add_documents(filtered_documents)
            self.documents.extend(filtered_documents)
        
        return self.vectorstore

    def delete_vectorstore(self):
        if self.qdrant_client:
            self.qdrant_client.delete_collection(self.collection_name)
            self.vectorstore = None
            self.documents = []
        return True

    def create_scheme_bm25(self, save_path: str, documents: List[Document]) -> BM25Retriever:
        self.bm25 = BM25Retriever.from_documents(documents=documents)
        try:
            with open(save_path, "wb") as f:
                pickle.dump(self.bm25, f)
        except OSError as e:
            raise
        return self.bm25

    def load_bm25(self, file_path: str):
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.bm25 = pickle.load(f)
            return self.bm25
        raise FileNotFoundError("BM25 files not found. Run create_bm25 first.")
