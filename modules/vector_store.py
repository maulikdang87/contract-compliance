import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, persist_directory="data/embeddings", collection_name="contracts"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, chunks: List[str], metadatas: Optional[List[Dict]] = None):
        embeddings = self.embedder.encode(chunks)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # FIXED: Don't pass empty metadata dictionaries
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas  # Pass None instead of empty dicts
        )

    def search_similar(self, query: str, k: int = 3):
        try:
            query_embedding = self.embedder.encode([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Search error: {e}")
            return []
