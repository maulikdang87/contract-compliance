from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import generativeai as genai
from google.generativeai.types import EmbedContentConfig
from typing import List
import numpy as np

class GeminiEmbeddingSearchAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
    def create_enhanced_embeddings(self, texts: List[str], task_type: str = "SEMANTIC_SIMILARITY"):
        """Create task-specific embeddings using Gemini"""
        result = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type=task_type,
        )
        return [np.array(e.values) for e in result.embeddings]

    def _enhance_legal_query(self, query: str):
        # You can add legal keyword expansion here if needed
        return query
    
    def _calculate_weighted_similarity(self, query_embedding, doc_embeddings):
        from numpy.linalg import norm
        return [np.dot(query_embedding, d) / (norm(query_embedding) * norm(d)) for d in doc_embeddings]

    def _rank_results(self, documents, similarities):
        return sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
    
    def intelligent_search(self, query: str, contract_sections: List[str]):
        enhanced_query = self._enhance_legal_query(query)
        query_embedding = self.create_enhanced_embeddings([enhanced_query], "RETRIEVAL_QUERY")[0]
        doc_embeddings = self.create_enhanced_embeddings(contract_sections, "RETRIEVAL_DOCUMENT")
        similarities = self._calculate_weighted_similarity(query_embedding, doc_embeddings)
        return self._rank_results(contract_sections, similarities)
