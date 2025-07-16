import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import re 


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

class EnhancedVectorStore(VectorStore):
    """Enhanced vector store with improved semantic search capabilities"""
    
    def __init__(self, persist_directory="data/embeddings", collection_name="contracts"):
        super().__init__(persist_directory, collection_name)
        
        # Legal term mappings for query enhancement
        self.legal_synonyms = {
            "termination": ["terminate", "end", "dissolution", "expiry", "cancellation"],
            "compensation": ["salary", "payment", "remuneration", "wages"],
            "confidentiality": ["non-disclosure", "proprietary", "confidential", "trade secret"],
            "benefits": ["perks", "entitlements", "allowances", "insurance"]
        }

    def preprocess_query(self, query: str) -> str:
        """Enhance query with legal synonyms and context"""
        query_lower = query.lower()
        enhanced_terms = []
        
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower:
                enhanced_terms.extend(synonyms)
        
        if enhanced_terms:
            return f"{query} {' '.join(enhanced_terms)}"
        return query

    def create_smart_chunks(self, text: str) -> List[Dict]:
        """Create semantically meaningful chunks based on contract structure"""
        chunks = []
        
        # Split by numbered sections first
        sections = re.split(r'\n\s*\d+\.\s*', text)
        
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            # Extract section title
            lines = section.strip().split('\n')
            title = lines[0] if lines else f"Section {i}"
            
            # Create chunk with metadata
            chunk_data = {
                "text": section.strip(),
                "section_title": title,
                "section_number": i,
                "word_count": len(section.split())
            }
            chunks.append(chunk_data)
        
        return chunks

    def add_documents_enhanced(self, text: str, metadata: Dict = None):
        """Add documents with enhanced chunking and metadata"""
        smart_chunks = self.create_smart_chunks(text)
        
        # Extract just the text for the existing add_documents method
        chunk_texts = [chunk["text"] for chunk in smart_chunks]
        
        # Use existing add_documents method
        self.add_documents(chunk_texts)

    def search_similar_enhanced(self, query: str, k: int = 3) -> List[str]:
        """Enhanced search with query preprocessing"""
        # Preprocess query
        enhanced_query = self.preprocess_query(query)
        
        # Use existing search method
        return self.search_similar(enhanced_query, k=k)