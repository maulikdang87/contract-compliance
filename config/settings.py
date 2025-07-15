import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Document Processing
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt']
    
    # Paths
    DATA_DIR = "data"
    CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
    RULES_DIR = os.path.join(DATA_DIR, "rules")
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
    
    # Agent Configuration
    MODEL_NAME = "gemini-pro"
    TEMPERATURE = 0.1
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.CONTRACTS_DIR, cls.RULES_DIR, cls.EMBEDDINGS_DIR]:
            os.makedirs(directory, exist_ok=True)

settings = Settings()
