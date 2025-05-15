import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydantic_settings import BaseSettings  # Import from pydantic_settings instead of pydantic
from functools import lru_cache

class Settings(BaseSettings):

    api_title: str = "Agentic RAG API"
    api_description: str = "API for an Agentic RAG system using LangChain tools"
    api_version: str = "0.1.0"
    # FastAPI settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug_mode: bool = True
    # Gemini API settings
    gemini_api_key: str = ""  # Set this in your environment or .env file
    # ChromaDB settings
    chromadb_path: str = "./chroma_db"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
