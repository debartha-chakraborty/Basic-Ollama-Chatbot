import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydantic_settings import BaseSettings  # Import from pydantic_settings instead of pydantic
from functools import lru_cache

class Settings(BaseSettings):
    # FastAPI settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug_mode: bool = True
    # Ollama settings
    ollama_host: str = "0.0.0.0"  # Docker service name
    ollama_port: int = 11434
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()