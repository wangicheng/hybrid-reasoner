import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file
load_dotenv()

# Define Project Root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_BASE_URL: str = "https://opencode.ai/zen/v1"
    LLM_MODEL_ID: str = "gemma-3-27b-it"
    
    # Use absolute paths based on PROJECT_ROOT
    QDRANT_PATH: str = str(PROJECT_ROOT / "data" / "qdrant_storage")
    DB_PATH: str = str(PROJECT_ROOT / "hybrid_reasoner.db")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
