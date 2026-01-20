import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_BASE_URL: str = "https://opencode.ai/zen/v1"
    LLM_MODEL_ID: str = "big-pickle"
    QDRANT_PATH: str = "data/qdrant_storage"
    DB_PATH: str = "hybrid_reasoner.db"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
