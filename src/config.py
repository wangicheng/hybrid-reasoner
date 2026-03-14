import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import List

# Load .env file
load_dotenv()

# Define Project Root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-placeholder"
    OPENAI_BASE_URL: str = "https://opencode.ai/zen/v1"
    GOOGLE_API_KEY: str = ""
    GOOGLE_API_KEYS: List[str] = []

    # Use absolute paths based on PROJECT_ROOT
    QDRANT_PATH: str = str(PROJECT_ROOT / "data" / "qdrant_storage")
    DB_PATH: str = str(PROJECT_ROOT / "data" / "hybrid_reasoner.db")
    
    # Dual-Track Fusion Weights (語意-屬性雙軌融合權重)
    # w1 = SEMANTIC_WEIGHT, w2 = ATTRIBUTE_WEIGHT
    FUSION_MODE: str = "multiplicative" # Options: "multiplicative", "additive"
    SEMANTIC_WEIGHT: float = 0.5
    ATTRIBUTE_WEIGHT: float = 0.5

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Parse multiple Google API keys from environment variable
_google_api_key_raw = os.environ.get("GOOGLE_API_KEY", "").strip()
if _google_api_key_raw:
    # Split by comma and clean whitespace
    settings.GOOGLE_API_KEYS = [key.strip() for key in _google_api_key_raw.split(",") if key.strip()]
else:
    settings.GOOGLE_API_KEYS = []
