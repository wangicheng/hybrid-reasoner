import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

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
    SEMANTIC_WEIGHT: float = 0.3
    ATTRIBUTE_WEIGHT: float = 0.7
    
    # BM25 Experimental settings
    ENABLE_BM25: bool = False
    BM25_WEIGHT: float = 0.3
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    TOP_K_BM25: int = 1000  # Default to fetching many to merge

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

def _parse_google_api_keys() -> List[str]:
    """Parse Google API keys from either GOOGLE_API_KEYS or GOOGLE_API_KEY."""
    raw_multi = os.environ.get("GOOGLE_API_KEYS", "").strip()
    raw_single = os.environ.get("GOOGLE_API_KEY", "").strip()
    raw_value = raw_multi or raw_single
    if not raw_value:
        return []

    return [key.strip() for key in raw_value.split(",") if key.strip()]


settings.GOOGLE_API_KEYS = _parse_google_api_keys()
