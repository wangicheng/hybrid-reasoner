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
    # Set to 0.4 / 0.6 based on 3-phase sweep optimization (best balance of semantic relevance & strict constraints)
    SEMANTIC_WEIGHT: float = 0.4
    ATTRIBUTE_WEIGHT: float = 0.6
    
    # BM25 recall & fusion settings
    ENABLE_BM25: bool = True
    BM25_FUSION_MODE: str = "log_dampened"  # Options: multiplicative, additive, log_dampened, tiebreaker
    BM25_BONUS_MAX: float = 0.005           # Optimal alpha/epsilon found in fusion sweep for micro-boosting
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    TOP_K_BM25: int = 1000  # Default to fetching many to merge
    BM25_WEIGHT: float = 0.3  # Legacy setting retained for compatibility

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
