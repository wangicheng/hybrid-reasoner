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

    # Reranker (PermSC Listwise Reranking)
    RERANK_ENABLED: bool = True
    RERANK_MODEL_ID: str = "gemma-4-31b-it"
    RERANK_CANDIDATE_LIMIT: int = 100
    RERANK_PERMUTATIONS: int = 3

    # DAT (Dynamic Alpha Tuning) + 3+1 Layer Scoring Pipeline
    ENABLE_DAT: bool = False                # Feature flag — DAT 預設關閉
    DAT_DEFAULT_ALPHA: float = 0.5          # Plot vs BM25 平衡預設 α
    DAT_TIMEOUT_MS: int = 2000              # LLM 超時門檻 (ms)
    DAT_SHORT_QUERY_THRESHOLD: int = 2      # 短 Query 早退門檻 (字元數)
    DAT_SHORT_QUERY_ALPHA: float = 0.3      # 短 Query 的固定 α (偏向 BM25)
    DAT_MODEL_ID: str = ""                  # DAT LLM model，空字串 = 使用預設

    # Layer 2: Tag Vector Bonus
    TAG_BONUS_BETA: float = 0.20            # 標籤乘數 β 基準值，tag-heavy 查詢自動提升
    TAG_TAU: float = 0.25                   # Layer 2 標籤向量溫度（比 L1 的 0.1 平緩，避免斷崖效應）

    # Layer 3: Required Tag Boost
    REQUIRED_TAG_BOOST: float = 10.0        # 必備標籤霸榜提權常數 (legacy, kept for reference)
    REQUIRED_TAG_MULTIPLIER: float = 3.0    # L3 乘法提權倍率（取代 +10 加法，保留排名信號）

    # Layer 0: Violation Penalty Multipliers (連乘積)
    PENALTY_BLOCKED_TAGS: float = 0.05      # 命中排斥標籤 → 死罪 (從 0.1 壓低，加強雷區隔離)
    PENALTY_REQUIRED_STATUS: float = 0.5    # 狀態不符 → 中罪 (維持 0.5 高摩擦力)
    PENALTY_REQUIRED_TAGS: float = 0.85     # 必備標籤缺失 → 輕罪 (從 0.8 放寬，對齊語意免疫機制)

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
