from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# --- 推薦版本：彈性高，適合快速迭代 ---

class Criterion(BaseModel):
    """
    Represents a single scoring factor.
    """
    name: str  # e.g., "numeric_range", "status_check", "keyword_match"
    weight: float = 1.0
    # 使用 Dict 保持最大彈性，不限制具體欄位名稱
    parameters: Dict[str, Any] = Field(default_factory=dict) 
    description: Optional[str] = None

class QueryParseResult(BaseModel):
    """
    The structured result of parsing a user's natural language query.
    """
    original_query: str
    search_terms: List[str] = Field(default_factory=list)
    criteria: List[Criterion]

# --- 書籍資料結構 (不變) ---
class NovelItem(BaseModel):
    id: str
    name: str
    intro: str
    words_total: int
    chapters_total: int
    publish_status: str
    click_count: int
    tags: List[str]
    classification: str
    url: Optional[str] = None