from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field

class ScoringParameters(BaseModel):
    field: Optional[str] = Field(None, description="Field to specific match on (e.g. tags, classification)")
    keyword: Optional[str] = Field(None, description="Keyword to search for")
    min_val: Optional[float] = Field(None, description="Minimum value for numeric range")
    max_val: Optional[float] = Field(None, description="Maximum value for numeric range")
    target_status: Optional[str] = Field(None, description="Target status (completed or ongoing)")
    query_text: Optional[str] = Field(None, description="Text for semantic similarity")
    author_name: Optional[str] = Field(None, description="Name of the author")
    # --- Stage 5: Numeric Ranking ---
    ranking_direction: Optional[str] = Field(None, description="For numeric_ranking: 'asc' (higher is better) or 'desc' (lower is better)")
    normalize_max: Optional[float] = Field(None, description="For numeric_ranking: the normalization ceiling (e.g. 2000000 for words)")

class ScoringCriteria(BaseModel):
    """
    Represents a single scoring factor derived from the user query.
    """
    name: str = Field(..., description="Name of the scoring function to apply (e.g., 'keyword_match', 'numeric_range')")
    is_negative: bool = Field(default=False, description="Set to true if this is a negative requirement (e.g., 'do not want', 'avoid').")
    parameters: ScoringParameters = Field(default_factory=ScoringParameters, description="Parameters to pass to the scoring function")
    description: Optional[str] = Field(None, description="Explanation of why this criteria was chosen")

class NegationConstraint(BaseModel):
    """表示一個負向約束條件（排除或降權）。"""
    term: str = Field(..., description="要排除或降權的詞彙/概念")
    strength: Literal["hard", "soft"] = Field(
        default="hard",
        description="'hard' = 絕對排除 (AND NOT)；'soft' = 柔性降權 (negative boost)"
    )
    weight: float = Field(
        default=-1.0,
        description="負權重係數。hard 排除時為 -1.0；soft 降權時為 -0.3 ~ -0.8"
    )
    reason: str = Field(default="", description="LLM 標注的排除原因")


class QueryIntent(BaseModel):
    """
    LLM 結構化意圖拆解結果。
    將使用者查詢拆分為正向搜尋詞 + 負向約束，
    避免否定詞直接進入 BM25 計分公式。
    """
    positive_terms: List[str] = Field(
        default_factory=list,
        description="淨化後的正向搜尋關鍵詞，僅包含使用者真正想要的概念"
    )
    hard_exclusions: List[NegationConstraint] = Field(
        default_factory=list,
        description="絕對排除項 (AND NOT)，命中即過濾"
    )
    soft_exclusions: List[NegationConstraint] = Field(
        default_factory=list,
        description="柔性排除項 (negative boost)，命中則扣分但不絕對過濾"
    )
    sanitized_bm25_query: str = Field(
        default="",
        description="淨化後僅含正向詞的 BM25 查詢字串"
    )
    boolean_query: str = Field(
        default="",
        description="完整布林邏輯表達式，例：(魔法學校) AND NOT (悲劇)"
    )


class QueryParseResult(BaseModel):
    """
    The structured result of parsing a user's natural language query.
    """
    original_query: str
    criteria: List[ScoringCriteria]
    search_terms: str = Field(default="", description="Keywords/phrases to use for initial candidate retrieval")
    generated_keywords: List[str] = Field(default_factory=list, description="LLM-generated domain-specific keywords for dynamic query expansion")
    extracted_tags: List[str] = Field(
        default_factory=list,
        description="Canonical tags selected by LLM from available taxonomy tags",
    )
    reference_books: List[str] = Field(default_factory=list, description="Book titles mentioned or referenced in the user query")
    # HyDE 核心：讓 LLM 寫一段「夢想中的書籍簡介」
    hypothetical_intro: str = Field(default="", description="A generated hypothetical novel introduction based on the query.")
    # 階段一：Pre-Retrieval 意圖解析
    query_intent: Optional[QueryIntent] = Field(
        default=None,
        description="LLM 結構化意圖拆解結果，包含正向詞、硬排除、軟排除"
    )

class NovelItem(BaseModel):
    id: str
    name: str
    intro: str
    words_total: int
    publish_status: str
    click_count: int
    bookmark_count: int
    tags: List[str]
    classification: str
    rating_score: Optional[float] = None
    rating_count: int = 0
    total_recommendations: int = 0
    url: Optional[str] = None
    cover_url: Optional[str] = None
    illname: Optional[str] = None
    backupname: Optional[str] = None
    is_animated: bool = False
