from typing import List, Optional, Any, Dict
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

class QueryParseResult(BaseModel):
    """
    The structured result of parsing a user's natural language query.
    """
    original_query: str
    criteria: List[ScoringCriteria]
    search_terms: List[str] = Field(default_factory=list, description="Keywords/phrases to use for initial candidate retrieval")
    generated_keywords: List[str] = Field(default_factory=list, description="LLM-generated domain-specific keywords for dynamic query expansion")
    reference_books: List[str] = Field(default_factory=list, description="Book titles mentioned or referenced in the user query")
    # HyDE 核心：讓 LLM 寫一段「夢想中的書籍簡介」
    hypothetical_intro: str = Field(default="", description="A generated hypothetical novel introduction based on the query.")

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
