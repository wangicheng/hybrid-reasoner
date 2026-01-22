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
    require_free: Optional[bool] = Field(None, description="Whether to require free books")
    allow_restricted: Optional[bool] = Field(None, description="Whether to allow restricted content")
    require_audio: Optional[bool] = Field(None, description="Whether to require audio")
    # --- Stage 5: Numeric Ranking ---
    ranking_direction: Optional[str] = Field(None, description="For numeric_ranking: 'asc' (higher is better) or 'desc' (lower is better)")
    normalize_max: Optional[float] = Field(None, description="For numeric_ranking: the normalization ceiling (e.g. 2000000 for words)")

class ScoringCriteria(BaseModel):
    """
    Represents a single scoring factor derived from the user query.
    """
    name: str = Field(..., description="Name of the scoring function to apply (e.g., 'keyword_match', 'numeric_range')")
    weight: float = Field(..., description="Weight of this criteria (0.0 to 1.0).")
    parameters: ScoringParameters = Field(default_factory=ScoringParameters, description="Parameters to pass to the scoring function")
    description: Optional[str] = Field(None, description="Explanation of why this criteria was chosen")

class QueryParseResult(BaseModel):
    """
    The structured result of parsing a user's natural language query.
    """
    original_query: str
    criteria: List[ScoringCriteria]
    search_terms: List[str] = Field(default_factory=list, description="Keywords/phrases to use for initial candidate retrieval")

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
