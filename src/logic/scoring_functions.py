from typing import Any, Dict
from src.logic.registry import ScoringRegistry

# Common synonyms for tag matching
SYNONYM_MAP = {
    "女同性戀": ["GL", "百合", "女同", "Lesbian"],
    "男同性戀": ["BL", "耽美", "男同", "Gay"],
    "情慾": ["H", "R18", "色情", "肉", "高H"],
    "肉文": ["H", "R18", "情慾", "高H"],
    "18禁": ["H", "R18", "情慾", "高H"],
}

@ScoringRegistry.register("keyword_match")
def score_keyword_match(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Checks if a keyword exists in a specific field.
    Params:
    - field: str (e.g., 'tags', 'classification', 'name')
    - keyword: str
    """
    field = params.get("field")
    keyword = params.get("keyword", "").lower()
    
    if not field or field not in item:
        return 0.0
    
    val = item[field]
    
    # Expand keyword with synonyms
    keywords_to_check = [keyword]
    if keyword in SYNONYM_MAP:
        keywords_to_check.extend([s.lower() for s in SYNONYM_MAP[keyword]])
    
    # Handle List (e.g. tags)
    if isinstance(val, list):
        # val is likely list of strings due to our DB flattening
        # check if ANY keyword matches ANY tag
        for v in val:
            v_str = str(v).lower()
            if not v_str: continue
            for k in keywords_to_check:
                if k in v_str:
                    return 1.0
        return 0.0
        
    text = str(val).lower()
    for k in keywords_to_check:
        if k in text:
            return 1.0
    return 0.0

@ScoringRegistry.register("numeric_range")
def score_numeric_range(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Checks if a value is within a range.
    Params:
    - field: str (e.g., 'words_total', 'click_count')
    - min_val: float
    - max_val: float
    """
    field = params.get("field")
    if not field or field not in item:
        return 0.0
        
    try:
        val = float(item[field])
    except (ValueError, TypeError):
        return 0.0
        
    min_val = params.get("min_val")
    max_val = params.get("max_val")
    
    if min_val is not None and val < min_val:
        return 0.0
    if max_val is not None and val > max_val:
        return 0.0
        
    return 1.0

@ScoringRegistry.register("status_check")
def score_status(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Checks novel status (completed/ongoing).
    Params:
    - target_status: str ('completed' or 'ongoing')
    """
    target = params.get("target_status", "").lower()
    current = str(item.get("publish_status", "")).lower()
    
    if not target:
        return 1.0 # No preference
        
    return 1.0 if target == current else 0.0

@ScoringRegistry.register("author_match")
def score_author_match(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Checks if the author matches.
    Params:
    - author_name: str
    """
    target_author = params.get("author_name", "").lower()
    item_author = str(item.get("author", "")).lower()
    item_nickname = str(item.get("author_nickname", "")).lower()
    
    if target_author in item_author or target_author in item_nickname:
        return 1.0
    return 0.0

@ScoringRegistry.register("is_free_check")
def score_is_free(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Checks if the book is free.
    Params:
    - require_free: bool
    """
    require_free = params.get("require_free", True)
    is_free = bool(item.get("is_free", False))
    
    if require_free and not is_free:
        return 0.0
    return 1.0

@ScoringRegistry.register("age_check")
def score_age_check(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Penalizes restricted content if user is underage or generally filters.
    Params:
    - allow_restricted: bool
    """
    allow_restricted = params.get("allow_restricted", False)
    restricted_age = item.get("restricted_age", 0)
    
    if not allow_restricted and restricted_age > 0:
        return 0.0
    return 1.0

@ScoringRegistry.register("audio_available")
def score_audio_available(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Checks if TTS/Audio is available.
    Params:
    - require_audio: bool
    """
    require_audio = params.get("require_audio", True)
    has_tts = bool(item.get("tts", False))
    
    if require_audio and not has_tts:
        return 0.0
    return 1.0

# --- Stage 5: Numeric Ranking (Soft Scoring) ---
import math

@ScoringRegistry.register("numeric_ranking")
def score_numeric_ranking(item: Dict[str, Any], params: Dict[str, Any]) -> float:
    """
    Soft scoring for numeric fields using sigmoid normalization.
    Higher/lower values get a proportionally higher score instead of a hard pass/fail.
    
    Params:
    - field: str (e.g., 'words_total')
    - ranking_direction: 'asc' (higher is better) or 'desc' (lower is better)
    - normalize_max: float (normalization ceiling, e.g., 2000000 for words)
    
    Returns a score between 0.0 and 1.0.
    """
    field = params.get("field")
    if not field or field not in item:
        return 0.5  # Neutral score if field missing
    
    try:
        val = float(item[field])
    except (ValueError, TypeError):
        return 0.5
    
    direction = params.get("ranking_direction", "asc")
    normalize_max = params.get("normalize_max", 1000000)
    
    # Normalize to 0-1 range using sigmoid-like function
    # For 'asc': higher values -> higher score
    # For 'desc': lower values -> higher score
    
    # Shift the sigmoid center to normalize_max / 2
    midpoint = normalize_max / 2
    steepness = 6.0 / normalize_max  # Controls the curve steepness
    
    if direction == "asc":
        # Sigmoid: 1 / (1 + exp(-steepness * (val - midpoint)))
        score = 1.0 / (1.0 + math.exp(-steepness * (val - midpoint)))
    else:  # desc
        # Inverted sigmoid
        score = 1.0 / (1.0 + math.exp(steepness * (val - midpoint)))
    
    return score
