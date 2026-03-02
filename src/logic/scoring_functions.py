from typing import Any, Dict, Tuple
from src.logic.registry import ScoringRegistry
import math

# Common synonyms for tag matching
SYNONYM_MAP = {
    "女同性戀": ["GL", "百合", "女同", "Lesbian"],
    "男同性戀": ["BL", "耽美", "男同", "Gay"],
    "情慾": ["H", "R18", "色情", "肉", "高H"],
    "肉文": ["H", "R18", "情慾", "高H"],
    "18禁": ["H", "R18", "情慾", "高H"],
}

@ScoringRegistry.register("keyword_match")
def score_keyword_match(item: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Checks if a keyword exists in a specific field.
    Returns: (score, reason)
    """
    field = params.get("field")
    keyword = params.get("keyword", "").lower()
    
    if not field or field not in item:
        return 0.0, f"欄位 '{field}' 不存在"
    
    val = item[field]
    
    # Expand keyword with synonyms
    keywords_to_check = [keyword]
    synonym_found = False
    if keyword in SYNONYM_MAP:
        keywords_to_check.extend([s.lower() for s in SYNONYM_MAP[keyword]])
    
    # Helper to format reason
    def format_success(match_term):
        if match_term == keyword:
            return f"符合關鍵字 '{keyword}'"
        return f"符合同義詞 '{match_term}' (搜尋: '{keyword}')"

    # Handle List (e.g. tags)
    if isinstance(val, list):
        for v in val:
            v_str = str(v).lower()
            if not v_str: continue
            for k in keywords_to_check:
                if k in v_str:
                    return 1.0, format_success(k)
        # --- Fallback Logic for list fields ---
        # (No fallback needed here, fallback is below for non-list primary fields)
        return 0.0, f"未找到關鍵字 '{keyword}'"
        
    text = str(val).lower()
    for k in keywords_to_check:
        if k in text:
            return 1.0, format_success(k)
            
    # --- Fallback Logic: Logical Expansion ---
    # If the primary field (e.g., 'classification') didn't match, 
    # check other semantic fields like 'tags', 'name', or 'intro'.
    # This handles cases where data is messy (e.g., Publisher in Classification field).
    
    # 1. Check Tags (if not already checked)
    if field != "tags" and "tags" in item:
        tags = item["tags"]
        if isinstance(tags, list):
            for t in tags:
                t_str = str(t).lower()
                for k in keywords_to_check:
                    if k in t_str:
                        return 0.8, f"從標籤中找到 '{k}' (原搜尋欄位: '{field}')"
        
    # 2. Check Name/Title
    if field != "name" and "name" in item:
        name_str = str(item["name"]).lower()
        for k in keywords_to_check:
            if k in name_str:
                return 0.8, f"從書名中找到 '{k}' (原搜尋欄位: '{field}')"
                
    # 3. Check Intro (if it's a genre/subject keyword, it likely appears in intro)
    if field != "intro" and "intro" in item and item["intro"]:
        intro_str = str(item["intro"]).lower()
        # Only check if keyword is significant length to avoid false positives in long text
        if len(keyword) >= 2: 
            for k in keywords_to_check:
                idx = intro_str.find(k)
                if idx != -1:
                    # 【否定句檢查】檢查關鍵字前面 5 個字有沒有否定詞
                    prefix = intro_str[max(0, idx-5):idx]
                    if any(neg in prefix for neg in ["不是", "非", "沒有", "並非"]):
                        return 0.0, f"簡介中提及 '{k}' 但疑似為否定句"
                    return 0.6, f"從簡介中找到 '{k}' (原搜尋欄位: '{field}')"
    
    return 0.0, f"內容不包含 '{keyword}'"

@ScoringRegistry.register("numeric_range")
def score_numeric_range(item: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Checks if a value is within a range.
    Returns: (score, reason)
    """
    field = params.get("field")
    if not field or field not in item:
        return 0.0, "數據缺失"
        
    try:
        val = float(item[field])
    except (ValueError, TypeError):
        return 0.0, "數據格式錯誤"
        
    min_val = params.get("min_val")
    max_val = params.get("max_val")
    
    if min_val is not None and val < min_val:
        return 0.0, f"數值 {int(val)} 低於下限 {int(min_val)}"
    if max_val is not None and val > max_val:
        return 0.0, f"數值 {int(val)} 高於上限 {int(max_val)}"
        
    return 1.0, f"數值 {int(val)} 符合範圍"

@ScoringRegistry.register("status_check")
def score_status(item: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Checks novel status (completed/ongoing).
    Returns: (score, reason)
    """
    target = params.get("target_status", "").lower()
    current = str(item.get("publish_status", "")).lower()
    
    if not target:
        return 1.0, "無狀態限制"
        
    # Standardize comparison logic
    is_target_completed = target in ["completed", "finished", "已完結", "完結"]
    is_target_ongoing = target in ["ongoing", "serializing", "連載中", "連載"]
    
    is_current_completed = any(x in current for x in ["completed", "finished", "完結"])
    is_current_ongoing = any(x in current for x in ["ongoing", "serializing", "連載"])

    if is_target_completed and is_current_completed:
         return 1.0, f"狀態符合 (已完結)"
    if is_target_ongoing and is_current_ongoing:
         return 1.0, f"狀態符合 (連載中)"
         
    # Fallback to simple matching
    if target == current or target in current or current in target:
        return 1.0, f"狀態符合 ({current})"
        
    return 0.0, f"狀態不符 (需求: {target}, 實際: {current})"

@ScoringRegistry.register("author_match")
def score_author_match(item: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Checks if the author matches.
    Returns: (score, reason)
    """
    target_author = params.get("author_name", "").lower()
    item_author = str(item.get("author", "")).lower()
    
    if target_author in item_author:
        return 1.0, f"作者匹配 ({item.get('author')})"
    return 0.0, f"作者不符"



# --- Stage 5: Numeric Ranking (Soft Scoring) ---

@ScoringRegistry.register("numeric_ranking")
def score_numeric_ranking(item: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, str]:
    """
    Soft scoring for numeric fields using sigmoid normalization.
    Returns: (score, reason)
    """
    field = params.get("field")
    if not field or field not in item:
        return 0.5, "無此數值，給予中立分數"
    
    try:
        val = float(item[field])
    except (ValueError, TypeError):
        return 0.5, "數值錯誤"
    
    direction = params.get("ranking_direction", "asc")
    normalize_max = params.get("normalize_max", 1000000)
    midpoint = normalize_max / 2
    steepness = 6.0 / normalize_max
    
    # Sigmoid overflow fix
    def safe_sigmoid(x):
        if x > 700:
            return 1.0
        if x < -700:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    if direction == "asc":
        # exp(-x), if (val - midpoint) is very positive, -x is very negative -> result ~1.0
        # if (val - midpoint) is very negative, -x is very positive -> result ~0.0
        x = steepness * (val - midpoint)
        score = safe_sigmoid(x)
        reason = f"數值 {int(val)} (高於平均)"
    else:
        # For desc, we want higher score for lower values.
        # Use negative x so safe_sigmoid handles inversion if x is positive (val > mid)
        x = -steepness * (val - midpoint)
        score = safe_sigmoid(x)
        reason = f"數值 {int(val)} (低於平均)"
    
    return score, reason
