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
    "廢物": ["廢柴", "吊車尾", "劣等生", "無能", "底邊", "魯蛇"],
    "爽文": ["龍傲天", "無雙", "開掛", "虐菜", "無敵"],
    "逆襲": ["打臉", "反殺", "復仇", "逆轉"],
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

    # Combine field value into a list for uniform processing
    items_to_check = val if isinstance(val, list) else [str(val)]
    # Retrieve LLM expanded keywords (if any) from params
    expanded_keywords = params.get("generated_keywords", [])
    
    total_score = 0.0
    matched_reasons = []

    for k_check in keywords_to_check:
        remaining_keyword = k_check
        
        # We need to process tags iteratively and consume remaining_keyword
        # Evaluate longer tags first to prevent smaller tags from stealing overlap
        sorted_items = sorted([str(v).lower() for v in items_to_check if str(v).lower()], key=len, reverse=True)
        
        for v_str in sorted_items:
            v_len = len(v_str)

            # Condition A: Complete Match OR Synonym Match
            if k_check in v_str or (v_str in k_check and v_str == k_check):
                total_score += 0.3
                reason_str = f"完全命中 '{v_str}'" if v_str == keyword else f"符合同義詞 '{v_str}'"
                matched_reasons.append(reason_str)
                break # Reached max base score for this keyword, no need to check other tags for it
                
            # Condition B: Dynamic Overlap / Sub-word match
            # "魔法" in "魔法學院"
            elif v_len >= 2 and (v_str in remaining_keyword or v_str in keyword):
                total_score += 0.15
                matched_reasons.append(f"包含子標籤 '{v_str}'")
                if v_str in remaining_keyword:
                    remaining_keyword = remaining_keyword.replace(v_str, "", 1)
                continue
                
            # Condition C: Dynamic Related Tags from LLM expansion
            for ext_kw in expanded_keywords:
                # Give higher associative points for expanded keywords that directly overlap
                if v_len >= 2 and (v_str in ext_kw or ext_kw in v_str):
                    total_score += 0.10
                    matched_reasons.append(f"關聯標籤 '{v_str}'")
                    break # Prevent multiple additions for the same tag

        if total_score >= 0.3:
            # If we hit an exact match, we can stop evaluating to save time,
            # but we should still let the current loop finish accumulating so we don't truncate early,
            # so we only break the outer k_check loop.
            pass

    final_score = min(total_score, 0.35) # Cap the score
    
    if final_score > 0:
        # Remove duplicates
        unique_reasons = list(dict.fromkeys(matched_reasons))
        return final_score, f"標籤符合: {', '.join(unique_reasons)}"

    # --- Fallback Logic: Logical Expansion for other fields ---
    # If the primary field (e.g., 'classification', 'tags') didn't yield any partial score, 
    # check other semantic fields like 'name', or 'intro'.
    
    # 2. Check Name/Title
    if field != "name" and "name" in item:
        name_str = str(item["name"]).lower()
        for k in keywords_to_check:
            if k in name_str:
                return 0.2, f"從書名中找到 '{k}' (原搜尋欄位: '{field}')"
                
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
                    return 0.15, f"從簡介中找到 '{k}' (原搜尋欄位: '{field}')"
    
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
    
    # 將影響力減半 (最大 0.5 分)，使其僅作為 Tie-breaker
    final_score = score * 0.5
    return final_score, reason
