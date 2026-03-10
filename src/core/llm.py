import os
import json
import functools
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from src.models.schemas import QueryParseResult
from src.core.api_utils import retry_on_rate_limit, _is_retryable
from src.core.keyword_extractor import KeywordExtractor

# 可用模型清單 (依優先順序排列，當前模型失敗時自動切換)
FALLBACK_MODELS = ["gemma-3-27b-it", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]

def _normalize_llm_output(parsed: Any, user_query: str) -> Dict[str, Any]:
    """
    Normalizes LLM output (dict or list) into a structure matching QueryParseResult.
    Handles snake_case conversion, missing fields, and flat lists.
    """
    import re
    
    def to_snake_case(name: str) -> str:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def recursive_normalize_keys(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_k = to_snake_case(k)
                new_obj[new_k] = recursive_normalize_keys(v)
            return new_obj
        elif isinstance(obj, list):
            return [recursive_normalize_keys(item) for item in obj]
        else:
            return obj

    # 1. Normalize keys to snake_case
    parsed = recursive_normalize_keys(parsed)

    final_result = {
        "original_query": user_query,
        "search_terms": [user_query], # Default
        "generated_keywords": [],
        "reference_books": [],
        "hypothetical_intro": "",
        "criteria": []
    }

    # 2. Handle Dict vs List
    if isinstance(parsed, dict):
        # Merge into defaults
        if "original_query" in parsed: final_result["original_query"] = parsed["original_query"]
        if "search_terms" in parsed: final_result["search_terms"] = parsed["search_terms"]
        if "generated_keywords" in parsed: final_result["generated_keywords"] = parsed["generated_keywords"]
        if "reference_books" in parsed: final_result["reference_books"] = parsed["reference_books"]
        if "hypothetical_intro" in parsed: final_result["hypothetical_intro"] = parsed["hypothetical_intro"]
        
        # Handle criteria/scoring_criteria alias
        if "criteria" in parsed:
            final_result["criteria"] = parsed["criteria"]
        elif "scoring_criteria" in parsed:
            final_result["criteria"] = parsed["scoring_criteria"]
            
    elif isinstance(parsed, list):
        # Assume list of criteria
        final_result["criteria"] = parsed

    # 3. Validate/Fix Criteria List
    valid_criteria = []
    if isinstance(final_result["criteria"], list):
        for item in final_result["criteria"]:
            if not isinstance(item, dict): continue
            
            # Fix Name/Function/Type aliases
            if "function" in item and "name" not in item:
                item["name"] = item.pop("function")
            if "type" in item and "name" not in item:
                item["name"] = item.pop("type")
            if "name" not in item: continue # Skip if no name

            # Fix Parameters
            if "parameters" not in item:
                item["parameters"] = {}
            
            # Move top-level params to parameters dict
            known_params = [
                "field", "keyword", "min_val", "max_val", "target_status", 
                "query_text", "author_name", "ranking_direction", "normalize_max"
            ]
            for param in known_params:
                if param in item:
                    if param not in item["parameters"]:
                        item["parameters"][param] = item.pop(param)
                    else:
                        item.pop(param) # Duplicate
            
            # Remove non-schema keys Gemma might add (e.g. "weight", "type")
            for extra_key in ["weight", "type", "function"]:
                item.pop(extra_key, None)
            
            # Fix Is Negative
            if "is_negative" not in item:
                item["is_negative"] = False
            else:
                item["is_negative"] = bool(item["is_negative"])

            valid_criteria.append(item)
    
    final_result["criteria"] = valid_criteria
    return final_result


@functools.lru_cache(maxsize=100)
def parse_query(user_query: str, model_id: Optional[str] = None) -> QueryParseResult:
    """
    使用 Google GenAI SDK (v1.0+) 將自然語言查詢轉換為結構化搜尋條件。
    支援多模型 fallback：當主要模型遇到配額限制或錯誤時，自動嘗試下一個模型。
    支援 API Key 輪換：當遇到速率限制時，自動切換到下一個 API Key。
    """
    from src.core.api_utils import get_current_api_key, get_api_key_rotator, _is_retryable
    
    selected_model = model_id or FALLBACK_MODELS[0]

    # 建立模型嘗試順序
    if selected_model and selected_model in FALLBACK_MODELS:
        models_to_try = [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
    elif selected_model:
        models_to_try = [selected_model] + FALLBACK_MODELS
    else:
        models_to_try = FALLBACK_MODELS

    # 手動定義 Schema (简化为语义搜索 + 硬过滤模式)
    manual_schema = {
        "type": "object",
        "properties": {
            "original_query": {"type": "string"},
            "search_terms": {"type": "array", "items": {"type": "string"}},
            "generated_keywords": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "5-10 specific domain keywords for semantic expansion."
            },
            "hypothetical_intro": {
                "type": "string",
                "description": "A creative, hypothetical book introduction (blurb) in Traditional Chinese that perfectly matches the user's intent. About 50-100 words."
            },
            "reference_books": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of book/novel/anime titles explicitly mentioned or referenced in the user query. Extract the title as-is from the query. Examples: '為美好世界獻上祝福' from '跟為美好世界獻上祝福差不多的', 'overlord' from '像overlord一樣的'"
            },
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["semantic_similarity", "status_check", "author_match", "numeric_range"],
                            "description": "Function name. Only use: semantic_similarity (for main search), status_check (completed/ongoing), author_match (specific author), numeric_range (word count only)"
                        },
                        "is_negative": {
                            "type": "boolean",
                            "description": "Only applicable for semantic_similarity. True for exclusions like '不要龍傲天'"
                        },
                        "description": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query_text": {
                                    "type": "string",
                                    "description": "For semantic_similarity: the semantic concept to search/exclude"
                                },
                                "target_status": {
                                    "type": "string",
                                    "description": "For status_check: 'completed', 'finished', 'ongoing', 'serializing'"
                                },
                                "author_name": {
                                    "type": "string",
                                    "description": "For author_match: author name"
                                },
                                "field": {
                                    "type": "string",
                                    "description": "For numeric_range: must be 'words_total'"
                                },
                                "min_val": {
                                    "type": "number",
                                    "description": "For numeric_range: minimum word count"
                                },
                                "max_val": {
                                    "type": "number",
                                    "description": "For numeric_range: maximum word count"
                                }
                            }
                        }
                    },
                    "required": ["name", "parameters"]
                }
            }
        },
        "required": ["original_query", "criteria", "search_terms", "generated_keywords", "hypothetical_intro", "reference_books"]
    }

    system_instruction = """
    You are a web novel recommendation assistant. Your goal is to break down the user's query into scoring criteria.
    Output a JSON object satisfying the schema.
    """
    
    full_system_instruction = """
    You are a web novel recommendation assistant. Your goal is to parse user queries into semantic search + optional filters.
    
    ### CORE PRINCIPLE
    - **Primary Method**: Semantic Vector Search (rely on embedding similarity)
    - **Optional Filters**: Only use when user explicitly specifies hard constraints
    
    ### Available Filter Functions (Database-level filtering, NOT scoring)
    Use these ONLY when user explicitly mentions these constraints:
    
    1. **status_check**(target_status): Use ONLY when user explicitly wants "完結" (completed) or "連載" (ongoing)
       - Valid values: "completed", "finished", "ongoing", "serializing"
       - Example: "找完結小說" → status_check(target_status="completed")
    
    2. **author_match**(author_name): Use ONLY when user specifies an author name
       - Example: "猫腻的小說" → author_match(author_name="猫腻")
    
    3. **numeric_range**(field="words_total", min_val, max_val): Use for HARD word count requirements
       - field MUST be "words_total" (no other fields allowed)
       - min_val/max_val in actual word count (e.g., 20萬字 = 200000)
       - **IMPORTANT**: Only use for STRICT requirements. If user says "最好", "推薦", "建議", "希望" (soft preferences), DO NOT use this filter
       - Example: "20萬字以上" (strict) → numeric_range(field="words_total", min_val=200000)
       - Example: "最好10萬字以上" (soft) → DO NOT use numeric_range, rely on semantic search
       - Example: "必須20-50萬字" (strict) → numeric_range(field="words_total", min_val=200000, max_val=500000)
    
    ### Semantic Search Strategy
    For ALL other requirements (genre, tags, plot, character traits, style), rely on semantic similarity:
    - **DO NOT use** `keyword_match` - it's deprecated
    - **DO NOT use** `numeric_ranking` - it's deprecated
    - Use `semantic_similarity` for positive semantic requirements (genre, tags, themes, tropes, etc.)
    - Use `semantic_similarity` with `is_negative: true` for exclusions
    
    ### IMPORTANT: Tag & Genre Search
    When user mentions specific tags or genres (e.g., "異世界", "後宮", "奇幻", "搞笑"), you MUST:
    1. Add a `semantic_similarity` criteria with `query_text` containing the tag/genre keywords
    2. Also include these terms in `search_terms` for direct vector matching
    3. Example: "找異世界後宮小說" → 
       - search_terms: ["異世界 後宮 小說"]
       - criteria: semantic_similarity(query_text="異世界轉生 後宮 冒險")
       - generated_keywords: ["穿越", "轉生", "魔法", "勇者", "冒險者", "女主角"]
    
    ### IMPORTANT: `is_negative` Rules
    - Use `is_negative: true` ONLY for explicit EXCLUSIONS
    - Example: "不要龍傲天" → semantic_similarity(query_text="龍傲天", is_negative=true)
    - Example: "不要修仙" → semantic_similarity(query_text="修仙", is_negative=true)
    - If user says "角色不要太多", it means WANTS "few characters" → semantic_similarity(query_text="角色少", is_negative=false)
    
    ### TASK 1: Reference Books (reference_books)
    If the user mentions ANY book, novel, anime, manga, or light novel title (with or without 《》), extract the title as-is.
    - Example: "跟為美好世界獻上祝福差不多的" → reference_books: ["為美好世界獻上祝福"]
    - Example: "像overlord一樣的" → reference_books: ["overlord"]
    - Example: "《無職轉生》風格" → reference_books: ["無職轉生"]
    - Example: "找類似轉生史萊姆跟盾之勇者的書" → reference_books: ["轉生史萊姆", "盾之勇者"]
    - If no book is mentioned, return an empty list: reference_books: []
    
    ### TASK 2: Query Expansion (generated_keywords)
    Generate 5-10 specific domain keywords in Traditional Chinese that capture the semantic intent.
    - Focus on genre-specific terms, tropes, themes
    - Example for "科幻": ["太空", "未來", "科技", "星際", "機器人", "時間旅行"]
    
    ### TASK 3: Hypothetical Document Embeddings (hypothetical_intro)
    Generate a hypothetical book introduction (50-100 words) that matches the query perfectly.
    - **Style**: Dramatic, engaging, using web novel tropes
    - **Language**: Traditional Chinese (繁體中文)
    - **Goal**: Create text semantically similar to actual book summaries in database
    
    **Examples:**
    - Query: "網遊小說"
      hypothetical_intro: "一款劃時代的虛擬實境遊戲《榮耀》橫空出世，主角葉修手持千機傘，帶領一群菜鳥重返巔峰..."
    
    - Query: "打臉爽文"
      hypothetical_intro: "家族棄少受盡冷眼，一朝覺醒無上血脈。曾經羞辱我的人，如今都要跪在我腳下顫抖！"
    
    - Query: "輕鬆治癒"
      hypothetical_intro: "厭倦了城市的喧囂，他回到鄉下繼承了一間破舊的小食堂。溫暖的料理，治癒了每一位客人的心靈..."
    
    ### Output Format
    Always include: original_query, search_terms, generated_keywords, reference_books, hypothetical_intro, criteria
    """

    last_exception = None

    for model_id in models_to_try:
        print(f"[llm] 嘗試使用模型: {model_id}")
        
        # Config params
        is_gemma = "gemma" in model_id.lower()
        rotator = get_api_key_rotator()
        api_key_attempts = 0
        max_api_key_attempts = len(rotator.api_keys)
        
        while api_key_attempts < max_api_key_attempts:
            try:
                api_key = get_current_api_key()
                client = genai.Client(api_key=api_key)
                
                # 定義內部呼叫函數以便使用 @retry_on_rate_limit 處理 429/503
                @retry_on_rate_limit(max_retries=2 if is_gemma else 1, base_delay=3.0)
                def _do_generate():
                    import re as _re
                    final_prompt = f"User Query: {user_query}"
                    
                    if is_gemma:
                        config_args = {}
                        final_contents = (
                            f"{full_system_instruction}\n\n"
                            f"Task: Parse this query:\n{final_prompt}\n\n"
                            "IMPORTANT: Output ONLY valid JSON (no markdown). Ensure keys are snake_case."
                        )
                    else:
                        config_args = {
                            "response_mime_type": "application/json",
                            "response_schema": manual_schema,
                            "system_instruction": full_system_instruction,
                        }
                        final_contents = final_prompt

                    response = client.models.generate_content(
                        model=model_id,
                        contents=final_contents,
                        config=types.GenerateContentConfig(**config_args)
                    )
                    
                    if not response.text:
                        raise ValueError("Empty response from LLM")
                        
                    raw_text = response.text.strip()
                    # Strip markdown
                    raw_text = _re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
                    raw_text = _re.sub(r"\n?```\s*$", "", raw_text)
                    raw_text = raw_text.strip()
                    
                    try:
                        parsed = json.loads(raw_text)
                    except json.JSONDecodeError as je:
                        print(f"[llm] JSON Decode Error: {je} \nRaw: {raw_text[:100]}...")
                        raise je

                    # Normalize
                    normalized_data = _normalize_llm_output(parsed, user_query)
                    
                    # Validate with Pydantic
                    return QueryParseResult.model_validate(normalized_data)

                # Execute call
                result = _do_generate()
                print(f"[llm] 成功使用模型: {model_id}")
                return result

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                
                if is_rate_limit and api_key_attempts < max_api_key_attempts - 1:
                    print(f"[llm] API Key 遇到速率限制。切換到下一個 Key...")
                    rotator.on_rate_limit_error()
                    api_key_attempts += 1
                    import time
                    time.sleep(2)
                    continue
                else:
                    last_exception = e
                    print(f"[llm] 模型 {model_id} 發生錯誤: {e}")
                    break  # Try next model

    # 所有模型都失敗，進入 fallback 邏輯
    if last_exception:
        print(f"[llm] 所有模型皆失敗，最後錯誤: {last_exception}")
        
        # --- Simplified Fallback Logic: Pure Semantic Search ---
        from src.models.schemas import ScoringCriteria, ScoringParameters
        
        print("[llm] 進入簡化 Fallback 模式：純語意搜尋")
        
        # 只生成一个 semantic_similarity criteria
        fallback_criteria = [
            ScoringCriteria(
                name="semantic_similarity", 
                is_negative=False,
                parameters=ScoringParameters(query_text=user_query)
            )
        ]

        return QueryParseResult(
            original_query=user_query,
            search_terms=[user_query],
            generated_keywords=[],  # Fallback 不生成关键词
            hypothetical_intro="",  # Fallback 不生成假设简介
            criteria=fallback_criteria
        )

