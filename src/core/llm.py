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
        "search_terms": user_query, # Default
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

    # 4. Normalize query_intent (Pre-Retrieval 意圖解析)
    if isinstance(parsed, dict) and "query_intent" in parsed:
        raw_intent = parsed["query_intent"]
        if isinstance(raw_intent, dict):
            # Normalize positive_terms
            positive_terms = raw_intent.get("positive_terms", [])
            if not isinstance(positive_terms, list):
                positive_terms = []

            # Normalize hard_exclusions
            hard_exclusions = []
            for exc in raw_intent.get("hard_exclusions", []):
                if isinstance(exc, dict) and exc.get("term"):
                    hard_exclusions.append({
                        "term": str(exc["term"]).strip(),
                        "strength": "hard",
                        "weight": float(exc.get("weight", -1.0)),
                        "reason": str(exc.get("reason", "")),
                    })

            # Normalize soft_exclusions
            soft_exclusions = []
            for exc in raw_intent.get("soft_exclusions", []):
                if isinstance(exc, dict) and exc.get("term"):
                    weight = float(exc.get("weight", -0.5))
                    # Clamp weight to [-0.8, -0.3] range
                    weight = max(-0.8, min(-0.3, weight))
                    soft_exclusions.append({
                        "term": str(exc["term"]).strip(),
                        "strength": "soft",
                        "weight": weight,
                        "reason": str(exc.get("reason", "")),
                    })

            final_result["query_intent"] = {
                "positive_terms": [str(t).strip() for t in positive_terms if str(t).strip()],
                "hard_exclusions": hard_exclusions,
                "soft_exclusions": soft_exclusions,
            }

    return final_result


from typing import Any, Dict, List, Optional, Tuple

@functools.lru_cache(maxsize=1000)
def parse_query(user_query: str, model_id: Optional[str] = None, tag_list: Optional[Tuple[str, ...]] = None) -> QueryParseResult:
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

    # 手動定義 Schema (语义搜索 + 硬过滤 + 結構化意圖拆解)
    manual_schema = {
        "type": "object",
        "properties": {
            "original_query": {"type": "string"},
            "search_terms": {"type": "string"},
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
                "description": "List of book/novel/anime titles explicitly mentioned as POSITIVE examples you should reference. DO NOT include titles the user dislikes or wants to avoid. Example: '像overlord' -> ['overlord'], but '不要像《刮鬍》' -> []"
            },
            "query_intent": {
                "type": "object",
                "description": "結構化意圖拆解：將查詢拆分為正向搜尋詞與負向約束，避免否定詞進入 BM25 計分",
                "properties": {
                    "positive_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "淨化後的正向搜尋關鍵詞，僅包含使用者真正想要的概念（不含否定修飾詞及被否定的詞彙）"
                    },
                    "hard_exclusions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string", "description": "要排除的標準化標籤/概念或特定書籍"},
                                "strength": {"type": "string", "enum": ["hard"], "description": "固定為 hard"},
                                "weight": {"type": "number", "description": "固定為 -1.0"},
                                "reason": {"type": "string", "description": "排除原因"}
                            },
                            "required": ["term"]
                        },
                        "description": "絕對排除項 (AND NOT)。注意：請將使用者的情緒化抱怨（如：變白癡、爛造、膚淺）轉譯為對應的小說標準標籤（如：降智、無腦爽文、毒草）。如果是明確討厭的書（如不要戀愛光譜），將書名當作標籤放入此處。"
                    },
                    "soft_exclusions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string", "description": "要降權的標準化標籤/概念"},
                                "strength": {"type": "string", "enum": ["soft"], "description": "固定為 soft"},
                                "weight": {"type": "number", "description": "負權重，-0.3 到 -0.8 之間"},
                                "reason": {"type": "string", "description": "降權原因"}
                            },
                            "required": ["term", "weight"]
                        },
                        "description": "柔性排除項：使用者表示「盡量不要」的概念。同樣需轉化為標準標籤。"
                    }
                },
                "required": ["positive_terms"]
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
        "required": ["original_query", "criteria", "search_terms", "generated_keywords", "hypothetical_intro", "reference_books", "query_intent"]
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
       - search_terms: "異世界 後宮 小說"
       - criteria: semantic_similarity(query_text="異世界轉生 後宮 冒險")
       - generated_keywords: ["穿越", "轉生", "魔法", "勇者", "冒險者", "女主角"]
    
    ### IMPORTANT: `is_negative` Rules
    - Use `is_negative: true` ONLY for explicit EXCLUSIONS
    - Example: "不要龍傲天" → semantic_similarity(query_text="龍傲天", is_negative=true)
    - Example: "不要修仙" → semantic_similarity(query_text="修仙", is_negative=true)
    - If user says "角色不要太多", it means WANTS "few characters" → semantic_similarity(query_text="角色少", is_negative=false)
    
    ### CRITICAL TASK: query_intent (結構化意圖拆解)
    You MUST decompose the query into positive_terms and negative constraints.
    This is the MOST IMPORTANT task — it prevents negation words from corrupting BM25 scoring.
    
    **Rules:**
    1. **positive_terms**: Extract ONLY what the user WANTS. Never include negation words (不要/排除/禁止) or negated concepts.
       - "推薦魔法學校小說，不要悲劇" → positive_terms: ["魔法學校", "奇幻", "校園"]
       - "找輕鬆異世界小說，不要後宮不要NTR" → positive_terms: ["輕鬆", "異世界", "冒險"]
    
    2. **hard_exclusions**: For absolute exclusions (不要/不想要/排除/禁止/不能有)
       - "不要悲劇" → hard_exclusions: [{term: "悲劇", strength: "hard", weight: -1.0, reason: "用戶明確排除悲劇"}]
       - "排除BL" → hard_exclusions: [{term: "BL", strength: "hard", weight: -1.0, reason: "用戶排除BL"}]
    
    3. **soft_exclusions**: For soft preferences (盡量不要/最好不要/少一點/希望不要)
       - "盡量不要後宮" → soft_exclusions: [{term: "後宮", strength: "soft", weight: -0.5, reason: "用戶偏好少後宮"}]
       - "最好不要太虐" → soft_exclusions: [{term: "虐心", strength: "soft", weight: -0.4, reason: "用戶偏好不要太虐"}]
       - The weight should be between -0.3 (mild preference) and -0.8 (strong preference)
    
    **Complete Example:**
    Query: "推薦魔法學校冒險小說，不要悲劇，不要主角死亡，盡量不要後宮"
    query_intent: {
        positive_terms: ["魔法學校", "冒險", "奇幻", "校園", "魔法"],
        hard_exclusions: [
            {term: "悲劇", strength: "hard", weight: -1.0, reason: "用戶明確排除悲劇"},
            {term: "主角死亡", strength: "hard", weight: -1.0, reason: "用戶明確排除主角死亡"}
        ],
        soft_exclusions: [
            {term: "後宮", strength: "soft", weight: -0.5, reason: "用戶偏好少後宮"}
        ]
    }
    
    ### TASK 1: Reference Books (reference_books)
    If the user mentions a book, novel, or anime AS A POSITIVE EXAMPLE (e.g. "想看類似《無職》的"), extract the title.
    CRITICAL: DO NOT extract titles the user explicitly states they DISLIKE or want to AVOID (e.g. "不要像《刮鬍》的", "《戀愛光譜》我不喜歡").
    - Example: "像overlord一樣的" → reference_books: ["overlord"]
    - Example: "不要像刀劍神域" → reference_books: [] (And put "刀劍神域" in hard_exclusions)
    - If no POSITIVE book is mentioned, return an empty list: reference_books: []

    
    ### TASK 2: Query Expansion (generated_keywords)
    Generate 5-10 specific domain keywords in Traditional Chinese that capture the semantic intent.
    - Focus on genre-specific terms, tropes, themes
    - IMPORTANT: Only include POSITIVE keywords the user wants. Do NOT include negated concepts.
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
    Always include: original_query, search_terms, generated_keywords, reference_books, hypothetical_intro, query_intent, criteria
    """
    
    if tag_list:
        tag_hint = f"\n\n### AVAILABLE TAGS (Method 2)\nUse the following tags for reference when generating keywords:\n{', '.join(tag_list)}"
        full_system_instruction += tag_hint

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
            search_terms=user_query,
            generated_keywords=[],  # Fallback 不生成关键词
            hypothetical_intro="",  # Fallback 不生成假设简介
            criteria=fallback_criteria
        )

