import os
import json
from google import genai
from google.genai import types
from src.models.schemas import QueryParseResult
from src.core.api_utils import retry_on_rate_limit, _is_retryable

# 可用模型清單 (依優先順序排列，當前模型失敗時自動切換)
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemma-3-27b-it"]

def parse_query(user_query: str) -> QueryParseResult:
    """
    使用 Google GenAI SDK (v1.0+) 將自然語言查詢轉換為結構化搜尋條件。
    支援多模型 fallback：當主要模型遇到配額限制時，自動嘗試下一個模型。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    selected_model = os.getenv("LLM_MODEL_ID", "").strip()

    # 建立模型嘗試順序：使用者選擇的模型優先，其餘作為 fallback
    if selected_model and selected_model in FALLBACK_MODELS:
        models_to_try = [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
    elif selected_model:
        models_to_try = [selected_model] + FALLBACK_MODELS
    else:
        models_to_try = FALLBACK_MODELS

    client = genai.Client(api_key=api_key)

    # 手動定義 Schema (避免 Pydantic 相容性問題)
    manual_schema = {
        "type": "object",
        "properties": {
            "original_query": {"type": "string"},
            "search_terms": {
                "type": "array",
                "items": {"type": "string"}
            },
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "weight": {"type": "number"},
                        "description": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "keyword": {"type": "string"},
                                "min_val": {"type": "number"},
                                "max_val": {"type": "number"},
                                "target_status": {"type": "string"},
                                "query_text": {"type": "string"},
                                "author_name": {"type": "string"},
                                "require_free": {"type": "boolean"},
                                "allow_restricted": {"type": "boolean"},
                                "require_audio": {"type": "boolean"},
                                "ranking_direction": {"type": "string"},
                                "normalize_max": {"type": "number"}
                            }
                        }
                    },
                    "required": ["name", "weight", "parameters"]
                }
            }
        },
        "required": ["original_query", "criteria", "search_terms"]
    }

    system_instruction = """
    You are a web novel recommendation assistant. Your goal is to break down the user's query into weighted scoring criteria.
    ... [Same parsing logic rules as before, omitted for brevity but implied to be known by the model] ...
    Output a JSON object satisfying the schema.
    """
    # For brevity in this rewrite, I'll rely on the model's general capability or re-inject instructions if needed.
    # To be safe and ensure high quality, I should include the full instructions.
    
    full_system_instruction = """
    You are a web novel recommendation assistant. Your goal is to break down the user's query into weighted scoring criteria.
    
    ### Available Scoring Functions
    1. **keyword_match** (field, keyword): For attributes like 'classification', 'tags', 'name', 'author'.
    2. **numeric_range** (field, min_val, max_val): For HARD FILTERING on 'words_total', 'click_count', etc. Use this when the user specifies an exact range (e.g., "at least 100k words").
    3. **numeric_ranking** (field, ranking_direction, normalize_max): For SOFT RANKING. Use this when the user implies a preference without a hard cutoff (e.g., "longer is better", "I want a very long novel"). 
       - `ranking_direction`: 'asc' means higher values are better; 'desc' means lower values are better.
       - `normalize_max`: A reasonable ceiling for normalization (e.g., 2000000 for words_total).
    4. **status_check** (target_status): 'completed' or 'ongoing'.
    5. **author_match** (author_name).
    6. **is_free_check** (require_free).
    7. **age_check** (allow_restricted).
    8. **audio_available** (require_audio).
    9. **semantic_similarity** (query_text): For abstract vibes/plots.

    Strategy: Map explicit intents to DB fields (keyword_match, status_check) with high confidence. Use semantic_similarity for nuances.
    **CRITICAL**: Use `keyword_match` ONLY for concrete genres/categories (e.g. 'Fantasy', 'Romance') or explicit tags. 
    - **NEVER** use `keyword_match` for descriptive adjectives (e.g. 'exciting', 'sad', 'funny', 'detailed', 'good plot'). Use `semantic_similarity` for these instead.
    - If the user says "I want a VERY long novel" without specifying a number, use numeric_ranking. If they say "at least 500k words", use numeric_range.

    ### IMPORTANT: DATASET LANGUAGE
    The underlying database uses **Traditional Chinese (繁體中文)** for all metadata (names, tags, classifications).
    - For `keyword_match` on fields like `classification` or `tags`, you **MUST** output the `keyword` in Traditional Chinese.
    - **Translate** user limits (e.g. "Fantasy" -> "奇幻", "Magic" -> "魔法", "Romance" -> "言情").
    - If the user queries in Chinese, use the exact Chinese terms they used (or synonyms).
    - Common Classifications: 奇幻, 言情, 都市, 玄幻, 靈異, 武俠, 科幻.
    """

    # --- 模型 Fallback 迴圈 ---
    last_exception = None
    for model_id in models_to_try:
        print(f"[llm] 嘗試使用模型: {model_id}")
        
        @retry_on_rate_limit(max_retries=2, base_delay=5.0)
        def _call_api():
            import re as _re
            # Gemma 3 (and potentially others) might not support system_instruction or JSON mode
            final_prompt = f"User Query: {user_query}"
            is_gemma = "gemma" in model_id.lower()

            if is_gemma:
                # Gemma doesn't support JSON mode — skip response_mime_type/response_schema
                config_args = {}
                # Prepend instruction to content and explicitly ask for JSON
                final_contents = (
                    f"{full_system_instruction}\n\n"
                    f"Task: Parse this query:\n{final_prompt}\n\n"
                    "IMPORTANT: Output ONLY valid JSON (no markdown, no explanation)."
                )
            else:
                # Standard Gemini behavior with JSON mode
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
            if response.text:
                raw_text = response.text.strip()
                # Strip markdown code fences (```json ... ``` or ``` ... ```)
                raw_text = _re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
                raw_text = _re.sub(r"\n?```\s*$", "", raw_text)
                raw_text = raw_text.strip()

                # Parse JSON generically first to handle list vs object
                parsed = json.loads(raw_text)

                if isinstance(parsed, list):
                    # Gemma may return a bare list of criteria objects
                    # Check if items look like criteria (have "name"/"function" + "weight"/"parameters")
                    criteria_list = []
                    for item in parsed:
                        if isinstance(item, dict):
                            # Normalize: Gemma might use "function" instead of "name"
                            if "function" in item and "name" not in item:
                                item["name"] = item.pop("function")
                            
                            # Ensure "parameters" exists
                            if "parameters" not in item:
                                item["parameters"] = {}
                            
                            # Move top-level parameter fields into parameters dict if present
                            # List of known parameters from ScoringParameters schema
                            known_params = [
                                "field", "keyword", "min_val", "max_val", "target_status", 
                                "query_text", "author_name", "require_free", "allow_restricted", 
                                "require_audio", "ranking_direction", "normalize_max"
                            ]
                            for param in known_params:
                                if param in item:
                                    # If param is at top level, move it to parameters
                                    # But prefer existing value in parameters if present (unlikely if structure is flat)
                                    if param not in item["parameters"]:
                                        item["parameters"][param] = item.pop(param)
                                    else:
                                        # Duplicate? Just remove top level
                                        item.pop(param)
                                        
                            # Ensure "weight" exists (default to 0.8 for soft match)
                            if "weight" not in item:
                                item["weight"] = 0.8

                            criteria_list.append(item)

                    parsed = {
                        "original_query": user_query,
                        "search_terms": [user_query],
                        "criteria": criteria_list
                    }

                return QueryParseResult.model_validate(parsed)
            else:
                raise ValueError("Empty response from Gemini")

        try:
            result = _call_api()
            print(f"[llm] 成功使用模型: {model_id}")
            return result
        except Exception as e:
            last_exception = e
            if _is_retryable(e):
                print(f"[llm] 模型 {model_id} 配額受限，嘗試下一個模型...")
                continue
            else:
                # 非配額錯誤 (如模型不存在)，也嘗試下一個
                print(f"[llm] 模型 {model_id} 發生錯誤: {e}，嘗試下一個模型...")
                continue

    # 所有模型都失敗，進入 fallback 邏輯
    if last_exception:
        e = last_exception
        print(f"[llm] 所有模型皆失敗，最後錯誤: {e}")
        
        # --- Enhanced Fallback Logic ---
        # 1. Check for explicit list format (e.g., 'tag1', 'tag2') to handle rate limits gracefully
        import re
        from src.models.schemas import ScoringCriteria, ScoringParameters
        
        # Look for quoted strings or comma-separated values
        # Regex matches content inside single or double quotes
        quoted_matches = re.findall(r"['\"](.*?)['\"]", user_query)
        
        fallback_criteria = []
        
        if quoted_matches:
            # If user provided quoted strings, assume they are emphatic keywords/tags
            for tag in quoted_matches:
                if tag.strip():
                    fallback_criteria.append(
                        ScoringCriteria(
                            name="keyword_match",
                            weight=1.0, # High weight for explicit tags
                            parameters=ScoringParameters(field="tags", keyword=tag.strip())
                        )
                    )
        
        # If no quotes found but contains commas, maybe split by comma?
        elif ',' in user_query or ' ' in user_query:
            parts = [p.strip() for p in user_query.replace(',', ' ').split()]
            # If parts look like specific tags (this is heuristic, might be risky for normal sentences)
            # But for fallback, getting *some* keyword matches is better than pure semantic on a weird string
            # Let's simple check if we have multiple parts
            if len(parts) > 1:
                # Add a few as keywords, but keep semantic similarity as primary
                for p in parts[:5]: # Limit to first 5 to avoid noise
                   fallback_criteria.append(
                        ScoringCriteria(
                            name="keyword_match",
                            weight=0.5, 
                            parameters=ScoringParameters(field="tags", keyword=p)
                        )
                    )

        # Always include Semantic Similarity as a base
        fallback_criteria.append(
            ScoringCriteria(
                name="semantic_similarity", 
                weight=1.0, 
                parameters=ScoringParameters(query_text=user_query)
            )
        )

        return QueryParseResult(
            original_query=user_query,
            search_terms=quoted_matches if quoted_matches else [user_query],
            criteria=fallback_criteria
        )
