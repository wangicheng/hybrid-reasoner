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
        "hypothetical_intro": "",
        "criteria": []
    }

    # 2. Handle Dict vs List
    if isinstance(parsed, dict):
        # Merge into defaults
        if "original_query" in parsed: final_result["original_query"] = parsed["original_query"]
        if "search_terms" in parsed: final_result["search_terms"] = parsed["search_terms"]
        if "generated_keywords" in parsed: final_result["generated_keywords"] = parsed["generated_keywords"]
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
            
            # Fix Name/Function
            if "function" in item and "name" not in item:
                item["name"] = item.pop("function")
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
            
            # Fix Is Negative
            if "is_negative" not in item:
                item["is_negative"] = False
            else:
                item["is_negative"] = bool(item["is_negative"])

            valid_criteria.append(item)
    
    final_result["criteria"] = valid_criteria
    return final_result


def _verify_narrative_logic_sync(query_trait: str, book_intro: str, model_id: Optional[str] = None) -> bool:
    """
    輕量級 NLI 審查（同步版本）：判斷書籍簡介的敘事邏輯是否符合使用者的特定要求。
    返回 True 表示符合或無法確定，False 表示明確違反邏輯。
    """
    if not book_intro or not query_trait:
        return True
        
    api_key = os.environ.get("GOOGLE_API_KEY")
    selected_model = model_id or os.getenv("LLM_MODEL_ID", "gemini-3-flash-preview").strip()
    
    if selected_model and selected_model in FALLBACK_MODELS:
        models_to_try = [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
    elif selected_model:
        models_to_try = [selected_model] + FALLBACK_MODELS
    else:
        models_to_try = FALLBACK_MODELS

    client = genai.Client(api_key=api_key)
    
    # 截短簡介避免 Token 浪費 (NLI 審查只需前 300 字即可判斷)
    truncated_intro = book_intro[:300]
    
    prompt = f"""
    Please perform a Natural Language Inference (NLI) check.
    User's desired story trait/logic: "{query_trait}"
    Book Introduction:
    '''
    {truncated_intro}
    '''
    Does the book introduction explicitly CONTRADICT the desired logic?
    For example, if the user asks for "廢物逆襲" (starts weak/trash, becomes strong), but the book intro portrays a character who "starts as the strongest and loses all power" or "was betrayed and lost everything" without a clear "starting as trash" narrative, that is a contradiction.
    
    Output ONLY valid JSON:
    {{"contradicts": true/false}}
    """
    
    for mid in models_to_try:
        try:
            @retry_on_rate_limit(max_retries=1, base_delay=1.0)
            def _do_generate():
                is_gemma = "gemma" in mid.lower()
                if is_gemma:
                     response = client.models.generate_content(
                        model=mid,
                        contents=prompt + "\nIMPORTANT: Output ONLY valid JSON (no markdown)."
                    )
                else:
                    response = client.models.generate_content(
                        model=mid,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema={"type": "object", "properties": {"contradicts": {"type": "boolean"}}, "required": ["contradicts"]}
                        )
                    )
                import re as _re
                raw_text = response.text.strip()
                raw_text = _re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
                raw_text = _re.sub(r"\n?```\s*$", "", raw_text)
                parsed = json.loads(raw_text.strip())
                return not parsed.get("contradicts", False) # Return False if it contradicts

            return _do_generate()
        except Exception as e:
            print(f"[llm] NLI Check Failed with model {mid}: {e}")
            continue
            
    # Default to True (allow the book) if NLI fails entirely
    return True


async def verify_narrative_logic_async(query_trait: str, book_intro: str, model_id: Optional[str] = None) -> bool:
    """
    非同步版 NLI 審查：透過 asyncio.to_thread 包裝同步函式，
    讓伺服器能利用 asyncio.gather 並行審核多本書而不卡死。
    """
    import asyncio
    return await asyncio.to_thread(_verify_narrative_logic_sync, query_trait, book_intro, model_id)


@functools.lru_cache(maxsize=100)
def parse_query(user_query: str, model_id: Optional[str] = None) -> QueryParseResult:
    """
    使用 Google GenAI SDK (v1.0+) 將自然語言查詢轉換為結構化搜尋條件。
    支援多模型 fallback：當主要模型遇到配額限制或錯誤時，自動嘗試下一個模型。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    selected_model = model_id or os.getenv("LLM_MODEL_ID", "").strip()

    # 建立模型嘗試順序
    if selected_model and selected_model in FALLBACK_MODELS:
        models_to_try = [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
    elif selected_model:
        models_to_try = [selected_model] + FALLBACK_MODELS
    else:
        models_to_try = FALLBACK_MODELS

    client = genai.Client(api_key=api_key)

    # 之前嘗試過用 NER (KeyBERT) 輔助提示 LLM，但效果不佳 (太瑣碎)。
    # 現在改回純 LLM 理解，但保留 KeywordExtractor 程式碼以備不時之需 (fallback logic)。
    # Extractor is NOT used here to prompt the LLM.

    # 手動定義 Schema (保留原有的 schema 定義)
    manual_schema = {
        "type": "object",
        "properties": {
            "original_query": {"type": "string"},
            "search_terms": {"type": "array", "items": {"type": "string"}},
            "generated_keywords": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "5-10 specific domain keywords."
            },
            "hypothetical_intro": {
                "type": "string",
                "description": "A creative, hypothetical book introduction (blurb) in Traditional Chinese that perfectly matches the user's intent. About 50-100 words."
            },
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "is_negative": {"type": "boolean"},
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
                                "ranking_direction": {"type": "string"},
                                "normalize_max": {"type": "number"}
                            }
                        }
                    },
                    "required": ["name", "parameters"]
                }
            }
        },
        "required": ["original_query", "criteria", "search_terms", "generated_keywords", "hypothetical_intro"]
    }

    system_instruction = """
    You are a web novel recommendation assistant. Your goal is to break down the user's query into scoring criteria.
    Output a JSON object satisfying the schema.
    """
    
    full_system_instruction = """
    You are a web novel recommendation assistant. Your goal is to break down the user's query into scoring criteria.
    
    ### Available Scoring Functions
    1. **keyword_match** (field, keyword): 'classification', 'tags', 'name'.
    2. **numeric_range** (field, min_val, max_val): ONLY 'words_total', 'bookmark_count', 'rating_score', 'total_recommendations'. DO NOT invent fields!
    3. **numeric_ranking** (field, ranking_direction, normalize_max): ONLY 'words_total', 'bookmark_count', 'rating_score', 'total_recommendations'. Soft ranking.
    4. **status_check** (target_status): 'completed', 'ongoing'.
    5. **author_match** (author_name).
    6. **semantic_similarity** (query_text): Concepts that do not fit in tags, e.g. "主角聰明", "角色少".

    Strategy: Use `keyword_match` ONLY for explicit existing genres/tags. Use `semantic_similarity` for story features (like "角色少").
    
    ### Rules for `is_negative`
    Use `is_negative: true` ONLY for explicit EXCLUSIONS (e.g., "不要龍傲天" -> `is_negative: true`, keyword: "龍傲天").
    DO NOT use `is_negative: true` if the user is asking for a trait, even if it uses negative words! 
    For example, "角色不要太多" means the user WANTS "few characters". You should use `semantic_similarity` with query_text="角色少" and `is_negative: false`.
    
    [NEW RULE: Negative Tag Generation for Logical Fallacies]
    If the user's query implies a strong positive progression (e.g., "廢物逆襲", "從弱小變強大"), you MUST proactively add an explicit EXCLUSION for the "fallen hero" trope to prevent false positives in vector search.
    Output: `is_negative: true`, `keyword_match` for "失去力量" or `semantic_similarity` for "曾經最強跌落神壇".

    ### TASK: DYNAMIC QUERY EXPANSION
    In `generated_keywords`, generate 5-10 specific terms (Traditional Chinese) relevant to the query concepts.
    If the user uses slang like "爽文", expand it to ["龍傲天", "開掛", "無雙", "無敵"]. If "廢物", expand it to ["廢柴", "吊車尾"].

    ### TASK: HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE)
    In addition to keywords, generate a `hypothetical_intro`.
    Imagine a perfect novel exists that satisfies the user's query. Write a short **Book Intro (Blurb)** for it.
    - **Style**: Use the tone typical of web novels (dramatic, engaging, using genre tropes).
    - **Language**: Traditional Chinese (繁體中文).
    - **Length**: 50 to 100 words.

    **Examples:**
    - Query: "網遊小說"
      - hypothetical_intro: "一款劃時代的虛擬實境遊戲《榮耀》橫空出世，主角葉修手持千機傘，帶領一群菜鳥重返巔峰..."
    - Query: "主角原本是廢物但是突然覺醒逆襲變強復仇的爽文"
      - hypothetical_intro: "家族棄少受盡冷眼，一朝覺醒無上血脈。曾經羞辱我的人，如今都要跪在我腳下顫抖！三十年河東，三十年河西..."
      - negative criteria strategy: Must add `is_negative: true` for "失去力量" to ensure it looks for weak-to-strong.

    Your goal is to create a text chunk that is semantically similar to the *actual summaries* in the database.
    """

    last_exception = None

    for model_id in models_to_try:
        print(f"[llm] 嘗試使用模型: {model_id}")
        
        # Config params
        is_gemma = "gemma" in model_id.lower()
        
        try:
            # 定義內部呼叫函數以便使用 @retry_on_rate_limit 處理 429/503
            # 如果是 Gemma，多給一些重試機會
            @retry_on_rate_limit(max_retries=3 if is_gemma else 2, base_delay=5.0)
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
            last_exception = e
            print(f"[llm] 模型 {model_id} 發生錯誤: {e}")
            # Try next model in models_to_try
            continue

    # 所有模型都失敗，進入 fallback 邏輯
    if last_exception:
        print(f"[llm] 所有模型皆失敗，最後錯誤: {last_exception}")
        
        # --- Fallback Logic ---
        import re
        from src.models.schemas import ScoringCriteria, ScoringParameters
        
        quoted_matches = re.findall(r"['\"](.*?)['\"]", user_query)
        fallback_criteria = []
        
        if quoted_matches:
            for tag in quoted_matches:
                if tag.strip():
                    fallback_criteria.append(
                        ScoringCriteria(
                            name="keyword_match",
                            is_negative=False,
                            parameters=ScoringParameters(field="tags", keyword=tag.strip())
                        )
                    )
        elif ',' in user_query or ' ' in user_query:
            parts = [p.strip() for p in user_query.replace(',', ' ').split()]
            if len(parts) > 1:
                for p in parts[:5]:
                   fallback_criteria.append(
                        ScoringCriteria(
                            name="keyword_match",
                            is_negative=False,
                            parameters=ScoringParameters(field="tags", keyword=p)
                        )
                    )

        fallback_criteria.append(
            ScoringCriteria(
                name="semantic_similarity", 
                is_negative=False,
                parameters=ScoringParameters(query_text=user_query)
            )
        )

        return QueryParseResult(
            original_query=user_query,
            search_terms=quoted_matches if quoted_matches else [user_query],
            hypothetical_intro="",
            criteria=fallback_criteria
        )
