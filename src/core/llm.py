import os
import json
import functools
from typing import Any, Dict, List, Optional, Tuple
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
                "query_text", "author_name", "require_free", "allow_restricted", 
                "require_audio", "ranking_direction", "normalize_max"
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
    2. **numeric_range** (field, min_val, max_val): ONLY 'words_total', 'click_count', 'chapters_total'. DO NOT invent fields!
    3. **numeric_ranking** (field, ranking_direction, normalize_max): ONLY 'words_total', 'click_count', 'chapters_total'. Soft ranking.
    4. **status_check** (target_status): 'completed', 'ongoing'.
    5. **author_match** (author_name).
    6. **semantic_similarity** (query_text): Concepts that do not fit in tags, e.g. "主角聰明", "角色少".

    Strategy: Use `keyword_match` ONLY for explicit existing genres/tags. Use `semantic_similarity` for story features (like "角色少").
    
    ### Rules for `is_negative`
    Use `is_negative: true` ONLY for explicit EXCLUSIONS (e.g., "不要龍傲天" -> `is_negative: true`, keyword: "龍傲天").
    DO NOT use `is_negative: true` if the user is asking for a trait, even if it uses negative words! 
    For example, "角色不要太多" means the user WANTS "few characters". You should use `semantic_similarity` with query_text="角色少" and `is_negative: false`.
    
    ### TASK: DYNAMIC QUERY EXPANSION
    In `generated_keywords`, generate 5-10 specific terms (Traditional Chinese) relevant to the query concepts.
    
    ### TASK: HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE)
    In addition to keywords, generate a `hypothetical_intro`.
    Imagine a perfect novel exists that satisfies the user's query. Write a short **Book Intro (Blurb)** for it.
    - **Style**: Use the tone typical of web novels (dramatic, engaging, using genre tropes).
    - **Language**: Traditional Chinese (繁體中文).
    - **Length**: 50 to 100 words.

    **Examples:**
    - Query: "網遊小說"
      - hypothetical_intro: "一款劃時代的虛擬實境遊戲《榮耀》橫空出世，主角葉修手持千機傘，帶領一群菜鳥重返巔峰..."
    - Query: "打臉爽文"
      - hypothetical_intro: "家族棄少受盡冷眼，一朝覺醒無上血脈。曾經羞辱我的人，如今都要跪在我腳下顫抖！三十年河東，三十年河西..."
    - Query: "輕鬆治癒"
      - hypothetical_intro: "厭倦了城市的喧囂，他回到鄉下繼承了一間破舊的小食堂。沒想到，這裡的客人竟然都是..."

    Your goal is to create a text chunk that is semantically similar to the *actual summaries* in the database.
    """

    last_exception = None

    for model_id in models_to_try:
        print(f"[llm] 嘗試使用模型: {model_id}")
        
        # Config params
        is_gemma = "gemma" in model_id.lower()
        
        try:
            # 定義內部呼叫函數以便使用 @retry_on_rate_limit 處理 429/503
            # Gemini 失敗一次就當作 token 用完，直接換 Gemma；Gemma 多給重試機會
            @retry_on_rate_limit(max_retries=3 if is_gemma else 1, base_delay=2.0)
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


# ============================================================
# 兩階段解析（支援 Tool Calling）
# ============================================================

def _stage1_analyze_intent(user_query: str, model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    第一階段：分析用戶意圖，決定是否需要調用工具
    
    Returns:
        {
            "tool_calls": [
                {"name": "analyze_book_mentions", "arguments": {...}}
            ],
            "preliminary_keywords": [...],
            "preliminary_criteria": [...]
        }
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    selected_model = model_id or os.getenv("LLM_MODEL_ID", "gemini-2.5-flash-lite")
    client = genai.Client(api_key=api_key)
    
    # 導入工具註冊表獲取可用工具
    from src.core.tool_registry import ToolRegistry
    tools_schema = ToolRegistry.get_tools_schema()
    
    # 第一階段 Schema - 詳細定義 arguments 結構
    stage1_schema = {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "description": "需要調用的工具列表。如果用戶提到了具體的書籍名稱並表達喜好，應該調用 analyze_book_mentions。",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "工具名稱，例如 'analyze_book_mentions'"},
                        "arguments": {
                            "type": "object", 
                            "description": "工具參數",
                            "properties": {
                                "liked_books": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "用戶喜歡的書籍名稱列表"
                                },
                                "disliked_books": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "用戶不喜歡的書籍名稱列表"
                                },
                                "neutral_books": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "用戶提到但未表達偏好的書籍名稱列表"
                                }
                            },
                            "required": ["liked_books", "disliked_books"]
                        }
                    },
                    "required": ["name", "arguments"]
                }
            },
            "preliminary_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "從查詢中提取的初步關鍵詞"
            },
            "preliminary_search_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "查詢的核心搜尋詞"
            }
        },
        "required": ["tool_calls", "preliminary_keywords", "preliminary_search_terms"]
    }
    
    # 構建工具描述
    tools_description = "\n".join([
        f"- **{t['name']}**: {t['description']}\n  參數: {json.dumps(t['parameters'], ensure_ascii=False)}"
        for t in tools_schema
    ])
    
    system_prompt = f"""你是一個查詢分析助手。你的任務是分析用戶的查詢，並決定是否需要調用工具來獲取更多資訊。

## 可用工具：
{tools_description}

## 判斷規則：
1. 如果用戶提到了具體的書籍名稱（如《刀劍神域》、「無職轉生」等），並且表達了喜歡或不喜歡的態度，應該調用 `analyze_book_mentions` 工具。
2. 如果用戶只是描述想要的類型（如「奇幻冒險」「完結小說」），不需要調用工具。
3. 書籍名稱可能用《》、「」包裹，也可能直接提及。

## 輸出要求：
- tool_calls: 如果需要調用工具，填入工具名稱和參數；如果不需要，返回空數組 []
- preliminary_keywords: 提取查詢中的核心關鍵詞
- preliminary_search_terms: 查詢的主要搜尋詞

## 範例：
用戶: "我喜歡《刀劍神域》，推薦類似的"
輸出: {{
  "tool_calls": [{{
    "name": "analyze_book_mentions",
    "arguments": {{
      "liked_books": ["刀劍神域"],
      "disliked_books": [],
      "neutral_books": []
    }}
  }}],
  "preliminary_keywords": ["VR遊戲", "冒險", "虛擬實境"],
  "preliminary_search_terms": ["類似刀劍神域"]
}}

用戶: "推薦完結的奇幻小說"
輸出: {{
  "tool_calls": [],
  "preliminary_keywords": ["奇幻", "完結"],
  "preliminary_search_terms": ["完結奇幻小說"]
}}
"""

    try:
        # Gemini 失敗一次就當作 token 用完，不重試
        @retry_on_rate_limit(max_retries=1, base_delay=1.0)
        def _do_stage1():
            response = client.models.generate_content(
                model=selected_model,
                contents=f"用戶查詢: {user_query}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=stage1_schema,
                    temperature=0.1
                )
            )
            
            if not response.text:
                raise ValueError("Empty response")
            
            return json.loads(response.text.strip())
        
        result = _do_stage1()
        print(f"[llm] Stage 1 分析完成: tool_calls={len(result.get('tool_calls', []))}")
        return result
        
    except Exception as e:
        print(f"[llm] Stage 1 分析失敗: {e}")
        return {
            "tool_calls": [],
            "preliminary_keywords": [],
            "preliminary_search_terms": [user_query]
        }


def parse_query_with_tools(
    user_query: str, 
    model_id: Optional[str] = None,
    enable_tools: bool = True
) -> Tuple[QueryParseResult, Dict[str, Any]]:
    """
    兩階段查詢解析（支援 Tool Calling）
    
    流程：
    1. 第一階段：分析意圖，決定是否需要調用工具
    2. 執行工具（如果需要）
    3. 第二階段：用增強後的輸入進行完整解析
    
    Args:
        user_query: 用戶的原始查詢
        model_id: LLM 模型 ID
        enable_tools: 是否啟用工具調用（預設 True）
    
    Returns:
        Tuple of (QueryParseResult, tool_context)
        - QueryParseResult: 最終的解析結果
        - tool_context: 工具調用的上下文資訊（包含 tool_results, enhanced_query 等）
    """
    from src.core.tool_registry import ToolRegistry, ToolCall, format_tool_result_for_query
    
    tool_context = {
        "tools_enabled": enable_tools,
        "tool_calls": [],
        "tool_results": [],
        "enhanced_query": user_query,
        "enhancement_text": ""
    }
    
    if not enable_tools:
        # 不使用工具，直接用原本的 parse_query
        result = parse_query(user_query, model_id=model_id)
        return result, tool_context
    
    # === 第一階段：分析意圖 ===
    print(f"[llm] 開始兩階段解析...")
    stage1_result = _stage1_analyze_intent(user_query, model_id=model_id)
    
    tool_calls_raw = stage1_result.get("tool_calls", [])
    
    if not tool_calls_raw:
        # 不需要調用工具，直接用原本的 parse_query
        print(f"[llm] 不需要工具調用，進行標準解析")
        result = parse_query(user_query, model_id=model_id)
        return result, tool_context
    
    # === 執行工具 ===
    print(f"[llm] 需要調用 {len(tool_calls_raw)} 個工具")
    tool_calls = [
        ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
        for tc in tool_calls_raw
    ]
    tool_context["tool_calls"] = tool_calls_raw
    
    tool_results = ToolRegistry.execute_all(tool_calls)
    tool_context["tool_results"] = [
        {"name": r.name, "success": r.success, "result": r.result, "error": r.error}
        for r in tool_results
    ]
    
    # 檢查是否有成功的結果
    successful_results = [r for r in tool_results if r.success]
    if not successful_results:
        print(f"[llm] 工具調用無成功結果，進行標準解析")
        result = parse_query(user_query, model_id=model_id)
        return result, tool_context
    
    # === 格式化增強文字 ===
    enhancement_text = format_tool_result_for_query(tool_results)
    tool_context["enhancement_text"] = enhancement_text
    
    # === 構建增強後的查詢 ===
    if enhancement_text:
        enhanced_query = f"{user_query}\n\n### 背景資訊（來自書籍分析）:\n{enhancement_text}"
        tool_context["enhanced_query"] = enhanced_query
        print(f"[llm] 增強查詢:\n{enhancement_text[:200]}...")
    else:
        enhanced_query = user_query
    
    # === 第二階段：完整解析 ===
    # 注意：parse_query 有 cache，所以用增強後的查詢可能不會被 cache
    # 這裡直接呼叫，讓它用增強後的輸入
    print(f"[llm] 進行第二階段完整解析...")
    
    # 為了避免 cache 問題，我們直接複製 parse_query 的核心邏輯但用增強後的查詢
    # 或者我們可以清除 cache 後再呼叫
    parse_query.cache_clear()  # 清除 cache 確保用增強後的查詢
    
    result = parse_query(enhanced_query, model_id=model_id)
    
    # 確保 original_query 保持為用戶原始輸入
    # 由於 QueryParseResult 是 Pydantic model，我們需要創建新的
    from src.models.schemas import QueryParseResult as QPR
    final_result = QPR(
        original_query=user_query,  # 保持原始查詢
        search_terms=result.search_terms,
        generated_keywords=result.generated_keywords,
        hypothetical_intro=result.hypothetical_intro,
        criteria=result.criteria
    )
    
    return final_result, tool_context
