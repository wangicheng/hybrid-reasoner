import os
import json
from google import genai
from google.genai import types
from src.models.schemas import QueryParseResult

def parse_query(user_query: str) -> QueryParseResult:
    """
    使用 Google GenAI SDK (v1.0+) 將自然語言查詢轉換為結構化搜尋條件。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    model_id = os.getenv("LLM_MODEL_ID", "gemini-2.0-flash")

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
    **IMPORTANT**: Distinguish between hard filters (numeric_range) and soft ranking (numeric_ranking). If the user says "I want a VERY long novel" without specifying a number, use numeric_ranking. If they say "at least 500k words", use numeric_range.

    ### IMPORTANT: DATASET LANGUAGE
    The underlying database uses **Traditional Chinese (繁體中文)** for all metadata (names, tags, classifications).
    - For `keyword_match` on fields like `classification` or `tags`, you **MUST** output the `keyword` in Traditional Chinese.
    - **Translate** user limits (e.g. "Fantasy" -> "奇幻", "Magic" -> "魔法", "Romance" -> "言情").
    - If the user queries in Chinese, use the exact Chinese terms they used (or synonyms).
    - Common Classifications: 奇幻, 言情, 都市, 玄幻, 靈異, 武俠, 科幻.
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=f"User Query: {user_query}",
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                response_mime_type="application/json",
                response_schema=manual_schema
            )
        )
        
        # New SDK returns an object, we access .text or specific fields.
        # For structured output, response.text is the JSON string.
        if response.text:
            result = QueryParseResult.model_validate_json(response.text)
            return result
        else:
             raise ValueError("Empty response from Gemini")

    except Exception as e:
        print(f"Error parsing query with Google GenAI: {e}")
        return QueryParseResult(original_query=user_query, search_terms=[user_query], criteria=[])
