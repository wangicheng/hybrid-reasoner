import os
import json
import google.generativeai as genai
from typing import List, Optional, Dict, Any
from src.models.schemas import QueryParseResult, Criterion

# Configure API Key
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def parse_query(user_query: str) -> QueryParseResult:
    """
    Uses Google Gemini to parse natural language queries into structured criteria.
    Compatible with the flexible Criterion(parameters=Dict) schema.
    """
    if not api_key:
        print("⚠️ Warning: No API Key found. Returning simple keyword search.")
        return QueryParseResult(
            original_query=user_query,
            search_terms=[user_query],
            criteria=[]
        )

    # Use a faster model for parsing
    # You can change this to 'gemini-pro' or 'gemini-1.5-flash' depending on your access
    model_name = os.getenv("LLM_MODEL_ID", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = f"""
    You are a search query parser for a novel recommendation system.
    Analyze the user's input and extract search criteria.

    ### Available Criteria (Name & Parameters):
    1. "numeric_range":
       - field: "words_total" (only supported field)
       - min_val: number (optional)
       - max_val: number (optional)
    
    2. "status_check":
       - target_status: "finished" or "ongoing"
    
    3. "keyword_match":
       - field: "classification" or "tags"
       - keyword: string (e.g., "Fantasy", "Romance")

    4. "author_match":
       - author_name: string

    5. "semantic_similarity":
       - (No parameters needed, use this when the user describes plot/theme)

    ### Input:
    "{user_query}"

    ### Output Format (JSON):
    Return ONLY a raw JSON object (no markdown formatting).
    Structure:
    {{
        "search_terms": ["keyword1", "keyword2"], // Keywords for vector search
        "criteria": [
            {{
                "name": "criteria_name",
                "weight": 1.0,
                "parameters": {{ "param_key": "param_value" }}
            }}
        ]
    }}

    Example: "Find completed fantasy novels over 200k words"
    Output:
    {{
        "search_terms": ["fantasy novel"],
        "criteria": [
            {{ "name": "status_check", "parameters": {{ "target_status": "finished" }} }},
            {{ "name": "numeric_range", "parameters": {{ "field": "words_total", "min_val": 200000 }} }},
            {{ "name": "semantic_similarity", "parameters": {{}} }}
        ]
    }}
    """

    try:
        response = model.generate_content(prompt)
        text_content = response.text
        
        # Clean up potential markdown code blocks (```json ... ```)
        if "```" in text_content:
            text_content = text_content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(text_content)
        
        # Validate and Construct (Automatic convert Dict to Pydantic models)
        return QueryParseResult(**data, original_query=user_query)

    except Exception as e:
        print(f"❌ LLM Parsing Error: {e}")
        # Fallback: Treat as simple keyword search with semantic similarity
        return QueryParseResult(
            original_query=user_query,
            search_terms=[user_query],
            criteria=[
                Criterion(name="semantic_similarity", weight=1.0, parameters={})
            ]
        )
