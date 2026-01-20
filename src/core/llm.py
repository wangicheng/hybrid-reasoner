from openai import OpenAI
import json
from src.config import settings
from src.models.schemas import QueryParseResult

def parse_query(user_query: str) -> QueryParseResult:
    """
    Parses a natural language query into structured scoring criteria using an OpenAI-compatible LLM.
    """
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL
    )

    prompt = f"""
    You are a web novel recommendation assistant. Your goal is to break down the user's query into weighted scoring criteria.
    
    ### Database Schema (Available Fields)
    Use these fields to filter results precisely.
    
    | Field Name       | Type    | Description                                      | Example Values                  |
    |------------------|---------|--------------------------------------------------|---------------------------------|
    | name             | Text    | The title of the novel                           | "My Hero", "Love Story"         |
    | author           | Text    | Name of the writer                               | "JK Rowling", "Unknown"         |
    | author_nickname  | Text    | Pen name or nickname of the writer               | "阿亞梅", "溫菊"                |
    | classification   | Text    | Main genre or category                           | "Fantasy", "Romance", "Horror"  |
    | attribute_name   | Text    | Book type attribute                              | "長篇小說", "中篇小說"          |
    | rank_title       | Text    | Ranking information (if available)               | "週書單排行榜 第 11 名"         |
    | tags             | List    | Specific themes or elements                      | ["magic", "school", "betrayal"] |
    | words_total      | Number  | Total word count                                 | 10000, 500000                   |
    | publish_status   | Text    | Publication status                               | "ongoing", "completed"          |
    | is_free          | Boolean | Whether the book is free to read                 | true, false                     |
    | tts              | Boolean | Text-to-speech / Audio availability              | true, false                     |
    | click_count      | Number  | Total clicks/views (Popularity)                  | 100, 50000                      |
    | bookmark_count   | Number  | Total bookmarks (Popularity)                     | 5, 1000                         |
    | restricted_age   | Number  | Age restriction (0=All ages, 18=Adult)           | 0, 18                           |

    ### Available Scoring Functions
    Construct a list of scoring criteria using these functions.
    
    1. **keyword_match** (parameters: `field`, `keyword`)
       - Use when the user targets a specific attribute (e.g., "romance novels", "tag: magic").
       - Valid `field`s: 'classification', 'tags', 'name', 'author', 'intro', 'slogan', 'attribute_name', 'rank_title'.
    2. **numeric_range** (parameters: `field`, `min_val`, `max_val`)
       - Use for quantitative constraints (e.g., "long novels", "popular books").
       - Valid `field`s: 'words_total', 'click_count', 'bookmark_count'.
    3. **status_check** (parameters: `target_status`)
       - Options: 'completed', 'ongoing'.
    4. **author_match** (parameters: `author_name`)
       - Specific author search. Matches against both 'author' and 'author_nickname'.
    5. **is_free_check** (parameters: `require_free`)
       - Set to true if "free" is mentioned.
    6. **age_check** (parameters: `allow_restricted`)
       - Set to true ONLY if explicit/R18 content is requested. Default false.
    7. **audio_available** (parameters: `require_audio`)
       - Set to true if "audiobook" or "listening" is mentioned.
    8. **semantic_similarity** (parameters: `query_text`)
       - Use this for abstract descriptions, plot details, or "vibes" that don't match specific metadata fields.
       - Example: "stories about overcoming grief", "political intrigue in a fantasy world".

    ### Strategy Guidelines
    - **Prioritize Database Fields**: If the user's intent clearly maps to a DB field (e.g., "Romance" -> `classification`, "Completed" -> `publish_status`), ALWAYS use the corresponding structured function (`keyword_match`, `status_check`) with a high weight.
    - **Use Semantic Search as Backup**: Use `semantic_similarity` for the remaining nuance or descriptive parts of the query.
    - **Example**: "Completed romance novels with magic" -> 
      - `status_check(completed)` (High weight)
      - `keyword_match(classification="Romance")` (High weight)
      - `keyword_match(tags="magic")` (Medium weight) OR `semantic_similarity("magic")`

    User Query: "{user_query}"
    
    Output a JSON object satisfying the QueryParseResult schema.
    Ensure weights sum to roughly 1.0.
    
    Examples:
    - "Short romance novels by 蔡芳紜": 
      {{
        "original_query": "Short romance novels by 蔡芳紜",
        "search_terms": ["romance novels", "蔡芳紜"],
        "criteria": [
            {{ "name": "keyword_match", "weight": 0.3, "description": "Looking for romance novels", "parameters": {{ "field": "classification", "keyword": "愛情" }} }},
            {{ "name": "numeric_range", "weight": 0.2, "description": "Short novels definition", "parameters": {{ "field": "words_total", "max_val": 50000 }} }},
            {{ "name": "author_match", "weight": 0.5, "description": "Specific author preference", "parameters": {{ "author_name": "蔡芳紜" }} }}
        ]
      }}
    - "Free audiobooks for kids":
      {{
        "original_query": "Free audiobooks for kids",
        "search_terms": ["free audiobooks", "kids"],
        "criteria": [
            {{ "name": "is_free_check", "weight": 0.3, "description": "User wants free books", "parameters": {{ "require_free": true }} }},
            {{ "name": "audio_available", "weight": 0.4, "description": "User wants audio format", "parameters": {{ "require_audio": true }} }},
            {{ "name": "age_check", "weight": 0.3, "description": "Content suitable for kids", "parameters": {{ "allow_restricted": false }} }}
        ]
      }}
    """

    try:
        completion = client.beta.chat.completions.parse(
            model=settings.LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Generate the output according to the schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=QueryParseResult,
        )

        parsed_result = completion.choices[0].message.parsed
        
        # Fallback/Validation if parsed_result is somehow None (though parse() usually raises on failure)
        if not parsed_result:
            raise ValueError("Failed to parse structured output")

        # Ensure defaults if needed (Pydantic usually handles this, but logic retention)
        # However, parsed_result is a distinct object. We can return it directly.
        return parsed_result

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return QueryParseResult(
            original_query=user_query,
            criteria=[],
            search_terms=[user_query]
        )
