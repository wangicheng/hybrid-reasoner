
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

from src.core.database import Database
from src.core.vector_store import VectorStore
from src.models.schemas import ScoringCriteria, ScoringParameters, QueryParseResult
from src.main import calculate_score

def reproduce():
    print("Starting reproduction...")
    
    # Mocking the parsed result as provided by the user
    # Note: user didn't provide search_terms or original_query, so we infer them
    # or leave them empty to see if that's the issue.
    # The user said "Parsed Criteria" was obtained, so LLM ran.
    
    criteria_list = [
        ScoringCriteria(
            name='status_check', 
            weight=0.5, 
            parameters=ScoringParameters(target_status='completed'), 
            description="Prioritize novels that have a 'completed' status."
        ),
        ScoringCriteria(
            name='keyword_match', 
            weight=0.3, 
            parameters=ScoringParameters(field='classification', keyword='romance'), 
            description="Match novels classified under 'romance'."
        ),
        ScoringCriteria(
            name='semantic_similarity', 
            weight=0.2, 
            parameters=ScoringParameters(query_text='romance novels'), 
            description="Identify novels that are semantically similar to the concept of 'romance novels'."
        )
    ]
    
    # We'll assume search_terms was populated by LLM as well, usually matching the query.
    # If the user says "Parsed Criteria" was correct, the LLM probably worked.
    # Let's try with a reasonable search term derived from criteria.
    
    search_terms_list = ["romance", "novels"]
    original_query = "romance novels"
    
    parse_result = QueryParseResult(
        original_query=original_query,
        criteria=criteria_list,
        search_terms=search_terms_list
    )
    
    print(f"Parsed Criteria: {parse_result.criteria}")
    
    db = Database()
    vs = VectorStore(collection_name="novels")
    
    search_terms = " ".join(parse_result.search_terms) or parse_result.original_query
    print(f"Searching for: '{search_terms}'")
    
    vector_results = vs.search(search_terms, limit=50)
    print(f"Vector search returned {len(vector_results)} results.")
    
    if not vector_results:
        print("No vector results found.")
        return

    # Retrieve full item details
    candidates = []
    vector_score_map = {}
    for hit in vector_results:
        # Check if hit is dict or object, usually dict from vs.search
        # vs.search signature in vector_store.py needs checking
        # Assuming dict based on main.py usage: hit["id"], hit["score"]
        
        item_id = hit.get("id")
        score = hit.get("score")
        
        item = db.get_item(item_id)
        if item:
            candidates.append(item)
            vector_score_map[str(item["id"])] = score
        else:
             print(f"Item {item_id} not found in DB.")
            
    print(f"Found {len(candidates)} candidates in DB.")
            
    # Scoring
    scored_items = []
    for item in candidates:
        v_score = vector_score_map.get(str(item["id"]), 0.0)
        score = calculate_score(item, parse_result.criteria, vs, vector_score=v_score)
        scored_items.append((item, score))
        
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop Recommendations:")
    for item, score in scored_items[:5]:
        status = "完結" if item.get("publish_status") == "completed" else "連載"
        print(f"[{score:.4f}] {item.get('name')} ({status}) - {item.get('classification')}")

if __name__ == "__main__":
    reproduce()
