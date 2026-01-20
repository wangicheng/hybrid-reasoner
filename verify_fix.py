
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path.cwd()))

from src.core.database import Database
from src.core.vector_store import VectorStore
from src.models.schemas import ScoringCriteria, ScoringParameters, QueryParseResult
from src.main import calculate_score, seed_data

def verify():
    print("Starting verification...")
    
    # 1. Initialize
    db = Database()
    vs = VectorStore(collection_name="novels")
    
    # 2. Seed (Crucial step for persistent storage first run)
    print("Seeding data...")
    seed_data(db, vs)
    
    # 3. Setup Search Criteria (Same as reproduction)
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
    
    parse_result = QueryParseResult(
        original_query="romance novels",
        criteria=criteria_list,
        search_terms=["romance", "novels"]
    )
    
    search_terms = " ".join(parse_result.search_terms)
    print(f"Searching for: '{search_terms}'")
    
    # 4. Search
    vector_results = vs.search(search_terms, limit=50)
    print(f"Vector search returned {len(vector_results)} results.")
    
    if not vector_results:
        print("FAIL: No vector results found even after seeding.")
        return

    # 5. Retrieve & Score
    candidates = []
    vector_score_map = {}
    for hit in vector_results:
        item = db.get_item(hit["id"])
        if item:
            candidates.append(item)
            vector_score_map[str(item["id"])] = hit["score"]
            
    print(f"Found {len(candidates)} candidates in DB.")
            
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
        
    if len(scored_items) > 0:
        print("SUCCESS: Recommendations generated.")
    else:
        print("FAIL: No scored items.")

if __name__ == "__main__":
    verify()
