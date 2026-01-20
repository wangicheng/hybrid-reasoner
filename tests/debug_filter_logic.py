import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.vector_store import VectorStore
from src.core.engine import HybridEngine
from qdrant_client.http import models as rest
from types import SimpleNamespace

def run_debug():
    print("=== Logic Push-down Debug/Verification Script ===")
    
    # 1. Setup Test Collection
    collection_name = "debug_filter_test"
    print(f"1. Initializing VectorStore with collection: {collection_name}")
    vs = VectorStore(collection_name=collection_name)
    
    # 2. Add Dummy Data
    print("2. Adding dummy data...")
    items = [
        {
            "id": 1, "name": "Short Story", "text_content": "A very short story.", 
            "words_total": 5000, "publish_status": "completed", "classification": "Fiction"
        },
        {
            "id": 2, "name": "Long Epic", "text_content": "A very long epic story.", 
            "words_total": 200000, "publish_status": "ongoing", "classification": "Fantasy"
        },
        {
            "id": 3, "name": "Medium Novel", "text_content": "A standard novel length.", 
            "words_total": 60000, "publish_status": "completed", "classification": "Romance"
        }
    ]
    # Note: verify your VectorStore.add_items supports these fields in payload
    # Current add_items implementation in vector_store.py takes raw item as payload.
    # So we just need to ensure fields are there.
    vs.add_items(items)
    print("   Data added.")

    # 3. Test Cases
    engine = HybridEngine()
    
    # helper to mock criteria
    def mock_criteria(name, params):
        return SimpleNamespace(name=name, parameters=params)

    # Case A: Filter by Word Count (> 100,000)
    print("\n--- Test Case A: Word Count > 100,000 ---")
    criteria_list = [
        mock_criteria("numeric_range", {"field": "words_total", "min_val": 100000})
    ]
    
    # Build filter
    q_filter = engine._build_qdrant_filter(criteria_list)
    print(f"Generated Filter: {q_filter}")
    
    # Execute Search (bypasing LLM parse_query, calling vs directly to isolate)
    results = vs.search("story", query_filter=q_filter)
    print(f"Results (Should only be 'Long Epic'): {[r['payload']['name'] for r in results]}")

    # Case B: Filter by Status (Completed)
    print("\n--- Test Case B: Status = Completed ---")
    criteria_list_b = [
        mock_criteria("status_check", {"target_status": "completed"})
    ]
    q_filter_b = engine._build_qdrant_filter(criteria_list_b)
    results_b = vs.search("story", query_filter=q_filter_b)
    print(f"Results (Should be 'Short Story' & 'Medium Novel'): {[r['payload']['name'] for r in results_b]}")

    # Case C: Combined (Completed AND > 10,000 words)
    print("\n--- Test Case C: Completed AND > 10,000 words ---")
    criteria_list_c = [
        mock_criteria("status_check", {"target_status": "completed"}),
        mock_criteria("numeric_range", {"field": "words_total", "min_val": 10000})
    ]
    q_filter_c = engine._build_qdrant_filter(criteria_list_c)
    results_c = vs.search("story", query_filter=q_filter_c)
    print(f"Results (Should be 'Medium Novel'): {[r['payload']['name'] for r in results_c]}")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    run_debug()
