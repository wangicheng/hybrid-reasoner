import sys
import os
from pathlib import Path

# Add project root to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.vector_store import VectorStore
from src.config import settings

def test_vector_search():
    print("=== Testing Search by Keyword (Vector Search) ===")
    
    # 1. Initialize Vector Store (The 'Who')
    print("Initializing VectorStore...")
    try:
        vs = VectorStore(collection_name="novels")
    except Exception as e:
        print(f"Error initializing VectorStore: {e}")
        return

    # 2. Define a keyword to search for
    keyword = "轉生"  # A common trope in light novels
    print(f"\nTarget Keyword: '{keyword}'")

    # 3. Simulate the 'How': Convert keyword to vector
    print("Converting keyword to vector...")
    # We can access the model directly to show the vector details
    vector = vs.model.encode(keyword)
    print(f"Vector generated! Dimension: {len(vector)}")
    print(f"First 5 dimensions: {vector[:5]}...")

    # 4. Perform the search
    print(f"\nSearching Qdrant for similar items...")
    results, query_vector = vs.search(keyword, limit=3)

    # 5. Show results
    print(f"\nFound {len(results)} results:")
    for i, res in enumerate(results, 1):
        payload = res["payload"]
        score = res["score"]
        title = payload.get("name", "Unknown")
        desc = payload.get("description", "")[:50] + "..."
        print(f"{i}. [Score: {score:.4f}] {title} - {desc}")

if __name__ == "__main__":
    test_vector_search()
