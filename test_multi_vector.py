"""
Test multi-vector search functionality
Tests the fusion of text semantic and tag semantic vectors
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.vector_store import VectorStore

def test_multi_vector_search():
    """Test multi-vector search with tag and text fusion"""
    print("[Test] Starting multi-vector search test...\n")
    
    # Initialize vector store with multi-vector collection
    vs = VectorStore(collection_name="novels_multi_vector")
    
    # Test queries
    test_queries = [
        "adventure knight",
        "mystery romance",
        "fantasy magic",
        "school friendship"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            results, vectors = vs.search_multi_vector(
                query,
                limit=5,
                text_weight=0.7,
                tag_weight=0.3
            )
            
            print(f"Found {len(results)} results\n")
            
            for i, hit in enumerate(results, 1):
                payload = hit.get('payload', {})
                book_name = payload.get('name', 'Unknown')
                book_id = payload.get('id', 'Unknown')
                
                text_score = hit.get('text_score', 0.0)
                tag_score = hit.get('tag_score', 0.0)
                fused_score = hit.get('score', 0.0)
                
                print(f"{i}. {book_name} (ID: {book_id})")
                print(f"   Fused Score: {fused_score:.4f}")
                print(f"   - Text Score: {text_score:.4f} (weight: 0.7)")
                print(f"   - Tag Score: {tag_score:.4f} (weight: 0.3)")
                print()
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("[Test] Multi-vector search test completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_multi_vector_search()
