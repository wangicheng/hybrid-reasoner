from src.core.engine import HybridEngine
import sys

def debug_search():
    print("Initializing HybridEngine...")
    try:
        engine = HybridEngine()
        print("HybridEngine initialized.")
    except Exception as e:
        print(f"Failed to init engine: {e}")
        return

    query = "阿亞梅的書"
    print(f"Running search for: {query}")
    
    try:
        result = engine.search(query, limit=5)
        print("Search completed.")
        print(f"Parsed Criteria: {result['parsed_criteria']}")
        print("Results:")
        for res in result['results']:
            print(f"- {res['item']['name']} (Score: {res['score']})")
            print(f"  Author: {res['item']['author']} / {res['item'].get('author_nickname')}")
            for b in res['breakdown']:
                print(f"    - {b}")
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Search failed: {e}")

if __name__ == "__main__":
    debug_search()
