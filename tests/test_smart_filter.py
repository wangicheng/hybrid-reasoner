import sys
import os

# Adds the project root to the path to ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from src.core.engine import HybridEngine
from src.core.llm import parse_query

def test_smart_filter_and_relaxation():
    print("🚀 Initializing HybridEngine...")
    engine = HybridEngine()

    # Test Case 1: Smart Filter (Targeting Tags)
    # Goal: Verify that searching for a keyword that exists in Tags but NOT in Classification returns results.
    query_1 = "推薦網遊小說"  # Assuming '網遊' is in tags, not necessarily classification
    print("\n" + "="*80)
    print(f"🔍 Test 1: Smart Filter verification - Query: 【 {query_1} 】")
    print("="*80)

    result_1 = engine.search(query_1, limit=3)
    
    print(f"   🤖 Parsed Criteria:")
    for c in result_1['parsed_criteria']:
        print(f"      - {c}")

    is_relaxed = result_1.get('is_relaxed', False)
    if result_1['results']:
        if is_relaxed:
             print(f"   ⚠️ Smart Filter found 0 results, but Auto-Relaxation saved the day! Found {len(result_1['results'])} results.")
        else:
             print(f"   ✅ Smart Filter worked! Found {len(result_1['results'])} results (Exact Match).")
        
        for item in result_1['results']:
            book = item['item']
            cls = book.get('classification')
            if isinstance(cls, dict):
                cls_name = cls.get('name')
            else:
                cls_name = str(cls)
                
            print(f"      - Book: {book.get('name')} | Class: {cls_name} | Tags: {book.get('tags')}")
    else:
        print("   ❌ Smart Filter failed (or no books with tag '網遊').")

    # Test Case 2: Automatic Relaxation
    # Goal: Verify that searching for a non-existent classification triggers relaxation.
    query_2 = "推薦火星文體小說" # A nonsensical genre
    print("\n" + "="*80)
    print(f"🔍 Test 2: Automatic Relaxation verification - Query: 【 {query_2} 】")
    print("="*80)
    
    result_2 = engine.search(query_2, limit=3)

    print(f"   🤖 Parsed Criteria:")
    for c in result_2['parsed_criteria']:
        print(f"      - {c}")
        
    if result_2.get('is_relaxed'):
        print(f"   ✅ Automatic Relaxation triggered! (is_relaxed=True)")
        print(f"   Found {len(result_2['results'])} results (semantic match only).")
    else:
        if not result_2['results']:
            print("   ❌ Relaxation failed to trigger (still 0 results).")
        else:
            print("   ⚠️  Results found WITHOUT relaxation? (Maybe '火星文體' actually exists?)")
            
    # Print one result to see if it makes sense contextually
    if result_2['results']:
        top_book = result_2['results'][0]['item']
        print(f"      - Top Result: {top_book.get('name')} (Score: {result_2['results'][0]['score']:.4f})")

if __name__ == "__main__":
    test_smart_filter_and_relaxation()
