"""
Debug Qdrant Content
"""
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.vector_store import VectorStore
from qdrant_client.http import models as rest

def main():
    print("=== Debugging Qdrant via VectorStore ===")
    vs = VectorStore(collection_name="novels")
    
    # 1. Check Total Count
    try:
        count_res = vs.client.count(collection_name="novels")
        print(f"Total Items in Qdrant: {count_res.count}")
    except Exception as e:
        print(f"Error checking count: {e}")
        return

    # 2. Check Payload Structure of a Sample
    print("\n--- Sample Item ---")
    results, next_page = vs.client.scroll(
        collection_name="novels",
        limit=1,
        with_payload=True
    )
    if results:
        payload = results[0].payload
        print(f"ID: {results[0].id}")
        print(f"Name: {payload.get('name')}")
        print(f"Classification Payload: {payload.get('classification')}")
    else:
        print("No items found using scroll.")
        return

    # 3. Test Filter: classification.name = "奇幻"
    target_genre = "奇幻"
    print(f"\n--- Testing Filter: classification.name = '{target_genre}' ---")
    
    filter_ = rest.Filter(
        must=[
            rest.FieldCondition(
                key="classification.name",
                match=rest.MatchValue(value=target_genre)
            )
        ]
    )
    
    try:
        filtered_count = vs.client.count(
            collection_name="novels",
            count_filter=filter_
        ).count
        print(f"Count for '{target_genre}': {filtered_count}")
    except Exception as e:
        print(f"Error testing filter: {e}")

    # 4. Test Filter: tags
    print(f"\n--- Testing Filter: tags (any) = '魔法' ---")
    # Note: Depending on how tags are stored (list of strings or list of objects), we check simple match
    # Based on json: "tags": {"data": [{"name": "..."}]} -> serialized?
    # Wait, db.add_item serializes tags to JSON (string).
    # But VectorStore ingestion uses `item` direct from JSON load?
    # books_crawled.json has "tags": { "data": [ ... ] }
    # So searching `tags` via keyword match might be hard if it's a complex object.
    
    if results:
        print(f"Tags Payload Type: {type(payload.get('tags'))}")
        print(f"Tags Payload: {payload.get('tags')}")

if __name__ == "__main__":
    main()
