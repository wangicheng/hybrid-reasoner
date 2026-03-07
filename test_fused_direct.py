#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct test of fused embedding search functionality."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.vector_store import VectorStore
from src.core.database import Database


def test_fused_search():
    """Test fused embedding vector search."""
    print("=" * 70)
    print("Direct Fused Embedding Search Test")
    print("=" * 70)
    
    # Initialize
    print("\n[1/3] Initializing database and vector store...")
    db = Database()
    vs_fused = VectorStore(collection_name="novels_fused")
    print("      [OK] Initialized")
    
    # Get collection info
    print("\n[2/3] Checking fused collection...")
    try:
        info = vs_fused.client.get_collection("novels_fused")
        print(f"      [OK] Collection: novels_fused")
        print(f"      [OK] Points: {info.points_count}")
    except Exception as e:
        print(f"      [ERROR] {e}")
        return 1
    
    # Test search
    print("\n[3/3] Testing searches...")
    test_queries = [
        ("adventure knight", "Should find action/adventure books"),
        ("love romance", "Should find romance books"),
        ("magic spell wizard", "Should find fantasy books"),
    ]
    
    for query, description in test_queries:
        print(f"\n  Query: '{query}'")
        print(f"  Expected: {description}")
        
        try:
            # Search using VectorStore
            results, _ = vs_fused.search(
                query,
                limit=5,
                with_payload=True
            )
            
            if results:
                print(f"  Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    payload = result.get("payload", {})
                    score = result.get("score", "N/A")
                    
                    title = payload.get("name", "N/A")
                    author = payload.get("author", "N/A")
                    tags = payload.get("tags", "")
                    
                    print(f"\n    {i}. [Score: {score:.4f}]")
                    print(f"       Title: {title}")
                    print(f"       Author: {author}")
                    if tags:
                        if isinstance(tags, str):
                            tags_str = tags[:60]
                        else:
                            tags_str = ", ".join(str(t)[:50] for t in tags)
                        print(f"       Tags: {tags_str}")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(test_fused_search())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
