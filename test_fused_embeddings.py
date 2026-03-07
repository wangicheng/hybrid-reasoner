#!/usr/bin/env python3
"""
測試融合向量功能
"""

import sys
from pathlib import Path

# 確保 src 可被導入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vector_store import VectorStore
from src.core.database import Database


def test_fused_text_generation():
    """測試融合文本生成"""
    print("\n" + "=" * 70)
    print("TEST 1: Fused Text Generation")
    print("=" * 70)
    
    db = Database()
    items = db.get_all_items()
    
    if not items:
        print("[FAIL] No data in database")
        return False
    
    # 測試前 3 個項目
    for i, item in enumerate(items[:3]):
        fused = VectorStore.build_fused_text(item)
        print(f"\nBook {i+1}: {item.get('name', 'N/A')}")
        print(f"  Tags: {item.get('tags', [])}")
        print(f"  Fused Text (first 100 chars):")
        print(f"    {fused[:100]}...")
    
    print("\n[PASS] Fused text generation OK")
    return True


def test_collections_exist():
    """測試 collection 是否存在"""
    print("\n" + "=" * 70)
    print("TEST 2: Qdrant Collection Check")
    print("=" * 70)
    
    vs_novels = VectorStore(collection_name="novels")
    vs_fused = VectorStore(collection_name="novels_fused")
    
    collections = vs_novels.client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    print(f"\nAvailable Collections:")
    for name in collection_names:
        print(f"  - {name}")
    
    print("\n[PASS] Collections check OK")
    return True


def test_fused_search():
    """測試融合向量搜尋能力（在生成融合向量後）"""
    print("\n" + "=" * 70)
    print("TEST 3: Fused Vector Search")
    print("=" * 70)
    
    vs_fused = VectorStore(collection_name="novels_fused")
    
    # 嘗試搜尋
    try:
        results, query_vector = vs_fused.search(
            "fantasy adventure story",
            limit=5,
            with_payload=True
        )
        
        print(f"\nSearch Query: 'fantasy adventure story'")
        print(f"Results: {len(results)} items")
        
        if results:
            for i, result in enumerate(results[:3]):
                print(f"\nResult {i+1}:")
                print(f"  ID: {result['id']}")
                print(f"  Similarity Score: {result['score']:.4f}")
                if result.get('payload'):
                    name = result['payload'].get('name', 'N/A')
                    print(f"  Title: {name}")
        
        print("\n[PASS] Fused vector search OK")
        return True
    except Exception as e:
        print(f"\n[FAIL] Search failed: {e}")
        print("  Note: If no results, may need to run generate_fused_embeddings.py first")
        return False


def main():
    print("\n" + "=" * 70)
    print("Fused Embeddings Test Suite")
    print("=" * 70)
    
    results = []
    
    # TEST 1: 融合文本生成
    try:
        results.append(("Fused Text Generation", test_fused_text_generation()))
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        results.append(("Fused Text Generation", False))
    
    # TEST 2: Collections 檢查
    try:
        results.append(("Collections Check", test_collections_exist()))
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        results.append(("Collections Check", False))
    
    # TEST 3: 融合向量搜尋
    try:
        results.append(("Fused Vector Search", test_fused_search()))
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        results.append(("Fused Vector Search", False))
    
    # 總結
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        print("\nNext Steps:")
        print("  1. Run: python scripts/generate_fused_embeddings.py")
        print("  2. Start search service for actual testing")
        return 0
    else:
        print("\n[ERROR] Some tests failed, check output above")
        return 1


if __name__ == "__main__":
    exit(main())

