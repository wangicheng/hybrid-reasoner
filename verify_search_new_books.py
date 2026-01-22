import sys
import os
import uuid

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.vector_store import VectorStore

def verify():
    print("initializing VectorStore...")
    vs = VectorStore(collection_name="novels")
    
    # Check count
    count_res = vs.client.count(collection_name="novels")
    print(f"\nTotal items in 'novels' collection: {count_res.count}")

    # Check specific ID
    target_original_id = "linovelib_2773"
    target_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, target_original_id))
    print(f"\nChecking existence of {target_original_id} -> UUID: {target_uuid}")
    
    points = vs.client.retrieve(
        collection_name="novels",
        ids=[target_uuid]
    )
    
    if points:
        print("✅ Found point by ID!")
        payload = points[0].payload
        print(f"  Title: {payload.get('name')}")
        print(f"  Source: {payload.get('source')}")
        print(f"  Tags: {payload.get('tags')}")
    else:
        print("❌ Point not found by ID.")

    # 1. Search for a keyword that appears in the new books
    query = "回復術士" 
    print(f"\nSearching for '{query}'...")
    query = "敗北女角" 
    print(f"\nSearching for '{query}'...")
    results = vs.search(query, limit=5)
    
    found_target = False
    for r in results:
        payload = r.get('payload', {})
        score = r.get('score', 0)
        title = payload.get('name', 'Unknown')
        source = payload.get('source', 'unknown')
        tags = payload.get('tags', [])
        
        print(f"  - [{score:.4f}] {title} (Source: {source})")
        print(f"    Tags: {tags}")
        
        if "敗北女角" in title:
            found_target = True
            
    if found_target:
        print("\n✅ Verification 1 Passed: Found '敗北女角' in search results.")
    else:
        print("\n❌ Verification 1 Failed: Did not find '敗北女角'.")

    # 2. Search for a generic term "異世界" and check if new source appears
    query = "異世界"
    print(f"\nSearching for '{query}'...")
    results = vs.search(query, limit=10)
    
    found_new_source = False
    for r in results:
        payload = r.get('payload', {})
        source = payload.get('source', 'unknown')
        if source == 'linovelib':
            title = payload.get('name', 'Unknown')
            score = r.get('score', 0)
            print(f"  - [{score:.4f}] {title} (Source: {source})")
            found_new_source = True
            
    if found_new_source:
        print("\n✅ Verification 2 Passed: Found 'linovelib' books in '異世界' search.")
    else:
        print("\n❌ Verification 2 Failed: No 'linovelib' books found for '異世界'.")

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test "回復術士"
    query = "回復術士"
    # Construct text exactly as VectorStore does
    # Payload from previous run:
    # Title: 回復術士的重啟人生
    # Source: linovelib
    # Tags: ['角川文庫', '二次元', '輕小說', '奇幻', '冒險', '轉生', '後宮']
    
    # We don't have author/slogan/intro easily available here without fetching payload from verifying script
    # But let's assume standard format
    doc_text = "Title: 回復術士的重啟人生\nAuthor: 月夜淚\nTags: 角川文庫, 二次元, 輕小說, 奇幻, 冒險, 轉生, 後宮"
    
    q_emb = model.encode(query)
    d_emb = model.encode(doc_text)
    
    score = util.cos_sim(q_emb, d_emb)
    print(f"\n[Debug] Similarity '{query}' vs Doc: {score.item():.4f}")
    
    verify()