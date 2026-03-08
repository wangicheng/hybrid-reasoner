"""
生成標籤向量 (Tag Semantic Embeddings)

此腳本為每本書的標籤生成向量，使用與fused embeddings相同的API key rotation機制。
標籤向量會與現有的文本向量合併到同一個Named Vectors集合中。

輸出：將標籤向量與現有文本向量結合，存儲到novels_multi_vector集合
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database
from src.core.vector_store import VectorStore
from src.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from google import genai
from google.genai import types
from src.core.api_utils import get_current_api_key, get_api_key_rotator
import time


def build_tag_text(item: dict) -> str:
    """
    構建標籤文本用於向量化。
    
    Args:
        item: 包含tags的書籍信息字典
        
    Returns:
        拼接後的標籤文本
    """
    tags_raw = item.get('tags', [])
    if isinstance(tags_raw, str):
        try:
            tags_raw = json.loads(tags_raw)
        except:
            tags_raw = []
    
    if isinstance(tags_raw, list):
        tags_text = ' '.join([str(t).strip() for t in tags_raw if t])
    else:
        tags_text = ''
    
    return tags_text


def generate_tag_embeddings():
    """
    主函數：生成所有書籍的標籤向量
    """
    print("[TagEmbeddings] Start generating tag embeddings...")
    
    # Initialize database and single vector store client
    db = Database()
    
    # Use single QdrantClient to avoid multiple access conflicts
    from pathlib import Path
    qdrant_path = Path(settings.QDRANT_PATH)
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path.resolve()))
    
    # Check if collection exists, create with Named Vectors if not
    collections = client.get_collections().collections
    collection_exists = any(c.name == "novels_multi_vector" for c in collections)
    
    if not collection_exists:
        print("[TagEmbeddings] Creating new Named Vectors collection: novels_multi_vector")
        client.create_collection(
            collection_name="novels_multi_vector",
            vectors_config={
                "text_semantic": rest.VectorParams(
                    size=3072,
                    distance=rest.Distance.COSINE
                ),
                "tag_semantic": rest.VectorParams(
                    size=3072,
                    distance=rest.Distance.COSINE
                ),
            }
        )
    else:
        print("[TagEmbeddings] Collection novels_multi_vector already exists. Continuing to add more vectors...")
    
    # Read all existing text vectors from novels collection
    print("[TagEmbeddings] Reading existing text vectors from novels collection...")
    
    existing_points = {}
    try:
        scroll_point = None
        total_fused = 0
        while True:
            response = client.scroll(
                collection_name="novels",
                offset=scroll_point,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            points, scroll_point = response
            for point in points:
                existing_points[point.id] = {
                    "vector": point.vector,
                    "payload": point.payload
                }
                total_fused += 1
            if scroll_point is None:
                break
    except Exception as e:
        print(f"[TagEmbeddings] Warning: Error reading novels: {e}")
    
    print(f"[TagEmbeddings] Retrieved {total_fused} existing vectors from novels")
    
    # Read all novels from database
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, tags, intro FROM novels ORDER BY id")
    all_novels = cursor.fetchall()
    conn.close()
    
    total_novels = len(all_novels)
    print(f"[TagEmbeddings] Total novels in database: {total_novels}")
    
    # Check existing IDs in novels_multi_vector
    existing_multi_ids = set()
    try:
        scroll_point = None
        while True:
            response = client.scroll(
                collection_name="novels_multi_vector",
                offset=scroll_point,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )
            points, scroll_point = response
            existing_multi_ids.update(p.id for p in points)
            if scroll_point is None:
                break
    except Exception:
        pass
    
    print(f"[TagEmbeddings] Existing items in novels_multi_vector: {len(existing_multi_ids)}")
    
    # Prepare items to process
    items_to_process = []
    tag_texts = []
    
    for novel_id, name, tags_raw, intro in all_novels:
        if novel_id in existing_multi_ids:
            continue
        
        tag_text = build_tag_text({"tags": tags_raw})
        if not tag_text.strip():
            tag_text = name if name else "unknown"
        
        # Skip if we still have no text
        if not tag_text or not tag_text.strip():
            print(f"  SKIP: Item {novel_id} has empty tag and name")
            continue
        
        items_to_process.append({
            "id": novel_id,
            "name": name,
            "intro": intro,
            "tags": tags_raw,
            "tag_text": tag_text
        })
        tag_texts.append(tag_text)
    
    print(f"[TagEmbeddings] Items to process: {len(items_to_process)}")
    
    if not items_to_process:
        print("[TagEmbeddings] All items already processed. Done!")
        return
    
    # Batch generate tag vectors
    batch_size = 50
    api_key = get_current_api_key()
    genai_client = genai.Client(api_key=api_key)
    embedding_model = "gemini-embedding-001"
    
    for i in range(0, len(items_to_process), batch_size):
        batch_items = items_to_process[i:i+batch_size]
        batch_tag_texts = tag_texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        
        print(f"\n[TagEmbeddings] Processing batch {batch_num}/{total_batches} ({len(batch_items)} items)")
        
        # Retry with API key rotation
        attempt = 0
        max_attempts = 5
        tag_embeddings = None
        
        while attempt < max_attempts:
            try:
                print(f"  Attempt {attempt + 1}/{max_attempts}: Calling Gemini API...")
                response = genai_client.models.embed_content(
                    model=embedding_model,
                    contents=batch_tag_texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                tag_embeddings = response.embeddings
                print(f"  SUCCESS: Generated {len(tag_embeddings)} tag vectors")
                break
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "INVALID_ARGUMENT" in error_str
                
                if is_rate_limit and attempt < max_attempts - 1:
                    print(f"  RATE_LIMIT: Rotating API key...")
                    rotator = get_api_key_rotator()
                    new_key = rotator.on_rate_limit_error()
                    genai_client = genai.Client(api_key=new_key)
                    print(f"  Using key index: {rotator.current_index}")
                    attempt += 1
                    time.sleep(5)
                else:
                    print(f"  ERROR: Failed to generate tag vectors: {e}")
                    raise
        
        if not tag_embeddings:
            print(f"  ERROR: Cannot generate vectors for batch {batch_num}. Skipping...")
            continue
        
        # Fix ID conversion and merge with existing text vectors
        points = []
        for item, tag_embedding in zip(batch_items, tag_embeddings):
            str_id = item["id"]
            
            # Convert ID to Qdrant format
            import hashlib
            numeric_id = int(hashlib.md5(str_id.encode()).hexdigest(), 16) % (2**63)
            
            # Prepare payload
            payload = {
                "id": str_id,
                "name": item["name"],
                "intro": item["intro"],
                "tags": json.loads(item["tags"]) if isinstance(item["tags"], str) else item["tags"]
            }
            
            # Get text_semantic vector from existing novels_fused if available
            text_semantic_vector = None
            if numeric_id in existing_points:
                text_semantic_vector = existing_points[numeric_id]["vector"]
            
            # Build Named Vectors structure - use dictionary for vector parameter
            vectors = {
                "tag_semantic": tag_embedding.values
            }
            
            if text_semantic_vector:
                vectors["text_semantic"] = text_semantic_vector
            else:
                print(f"  WARNING: Item {str_id} has no existing text vector. Using zero vector...")
                vectors["text_semantic"] = [0.0] * 3072
            
            # PointStruct accepts vector as Dict for Named Vectors
            point = rest.PointStruct(
                id=numeric_id,
                vector=vectors,
                payload=payload
            )
            points.append(point)
        
        # Upload to novels_multi_vector collection
        if points:
            print(f"  Uploading {len(points)} multi-vector points to Qdrant...")
            client.upsert(
                collection_name="novels_multi_vector",
                points=points
            )
            print(f"  Batch {batch_num} completed")
        
        # Avoid hitting rate limits too fast
        if i + batch_size < len(items_to_process):
            time.sleep(2)
    
    print("\n[TagEmbeddings] SUCCESS: Tag embedding generation completed!")
    print(f"[TagEmbeddings] Collection: novels_multi_vector")
    print(f"[TagEmbeddings] Original collection preserved: novels")


if __name__ == "__main__":
    try:
        generate_tag_embeddings()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
