#!/usr/bin/env python3
"""
Generate and store fused embedding vectors for all books.
Fused text combines: [TITLE] + [TAGS] + [ABSTRACT]
Supports resume from checkpoint and automatic invalid key skipping.
"""

import sys
from pathlib import Path

# 確保 src 可被導入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database
from src.core.vector_store import VectorStore
from src.core.api_utils import get_api_key_rotator
from qdrant_client import QdrantClient
import hashlib


def get_processed_ids(qdrant_path: str) -> set:
    """Get set of already processed item IDs from Qdrant."""
    try:
        client = QdrantClient(path=qdrant_path)
        collection_info = client.get_collection("novels_fused")
        print(f"[INFO] Qdrant already has {collection_info.points_count} items")
        return set()  # We'll skip by checking if embedding exists instead
    except Exception as e:
        print(f"[INFO] No existing collection yet: {e}")
        return set()


def main():
    print("=" * 70)
    print("Fused Embedding Generator v2 (with resume & key management)")
    print("=" * 70)
    
    # Initialize resources
    print("\n[1/3] Initializing resources...")
    db = Database()
    vs_fused = VectorStore(collection_name="novels_fused")
    rotator = get_api_key_rotator()
    
    print(f"      [OK] Database connected")
    print(f"      [OK] Qdrant connected (collection: novels_fused)")
    print(f"      [OK] API Key Rotator initialized with {len(rotator.api_keys)} keys")
    
    # Get all books
    print("\n[2/3] Reading all books...")
    all_items = db.get_all_items()
    print(f"      [OK] Read {len(all_items)} books from database")
    
    if not all_items:
        print("\n[ERROR] No books in database")
        return 1
    
    # Check already processed
    print("\n[3/3] Checking for already processed items...")
    try:
        client = QdrantClient(path=vs_fused.qdrant_path)
        collection_info = client.get_collection("novels_fused")
        already_count = collection_info.points_count
        print(f"      [OK] Already processed: {already_count} items")
        
        # Get the IDs that have been processed
        processed_ids = set()
        scroll_result = client.scroll(collection_name="novels_fused", limit=10000)
        for point in scroll_result[0]:
            processed_ids.add(point.id)
        
        items_to_process = []
        for item in all_items:
            # Convert item id to numeric hash like in add_fused_items
            item_hash = int(hashlib.md5(item['id'].encode()).hexdigest(), 16) % (2**63)
            if item_hash not in processed_ids:
                items_to_process.append(item)
        
        print(f"      [OK] Items to process: {len(items_to_process)}")
    except Exception as e:
        print(f"      [WARNING] Could not get processed items: {e}")
        items_to_process = all_items
        print(f"      [INFO] Will process all {len(items_to_process)} items")
    
    if not items_to_process:
        print("\n[SUCCESS] All items already processed!")
        return 0
    
    # Generate fused embeddings with retry logic
    print(f"\n[4/4] Generating fused embeddings for {len(items_to_process)} items...")
    print("-" * 70)
    
    invalid_keys = set()
    batch_size = 50
    
    try:
        for batch_idx in range(0, len(items_to_process), batch_size):
            batch = items_to_process[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            total_batches = (len(items_to_process) + batch_size - 1) // batch_size
            
            # Retry logic for this batch
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    print(f"[Batch {batch_num}/{total_batches}] Processing {len(batch)} items (attempt {attempt + 1}/{max_retries})...")
                    vs_fused.add_fused_items(batch)
                    print(f"[Batch {batch_num}/{total_batches}] ✓ Successfully uploaded {len(batch)} vectors")
                    break  # Success, move to next batch
                    
                except Exception as e:
                    error_str = str(e)
                    current_key_idx = rotator.current_index
                    
                    if "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str:
                        invalid_keys.add(current_key_idx)
                        print(f"[Batch {batch_num}/{total_batches}] ✗ Key {current_key_idx} invalid (ID: {rotator.api_keys[current_key_idx][:20]}...)")
                        
                        # Try to find a valid key
                        rotator.rotate()
                        new_key_idx = rotator.current_index
                        
                        if attempt < max_retries - 1:
                            print(f"[Batch {batch_num}/{total_batches}] → Trying key {new_key_idx}...")
                            continue
                        else:
                            print(f"[Batch {batch_num}/{total_batches}] ✗ Failed after {max_retries} retries with different keys")
                            raise
                    elif "RESOURCE_EXHAUSTED" in error_str:
                        # Rate limit, just rotate and retry
                        rotator.rotate()
                        new_key_idx = rotator.current_index
                        print(f"[Batch {batch_num}/{total_batches}] ⏱ Rate limited. Rotating to key {new_key_idx}...")
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(5)
                            continue
                        else:
                            raise
                    else:
                        # Other error
                        print(f"[Batch {batch_num}/{total_batches}] ✗ Unexpected error: {error_str[:100]}")
                        raise
        
        print("-" * 70)
        print("\n[SUCCESS] Fused embedding generation completed!")
        print(f"          Processed: {len(items_to_process)} new items")
        if invalid_keys:
            print(f"          Invalid keys: {sorted(invalid_keys)}")
            print(f"          Invalid key IDs: {[rotator.api_keys[i][:20] + '...' for i in sorted(invalid_keys)]}")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print("\n[ERROR] Fused embedding generation failed!")
        print(f"        Error: {e}")
        if invalid_keys:
            print(f"        Invalid keys found: {sorted(invalid_keys)}")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
