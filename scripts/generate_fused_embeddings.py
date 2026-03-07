#!/usr/bin/env python3
"""
離線腳本：生成融合向量（書名 + 標籤 + 簡介）
用法：python scripts/generate_fused_embeddings.py
"""

import sys
from pathlib import Path

# 確保 src 可被導入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database
from src.core.vector_store import VectorStore


def main():
    print("=" * 70)
    print("Fused Embedding Generator")
    print("=" * 70)
    
    # Initialize resources
    print("\n[1/3] Initializing resources...")
    db = Database()
    
    # Use separate collection for fused vectors
    vs_fused = VectorStore(collection_name="novels_fused")
    
    print("      [OK] Database connected")
    print("      [OK] Qdrant connected (collection: novels_fused)")
    
    # Read all books
    print("\n[2/3] Reading all books...")
    all_items = db.get_all_items()
    print(f"      [OK] Read {len(all_items)} books")
    
    if not all_items:
        print("\n[ERROR] No books in database")
        return 1
    
    # Generate fused embeddings
    print("\n[3/3] Generating fused embeddings and storing...")
    print(f"      Will generate fused embeddings for {len(all_items)} books...")
    
    try:
        vs_fused.add_fused_items(all_items)
        print("\n" + "=" * 70)
        print("[SUCCESS] Fused embedding generation completed!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Use VectorStore(collection_name='novels_fused') in searches")
        print("  - Fused embeddings contain: Title + Tags + Introduction")
        print("  - Other scoring rules remain unchanged")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate fused embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
