#!/usr/bin/env python3
"""
生成書名+簡介向量 (novels collection)

此腳本從資料庫讀取所有書籍，生成書名+簡介的語意向量。
與novels_fused不同，這裡不包含標籤信息。

輸出：將向量存儲到novels collection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database
from src.core.vector_store import VectorStore


def main():
    print("=" * 70)
    print("生成書名+簡介向量 (novels collection)")
    print("=" * 70)
    
    # Initialize resources
    print("\n[1/3] 初始化資源...")
    db = Database()
    vs = VectorStore(collection_name="novels")
    
    print("      [OK] 資料庫連接成功")
    print("      [OK] Qdrant連接成功 (collection: novels)")
    
    # Read all books
    print("\n[2/3] 讀取所有書籍...")
    all_items = db.get_all_items()
    print(f"      [OK] 讀取 {len(all_items)} 本書籍")
    
    if not all_items:
        print("\n[ERROR] 資料庫中沒有書籍")
        return 1
    
    # Generate vectors for title + intro (NOT including tags)
    print("\n[3/3] 生成書名+簡介向量...")
    print(f"      將為 {len(all_items)} 本書籍生成向量...")
    print("      注意：不包含標籤，僅包含書名和簡介")
    
    try:
        vs.add_items(all_items)
        print("\n" + "=" * 70)
        print("[SUCCESS] 書名+簡介向量生成完成！")
        print("=" * 70)
        print("\n摘要:")
        print(f"  - Collection: novels")
        print(f"  - 向量數量: {len(all_items)}")
        print(f"  - 向量內容: 書名 + 簡介 (不含標籤)")
        print("\n下一步:")
        print("  - 執行 python scripts/generate_tag_embeddings.py")
        print("  - 這將生成標籤向量並與此向量結合")
        print("  - 結果存儲在 novels_multi_vector collection")
        
    except Exception as e:
        print(f"\n[ERROR] 生成向量失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
