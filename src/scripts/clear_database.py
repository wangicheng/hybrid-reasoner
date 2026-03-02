"""
清理資料庫腳本
重置 SQLite 和 Qdrant 向量庫
"""

import os
import shutil
import sqlite3
from pathlib import Path

# 設定路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "novels.db"
QDRANT_PATH = DATA_DIR / "qdrant_storage"


def clear_sqlite():
    """清空 SQLite 資料庫"""
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 取得現有資料數量
        cursor.execute("SELECT COUNT(*) FROM novels")
        count = cursor.fetchone()[0]
        print(f"SQLite: 現有 {count} 筆資料")
        
        # 清空表格
        cursor.execute("DELETE FROM novels")
        conn.commit()
        
        # 重置 auto-increment (如果有)
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='novels'")
        conn.commit()
        
        # VACUUM 清理空間
        cursor.execute("VACUUM")
        conn.close()
        
        print("SQLite: 已清空所有資料")
    else:
        print("SQLite: 資料庫不存在")


def clear_qdrant():
    """清空 Qdrant 向量庫"""
    if QDRANT_PATH.exists():
        # 列出 collections
        collections_dir = QDRANT_PATH / "collection"
        if collections_dir.exists():
            collections = list(collections_dir.iterdir())
            print(f"Qdrant: 找到 {len(collections)} 個 collections")
            
            # 刪除所有 collection 資料夾
            for coll in collections:
                if coll.is_dir():
                    shutil.rmtree(coll)
                    print(f"  已刪除: {coll.name}")
        
        print("Qdrant: 已清空所有向量資料")
    else:
        print("Qdrant: 儲存路徑不存在")


def clear_all():
    """清空所有資料"""
    print("=" * 50)
    print("開始清理資料庫")
    print("=" * 50)
    
    clear_sqlite()
    print()
    clear_qdrant()
    
    print()
    print("=" * 50)
    print("清理完成！")
    print("=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='清理資料庫')
    parser.add_argument('--sqlite-only', action='store_true',
                        help='只清空 SQLite')
    parser.add_argument('--qdrant-only', action='store_true',
                        help='只清空 Qdrant')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='跳過確認提示')
    
    args = parser.parse_args()
    
    if not args.yes:
        response = input("確定要清空資料庫嗎？此操作無法復原！ (y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    if args.sqlite_only:
        clear_sqlite()
    elif args.qdrant_only:
        clear_qdrant()
    else:
        clear_all()


if __name__ == "__main__":
    main()
