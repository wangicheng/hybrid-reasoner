import sqlite3
import json
from typing import List, Dict, Any, Optional
from src.config import settings

class Database:
    def __init__(self):
        self.path = settings.DB_PATH
        self._init_db()

    def get_connection(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Initial Create
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS novels (
                id TEXT PRIMARY KEY,
                name TEXT,
                author TEXT,
                intro TEXT,
                words_total INTEGER,
                publish_status TEXT,
                click_count INTEGER,
                bookmark_count INTEGER,
                classification TEXT,
                tags JSON,
                rating_score REAL,
                rating_count INTEGER,
                total_recommendations INTEGER,
                source TEXT,
                url TEXT,
                cover_url TEXT,
                illname TEXT,
                backupname TEXT,
                is_animated BOOLEAN
            )
        """)
        
        # Migration: Check for new columns and add if missing
        cursor.execute("PRAGMA table_info(novels)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "rating_score" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN rating_score REAL")
        if "rating_count" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN rating_count INTEGER")
        if "total_recommendations" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN total_recommendations INTEGER")
        if "source" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN source TEXT")
        if "url" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN url TEXT")
        if "cover_url" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN cover_url TEXT")
        if "illname" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN illname TEXT")
        if "backupname" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN backupname TEXT")
        if "is_animated" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN is_animated BOOLEAN")
            
        conn.commit()
        conn.close()

    def add_item(self, item: Dict[str, Any]):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        i_id = str(item.get("id"))
        tags = item.get("tags", [])
        if tags and isinstance(tags, dict) and "data" in tags:
            tags = [t.get("name") for t in tags["data"]]
        elif tags and isinstance(tags, list) and isinstance(tags[0], dict):
             tags = [t.get("name") for t in tags]
             
        cursor.execute("""
            INSERT OR REPLACE INTO novels (
                id, name, author, intro, words_total, 
                publish_status, click_count, bookmark_count, classification, tags,
                rating_score, rating_count, total_recommendations, source, url, cover_url,
                illname, backupname, is_animated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            i_id,
            item.get("name"),
            item.get("author") or "Unknown",
            item.get("intro"),
            item.get("words_total", 0),
            item.get("publish_status", "unknown"),
            item.get("click_count", 0),
            item.get("bookmark_count", 0),
            item.get("classification", ""),
            json.dumps(tags),
            item.get("rating_score"),
            item.get("rating_count", 0),
            item.get("total_recommendations", 0),
            item.get("source"),
            item.get("url"),
            item.get("cover_url"),
            item.get("illname"),
            item.get("backupname"),
            item.get("is_animated", False)
        ))
        conn.commit()
        conn.close()

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM novels WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
            return d
        return None

    def get_all_items(self) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM novels")
        rows = cursor.fetchall()
        conn.close()
        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
            items.append(d)
        return items

    def search_by_title_fuzzy(self, keyword: str) -> List[Dict[str, Any]]:
        if not keyword:
            return []
            
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Strategy 1: direct LIKE
        pattern = f"%{keyword}%"
        cursor.execute("SELECT * FROM novels WHERE name LIKE ?", (pattern,))
        rows = cursor.fetchall()
        
        # Strategy 2: if no results and keyword is long enough,
        # try per-character gap matching (e.g. "為美好世界" → "%為%美%好%世%界%")
        # This handles cases where user drops particles (的/了/之) or slightly misspells
        if not rows and len(keyword) >= 4:
            gap_pattern = "%" + "%".join(keyword) + "%"
            cursor.execute("SELECT * FROM novels WHERE name LIKE ?", (gap_pattern,))
            rows = cursor.fetchall()
        
        conn.close()
        
        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
            items.append(d)
        return items

    def search_by_author(self, author_name: str) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        pattern = f"%{author_name}%"
        cursor.execute("SELECT * FROM novels WHERE author LIKE ?", (pattern,))
        
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
            items.append(d)
        return items

    def search_by_tags_fuzzy(self, tags: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """Search novels that contain any of the specified tag keywords."""
        if not tags:
            return []
            
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build OR condition for multiple tags
        conditions = []
        params = []
        for tag in tags:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')  # Search for the tag within the JSON array string
            
        where_clause = " OR ".join(conditions)
        query = f"SELECT * FROM novels WHERE {where_clause} LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
            items.append(d)
        return items
