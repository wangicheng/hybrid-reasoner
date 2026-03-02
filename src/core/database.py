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
                author_nickname TEXT,
                slogan TEXT,
                intro TEXT,
                words_total INTEGER,
                chapters_total INTEGER,
                publish_status TEXT,
                click_count INTEGER,
                bookmark_count INTEGER,
                restricted_age INTEGER,
                is_free BOOLEAN,
                tts BOOLEAN,
                classification TEXT,
                attribute_name TEXT,
                rank_title TEXT,
                tags JSON,
                attributes JSON
            )
        """)
        
        # Migration: Check for new columns and add if missing
        cursor.execute("PRAGMA table_info(novels)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "author_nickname" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN author_nickname TEXT")
        if "attribute_name" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN attribute_name TEXT")
        if "rank_title" not in columns:
            cursor.execute("ALTER TABLE novels ADD COLUMN rank_title TEXT")
        # New columns for rating
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
        # New columns for v2 crawler
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
        
        # Extract fields
        i_id = str(item.get("id"))
        tags = item.get("tags", [])
        if tags and isinstance(tags, dict) and "data" in tags:
            # Handle {"data": [{"name": "tag1"}, ...]}
            tags = [t.get("name") for t in tags["data"]]
        elif tags and isinstance(tags, list) and isinstance(tags[0], dict):
             # Handle [{"name": "tag1"}, ...] (fallback or different format)
             tags = [t.get("name") for t in tags]
             
        # Extraction
        user_info = item.get("user") or {}
        statistic = item.get("statistic") or {}
        classifier = item.get("classification") or {}
        attribute = item.get("attribute") or {}
        
        cursor.execute("""
            INSERT OR REPLACE INTO novels (
                id, name, author, author_nickname, slogan, intro, words_total, chapters_total, 
                publish_status, click_count, bookmark_count, restricted_age, 
                is_free, tts, classification, attribute_name, rank_title, tags, attributes,
                rating_score, rating_count, total_recommendations, source, url, cover_url,
                illname, backupname, is_animated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            i_id,
            item.get("name"),
            user_info.get("name") or item.get("author") or "Unknown",
            user_info.get("nickname"),
            item.get("slogan"),
            item.get("intro"),
            item.get("words_total", 0),
            item.get("chapters_total", 0),
            item.get("publish_status", "unknown"),
            statistic.get("click_count", 0) if isinstance(statistic, dict) else item.get("click_count", 0),
            statistic.get("bookmark_count", 0) if isinstance(statistic, dict) else item.get("bookmark_count", 0),
            item.get("restricted_age", 0),
            item.get("is_free", False),
            item.get("tts", False),
            classifier.get("name") if isinstance(classifier, dict) else str(item.get("classification", "")),
            attribute.get("name") if isinstance(attribute, dict) else None,
            statistic.get("rank_title") if isinstance(statistic, dict) else None,
            json.dumps(tags),
            json.dumps(item),
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
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
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
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
            items.append(d)
        return items

    def search_by_title_fuzzy(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Scan table for title that contains the keyword.
        """
        if not keyword:
            return []
            
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        pattern = f"%{keyword}%"
        cursor.execute("SELECT * FROM novels WHERE name LIKE ?", (pattern,))
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
            items.append(d)
        return items

    def search_by_author(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Search for novels by author name or nickname.
        """
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use LIKE for partial matching
        pattern = f"%{author_name}%"
        cursor.execute("""
            SELECT * FROM novels 
            WHERE author LIKE ? OR author_nickname LIKE ?
        """, (pattern, pattern))
        
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["attributes"] = json.loads(d["attributes"]) if d["attributes"] else {}
            items.append(d)
        return items
