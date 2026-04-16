import sqlite3
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings


@lru_cache(maxsize=1)
def _load_allowed_tags() -> set[str]:
    tags_path = Path(__file__).resolve().parents[2] / "data" / "all_tags.json"
    try:
        with tags_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with tags_path.open("r", encoding="utf-16") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Tag metadata file '{tags_path}' not found.") from exc

    if not isinstance(data, list):
        raise RuntimeError(
            f"Tag metadata file '{tags_path}' has an unexpected format."
        )

    return {str(tag).strip() for tag in data if str(tag).strip()}


def _normalize_tag_list(raw_tags: Any) -> List[str]:
    if isinstance(raw_tags, str):
        try:
            raw_tags = json.loads(raw_tags)
        except Exception:
            raw_tags = [raw_tags]

    if isinstance(raw_tags, dict) and "data" in raw_tags:
        raw_tags = raw_tags["data"]

    if not isinstance(raw_tags, list):
        return []

    allowed = _load_allowed_tags()
    normalized: List[str] = []
    seen = set()
    for tag in raw_tags:
        if isinstance(tag, dict):
            value = str(tag.get("name", "")).strip()
        else:
            value = str(tag).strip()

        if not value or value not in allowed or value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return normalized

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
        tags = _normalize_tag_list(item.get("tags", []))
             
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
            d["tags"] = _normalize_tag_list(d["tags"]) if d.get("tags") else []
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
            d["tags"] = _normalize_tag_list(d["tags"]) if d.get("tags") else []
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
            d["tags"] = _normalize_tag_list(d["tags"]) if d.get("tags") else []
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
            d["tags"] = _normalize_tag_list(d["tags"]) if d.get("tags") else []
            items.append(d)
        return items

    def search_by_tags_any(self, tags: List[str], limit: int = 10000) -> List[Dict[str, Any]]:
        normalized_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        if not normalized_tags:
            return []

        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        clauses = []
        params: List[Any] = []
        for tag in normalized_tags:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        sql = f"SELECT * FROM novels WHERE {' OR '.join(clauses)} LIMIT ?"
        params.append(limit)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            d = dict(row)
            d["tags"] = _normalize_tag_list(d["tags"]) if d.get("tags") else []
            items.append(d)
        return items


class BM25Index:
    """
    以 rank_bm25 (Okapi BM25) 在內存中對 novels 資料表建立索引。
    使用 jieba 進行中文分詞，建立三個獨立索引：title / intro / author。

    Usage:
        db = Database()
        bm25 = BM25Index(db)          # 自動建立索引
        results = bm25.search_title("為美好世界獻上祝福", top_k=10)
        # -> [{"item": {...}, "bm25_score": 12.34}, ...]
    """

    # 中文常見停用詞
    _STOPWORDS = frozenset({
        '的', '了', '和', '是', '就', '都', '而', '及', '與', '著',
        '或', '一個', '沒有', '我們', '你們', '他們', '她們', '它們',
        '這', '那', '在', '有', '不', '人', '也', '到', '說', '要',
        '會', '對', '上', '下', '把', '讓', '被', '從', '去', '又',
        '很', '過', '之', '所', '能', '為', '以', '如', '更', '但',
        '卻', '才', '啊', '吧', '呢', '嗎', '喔', '哦', '耶', '呀',
    })

    def __init__(self, db: 'Database'):
        self.db = db
        self._items: List[Dict[str, Any]] = []

        # BM25 索引實例
        self._title_index = None
        self._intro_index = None
        self._author_index = None

        self.build()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """使用 jieba 分詞 + 停用字過濾 + 小寫化。"""
        if not text:
            return []

        tokens = []
        for word in jieba.cut(text):
            word = word.strip().lower()
            if len(word) < 1:
                continue
            if word in BM25Index._STOPWORDS:
                continue
            tokens.append(word)
        return tokens

    def build(self) -> None:
        """從 DB 讀取所有 novels 並建立 BM25 索引。"""
        start = time.perf_counter()

        self._items = self.db.get_all_items()

        title_corpus: List[List[str]] = []
        intro_corpus: List[List[str]] = []
        author_corpus: List[List[str]] = []

        for item in self._items:
            # 書名索引：合併 name + backupname + illname
            title_parts = ' '.join(filter(None, [
                item.get('name', ''),
                item.get('backupname', ''),
                item.get('illname', ''),
            ]))
            title_corpus.append(self._tokenize(title_parts))

            # 簡介索引
            intro_corpus.append(self._tokenize(item.get('intro', '') or ''))

            # 作者索引
            author_corpus.append(self._tokenize(item.get('author', '') or ''))

        # 建立 BM25 索引（空語料庫時使用空列表避免錯誤）
        self._title_index = BM25Okapi(title_corpus) if title_corpus else None
        self._intro_index = BM25Okapi(intro_corpus) if intro_corpus else None
        self._author_index = BM25Okapi(author_corpus) if author_corpus else None

        elapsed = time.perf_counter() - start
        print(f"[BM25Index] Built indexes for {len(self._items)} novels in {elapsed:.2f}s")

    def rebuild(self) -> None:
        """重建索引（資料變更後呼叫）。"""
        self.build()

    def _search(self, index, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """通用搜尋方法：對指定索引進行 BM25 查詢。"""
        if index is None or not self._items:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = index.get_scores(query_tokens)

        # 取 top_k 且 score > 0
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            results.append({
                'item': self._items[idx],
                'bm25_score': score,
            })

        return results

    def search_title(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 書名搜尋。"""
        return self._search(self._title_index, query, top_k)

    def search_intro(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 簡介/內容搜尋。"""
        return self._search(self._intro_index, query, top_k)

    def search_author(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 作者搜尋。"""
        return self._search(self._author_index, query, top_k)

