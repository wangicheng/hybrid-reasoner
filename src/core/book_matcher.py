"""
Book Title Matcher — 三層策略從使用者查詢中識別書名並取得標籤

Layer 1: LLM 識別的 reference_books → difflib 模糊比對
Layer 2: 書名號《》「」包裹的書名 → LIKE + gap + difflib
Layer 3: search_terms (>=4字) → LIKE + gap + difflib

每層都會先嘗試 LIKE 精確匹配，失敗則用 difflib 兜底。
"""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

from src.core.database import Database, BM25Index


class BookMatcher:
    """負責將模糊書名比對到書庫中的正式書名並取得標籤。"""

    def __init__(self, db: Database, bm25_index: Optional[BM25Index] = None):
        self.db = db
        self.bm25_index = bm25_index
        self._title_cache: Optional[List[Dict[str, Any]]] = None

    # ─── Public API ───

    def extract_reference_tags(
        self,
        user_query: str,
        search_terms: str = "",
        reference_books: Optional[List[str]] = None,
        query_intent: Optional[Any] = None,
    ) -> List[str]:
        """
        從使用者查詢提取參考小說的標籤（去重保序）。

        Args:
            user_query:       原始使用者輸入
            search_terms:     LLM 提取的搜尋關鍵詞
            reference_books:  LLM 明確標識的書名列表
            query_intent:     意圖解析結果，用於避開負面書名

        Returns:
            合併後的標籤清單
        """
        extracted_tags: List[str] = []
        found_books: set = set()
        
        # 建立負面排除清單（來自 LLM 解析的不要的書 / 概念）
        exclusions = set()
        if query_intent:
            if hasattr(query_intent, "hard_exclusions"):
                exclusions.update(e.term for e in query_intent.hard_exclusions)
            if hasattr(query_intent, "soft_exclusions"):
                exclusions.update(e.term for e in query_intent.soft_exclusions)

        def _try(title: str, source: str):
            # 檢查是否在負面清單中，如果包含負面詞則跳過
            for exc in exclusions:
                if title in exc or exc in title:
                    print(f"[BookMatcher] 忽略負面參考書: {title} (Matches exclusion: {exc})")
                    return
            self._try_match_book(title, source, extracted_tags, found_books)

        # Layer 1: LLM reference_books（最高優先級，LLM 已被要求過濾負面教材）
        for title in (reference_books or []):
            _try(title, "LLM reference_books")

        # Layer 2: 書名號包裹 (加上負面過濾)
        for pattern in [r'《([^》]+)》', r'「([^」]+)」', r'『([^』]+)』', r'"([^"]+)"']:
            for match in re.findall(pattern, user_query):
                _try(match, "書名號")

        # Layer 3 (已移除): 原本使用 search_terms 做模糊匹配，但會導致如 "戰爭 戀愛" 匹配到毫不相干的搞笑作品《歸宅戰爭》，引發嚴重的字面陷阱。

        return list(dict.fromkeys(extracted_tags))

    def fuzzy_match_title(self, query_title: str, threshold: float = 0.55) -> Optional[Dict[str, Any]]:
        """
        用 difflib SequenceMatcher 尋找最佳匹配書名。

        Returns:
            {"name": str, "tags": list, "score": float} 或 None
        """
        if not query_title or len(query_title) < 2:
            return None

        cache = self._load_title_cache()
        query_lower = query_title.lower().strip()

        best_match = None
        best_ratio = 0.0

        for book in cache:
            book_name = book["name"]
            if not book_name or len(book_name) < 2:
                continue
            name_lower = book_name.lower().strip()

            if query_lower in name_lower or name_lower in query_lower:
                ratio = 0.95
            else:
                ratio = SequenceMatcher(None, query_lower, name_lower).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = book

        if best_ratio >= threshold and best_match:
            return {"name": best_match["name"], "tags": best_match["tags"], "score": best_ratio}
        return None

    # ─── Internal helpers ───

    def _try_match_book(
        self,
        title: str,
        source: str,
        extracted_tags: List[str],
        found_books: set,
    ):
        """嘗試匹配書名：BM25 → LIKE → difflib。"""
        if not title or len(title) < 2:
            return

        # 1) BM25 書名搜尋（優先）
        if self.bm25_index is not None:
            bm25_results = self.bm25_index.search_title(title, top_k=3)
            if bm25_results:
                book = bm25_results[0]['item']
                bm25_score = bm25_results[0]['bm25_score']
                if book.get('name') and book['name'] not in found_books:
                    found_books.add(book['name'])
                    tags = book.get('tags', [])
                    if tags:
                        print(f"[BookMatcher] {source} BM25匹配: 《{book['name']}》 (query='{title}', bm25={bm25_score:.2f})")
                        print(f"[BookMatcher] 提取標籤: {', '.join(tags[:8])}")
                        extracted_tags.extend(tags)
                return

        # 2) LIKE + gap LIKE (fallback)
        results = self.db.search_by_title_fuzzy(title)
        if results:
            book = results[0]
            if book["name"] not in found_books:
                found_books.add(book["name"])
                tags = book.get("tags", [])
                if tags:
                    print(f"[BookMatcher] {source} LIKE匹配: 《{book['name']}》 (query='{title}')")
                    print(f"[BookMatcher] 提取標籤: {', '.join(tags[:8])}")
                    extracted_tags.extend(tags)
            return

        # 3) difflib 兜底
        match = self.fuzzy_match_title(title)
        if match and match["name"] not in found_books:
            found_books.add(match["name"])
            tags = match.get("tags", [])
            if tags:
                print(f"[BookMatcher] {source} difflib匹配: 《{match['name']}》 (query='{title}', score={match['score']:.2f})")
                print(f"[BookMatcher] 提取標籤: {', '.join(tags[:8])}")
                extracted_tags.extend(tags)

    def _load_title_cache(self) -> List[Dict[str, Any]]:
        """懶載入所有書名+標籤到記憶體。"""
        if self._title_cache is not None:
            return self._title_cache

        conn = self.db.get_connection()
        import sqlite3, json
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, tags FROM novels")
        rows = cursor.fetchall()
        conn.close()

        self._title_cache = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
            self._title_cache.append(d)
        print(f"[BookMatcher] Title cache loaded: {len(self._title_cache)} books")
        return self._title_cache
