"""Book matching helpers for soft LLM context only.

This module should not influence retrieval or ranking directly. It finds books
mentioned in the user's query and formats them as context for query parsing.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from src.core.database import Database


class BookMatcher:
    def __init__(
        self,
        db: Database,
        allowed_book_ids: Optional[set[str]] = None,
    ):
        self.db = db
        self.allowed_book_ids = (
            {
                str(book_id).strip()
                for book_id in allowed_book_ids
                if str(book_id).strip()
            }
            if allowed_book_ids
            else None
        )
        self._title_cache: Optional[List[Dict[str, Any]]] = None

    def extract_related_books(self, user_query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find books mentioned in the query and return soft context entries."""
        if not user_query or len(user_query.strip()) < 2:
            return []

        related_books: List[Dict[str, Any]] = []
        seen_names: set[str] = set()

        candidates = self._extract_title_candidates(user_query)
        for title in candidates:
            match = self._match_book(title)
            if not match:
                continue

            book_name = match["name"]
            if book_name in seen_names:
                continue

            seen_names.add(book_name)
            related_books.append(match)
            if len(related_books) >= limit:
                return related_books

        return related_books

    def extract_reference_tags(
        self,
        user_query: str,
        search_terms: str = "",
    ) -> List[str]:
        """Backward-compatible wrapper.

        The engine no longer uses these tags for retrieval. Keep this around so
        older code paths can still inspect the matched books if needed.
        """
        related_books = self.extract_related_books(user_query)
        tags: List[str] = []
        seen = set()
        for book in related_books:
            for tag in book.get("tags", []):
                if tag in seen:
                    continue
                seen.add(tag)
                tags.append(tag)
        return tags

    def build_related_book_context(self, related_books: List[Dict[str, Any]]) -> str:
        """Serialize related books into a compact prompt context."""
        if not related_books:
            return ""

        payload = []
        for book in related_books:
            payload.append(
                {
                    "name": book.get("name", ""),
                    "author": book.get("author", ""),
                    "classification": book.get("classification", ""),
                    "publish_status": book.get("publish_status", ""),
                    "words_total": book.get("words_total", 0),
                    "tags": list(book.get("tags", []))[:8],
                    "intro": self._trim_text(str(book.get("intro", "")), 220),
                    "match_source": book.get("match_source", ""),
                    "match_score": round(float(book.get("match_score", 0.0) or 0.0), 3),
                }
            )

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def fuzzy_match_title(
        self, query_title: str, threshold: float = 0.55
    ) -> Optional[Dict[str, Any]]:
        if not query_title or len(query_title.strip()) < 2:
            return None

        cache = self._load_title_cache()
        query_normalized = self._normalize_text(query_title)

        best_match = None
        best_ratio = 0.0

        for book in cache:
            book_name = str(book.get("name", "")).strip()
            if len(book_name) < 2:
                continue

            name_normalized = self._normalize_text(book_name)
            if query_normalized in name_normalized or name_normalized in query_normalized:
                ratio = 0.95
            else:
                ratio = SequenceMatcher(None, query_normalized, name_normalized).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = book

        if best_match and best_ratio >= threshold:
            return self._book_to_context(
                best_match,
                score=best_ratio,
                source="sequence_matcher",
                matched_title=query_title,
            )
        return None

    def _match_book(self, title: str) -> Optional[Dict[str, Any]]:
        if not title or len(title.strip()) < 2:
            return None

        results = self.db.search_by_title_fuzzy(
            title,
            allowed_book_ids=self.allowed_book_ids,
        )
        if results:
            return self._book_to_context(
                results[0],
                score=1.0,
                source="database_title_match",
                matched_title=title,
            )

        return self.fuzzy_match_title(title)

    def _extract_title_candidates(self, user_query: str) -> List[str]:
        candidates: List[str] = []
        patterns = [
            r"《([^》]+)》",
            r"『([^』]+)』",
            r"“([^”]+)”",
            r'"([^"]+)"',
            r"'([^']+)'",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, user_query):
                candidate = str(match).strip()
                if len(candidate) >= 2:
                    candidates.append(candidate)

        candidates.extend(self._find_inline_title_mentions(user_query))
        return self._dedupe_preserve_order(candidates)

    def _find_inline_title_mentions(self, user_query: str) -> List[str]:
        normalized_query = self._normalize_text(user_query)
        if not normalized_query:
            return []

        mentions: List[str] = []
        for book in self._load_title_cache():
            name = str(book.get("name", "")).strip()
            normalized_name = self._normalize_text(name)
            if len(normalized_name) < 2:
                continue
            if normalized_name in normalized_query:
                mentions.append(name)
        return mentions

    def _book_to_context(
        self,
        book: Dict[str, Any],
        score: float,
        source: str,
        matched_title: str,
    ) -> Dict[str, Any]:
        return {
            "id": str(book.get("id", "")),
            "name": str(book.get("name", "")).strip(),
            "author": str(book.get("author", "")).strip(),
            "classification": str(book.get("classification", "")).strip(),
            "publish_status": str(book.get("publish_status", "")).strip(),
            "words_total": book.get("words_total", 0) or 0,
            "intro": str(book.get("intro", "")).strip(),
            "tags": list(book.get("tags", [])),
            "match_source": source,
            "match_score": float(score),
            "matched_title": matched_title,
        }

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "")).lower()

    @staticmethod
    def _trim_text(value: str, max_len: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _load_title_cache(self) -> List[Dict[str, Any]]:
        if self._title_cache is not None:
            return self._title_cache

        self._title_cache = self.db.get_all_items(
            allowed_book_ids=self.allowed_book_ids
        )
        print(f"[BookMatcher] Title cache loaded: {len(self._title_cache)} books")
        return self._title_cache
