import json
import time
from typing import Any, Dict, List, Optional

from src.core.database import Database
from src.core.llm import _generate_json_from_contents
from src.core.model_catalog import normalize_model_id


class SinglePromptLLMEngine:
    """Naive baseline that stuffs the whole catalog into one prompt."""

    ENGINE_NAME = "SinglePromptLLMEngine"
    PARSER_VARIANT = "single_prompt_catalog_full"

    def __init__(
        self,
        db: Optional[Database] = None,
        include_intro: bool = True,
        intro_char_limit: Optional[int] = None,
        allowed_book_ids: Optional[set[str]] = None,
        mode: str = "baseline",
    ) -> None:
        self.db = db if db is not None else Database()
        self.include_intro = include_intro
        self.intro_char_limit = intro_char_limit
        self.mode = mode
        self.allowed_book_ids = self._normalize_allowed_book_ids(allowed_book_ids)
        self.catalog = self._load_catalog()
        self.catalog_by_id = {
            str(item.get("id", "")).strip(): item
            for item in self.catalog
            if str(item.get("id", "")).strip()
        }
        self.catalog_prompt = self._build_catalog_prompt(self.catalog)

    @staticmethod
    def _normalize_allowed_book_ids(
        allowed_book_ids: Optional[set[str]] = None,
    ) -> Optional[set[str]]:
        if not allowed_book_ids:
            return None
        normalized = {
            str(book_id).strip()
            for book_id in allowed_book_ids
            if str(book_id).strip()
        }
        return normalized or None

    def _load_catalog(self) -> List[Dict[str, Any]]:
        items = self.db.get_all_items(allowed_book_ids=self.allowed_book_ids)
        return sorted(
            items,
            key=lambda item: (
                str(item.get("id", "")).strip(),
                str(item.get("name", "")).strip(),
            ),
        )

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> List[str]:
        if isinstance(raw_tags, str):
            try:
                parsed = json.loads(raw_tags)
            except Exception:
                return []
            raw_tags = parsed
        if isinstance(raw_tags, list):
            return [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        return []

    def _normalize_intro(self, value: Any) -> str:
        text = " ".join(str(value or "").split()).strip()
        if not text or not self.include_intro:
            return ""
        if self.intro_char_limit is None or self.intro_char_limit <= 0:
            return text
        if len(text) <= self.intro_char_limit:
            return text
        return text[: self.intro_char_limit].rstrip() + "..."

    def _book_card(self, item: Dict[str, Any]) -> str:
        book_id = str(item.get("id", "")).strip()
        tags = ",".join(self._normalize_tags(item.get("tags", [])))
        intro = self._normalize_intro(item.get("intro", ""))

        lines = [
            f"ID: {book_id}",
            f"Title: {str(item.get('name', '')).strip()}",
            f"Author: {str(item.get('author', '')).strip()}",
            f"Status: {str(item.get('publish_status', '')).strip()}",
            f"Words: {int(item.get('words_total', 0) or 0)}",
            f"Classification: {str(item.get('classification', '')).strip()}",
            f"Tags: {tags}",
        ]
        if intro:
            lines.append(f"Intro: {intro}")
        return "\n".join(lines)

    def _build_catalog_prompt(self, items: List[Dict[str, Any]]) -> str:
        cards = [self._book_card(item) for item in items]
        return "\n\n---\n\n".join(cards)

    @staticmethod
    def _ranking_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ranked_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "book_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["book_id", "reason"],
                    },
                },
            },
            "required": ["ranked_ids", "notes"],
        }

    @staticmethod
    def _system_instruction(limit: int, mode: str = "baseline") -> str:
        if mode == "rerank":
            return f"""
You are an expert reranker for web novels.

You will receive:
- The user query
- A list of candidate books that have ALREADY passed primary structured filters (status, author, etc.).

Your task:
- Analyze the user query and the book descriptions (Intros) carefully.
- Focus on semantic alignment, themes, and emotional tone.
- Rank the top {limit} books that best match the query's underlying intent.
- Return JSON only.

Output contract:
- `ranked_ids`: ordered best-to-worst list of book ids.
- `notes`: short reasons explaining the semantic fit for each selected id.

Rules:
- Trust that the provided candidates satisfy all hard constraints.
- Prioritize matches based on content, plot, and writing style described in the Intro.
- Only use ids that appear in the provided catalog.
- If unsure, prefer returning fewer ids over hallucinating.
""".strip()

        return f"""
You are a deliberately naive whole-catalog ranking baseline for web novels.

You will receive:
- the user query
- a giant catalog dump containing every book in the database

Your task:
- read the catalog dump
- pick the best {limit} books for the query
- return JSON only

Output contract:
- `ranked_ids`: ordered best-to-worst list of book ids
- `notes`: optional short reasons for some selected ids

Rules:
- Only use ids that appear in the provided catalog.
- Do not invent ids, titles, or metadata.
- If unsure, prefer returning fewer ids over hallucinating.
- Keep reasons short and concrete.
- Use the catalog as-is. Do not apply hidden tools, external knowledge, or extra filtering stages.
""".strip()

    @staticmethod
    def _extract_notes_map(raw_notes: Any) -> Dict[str, str]:
        notes_map: Dict[str, str] = {}
        if not isinstance(raw_notes, list):
            return notes_map

        for note in raw_notes:
            if not isinstance(note, dict):
                continue
            book_id = str(note.get("book_id", "")).strip()
            reason = str(note.get("reason", "")).strip()
            if not book_id or not reason or book_id in notes_map:
                continue
            notes_map[book_id] = reason
        return notes_map

    def _sanitize_ranked_ids(self, ranked_ids: Any, limit: int) -> List[str]:
        if not isinstance(ranked_ids, list):
            return []

        valid_ids = set(self.catalog_by_id)
        deduped: List[str] = []
        seen = set()
        for raw_id in ranked_ids:
            book_id = str(raw_id or "").strip()
            if not book_id or book_id in seen or book_id not in valid_ids:
                continue
            seen.add(book_id)
            deduped.append(book_id)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _build_result_entry(
        item: Dict[str, Any],
        rank: int,
        total: int,
        explanation: Optional[str],
    ) -> Dict[str, Any]:
        if total <= 1:
            score = 1.0
        else:
            score = float((total - rank + 1) / total)

        return {
            "item": item,
            "score": score,
            "vector_score": 0.0,
            "breakdown": [
                {
                    "criteria": "single_prompt_rank",
                    "label": "Single Prompt Rank",
                    "raw_score": score,
                    "weighted_score": score,
                    "is_filter": False,
                    "reason": f"rank {rank}/{total} from naive single-prompt catalog ranking",
                }
            ],
            "payload": {},
            "explanation": explanation,
        }

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ = explain, cache_namespace

        normalized_model = normalize_model_id(model_id)
        started_at = time.perf_counter()
        contents = (
            f"User Query:\n{user_query}\n\n"
            f"Catalog Size: {len(self.catalog)} books\n\n"
            "Full Catalog Dump:\n"
            f"{self.catalog_prompt}"
        )

        try:
            # compute system instruction once so we can measure lengths
            system_instr = self._system_instruction(limit, mode=self.mode)

            # print prompt length diagnostics so callers can observe actual size
            contents_len = len(contents)
            system_len = len(system_instr)
            total_len = contents_len + system_len
            contents_bytes = len(contents.encode("utf-8"))
            system_bytes = len(system_instr.encode("utf-8"))
            total_bytes = contents_bytes + system_bytes
            print(
                f"[single_prompt_llm] contents_chars={contents_len} contents_bytes={contents_bytes} "
                f"system_chars={system_len} system_bytes={system_bytes} total_chars={total_len} total_bytes={total_bytes}"
            )

            payload, call_metadata = _generate_json_from_contents(
                contents=contents,
                task_label="single_prompt_catalog_rank",
                system_instruction=system_instr,
                response_schema=self._ranking_schema(),
                model_id=normalized_model,
                sampling_temperature=0.2,
                enforce_rate_limit=True,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            call_metadata = getattr(exc, "llm_call_metadata", {})
            setattr(
                exc,
                "parser_metadata",
                {
                    "engine_mode": "single_prompt_llm",
                    "parser_variant": self.PARSER_VARIANT,
                    "catalog_size": len(self.catalog),
                    "subset_enabled": self.allowed_book_ids is not None,
                    "subset_size": len(self.allowed_book_ids) if self.allowed_book_ids else None,
                    "prompt_char_count": len(contents),
                    "prompt_bytes_utf8": len(contents.encode("utf-8")),
                    "include_intro": self.include_intro,
                    "intro_char_limit": self.intro_char_limit,
                    "latency_ms": elapsed_ms,
                    "request_count": int(call_metadata.get("request_count", 0) or 0),
                    "retry_count": int(call_metadata.get("retry_count", 0) or 0),
                    "first_attempt_success": bool(call_metadata.get("first_attempt_success", False)),
                    "used_response_schema": bool(call_metadata.get("used_response_schema", False)),
                    "parse_source": str(call_metadata.get("parse_source", "failed")),
                    "recovered_from_raw_text": bool(call_metadata.get("recovered_from_raw_text", False)),
                    "model_id": str(call_metadata.get("model_id") or normalized_model),
                    "last_retry_error": str(call_metadata.get("last_retry_error", "")),
                    "error": str(exc),
                },
            )
            raise

        ranked_ids = self._sanitize_ranked_ids(payload.get("ranked_ids", []), limit)
        notes_map = self._extract_notes_map(payload.get("notes"))
        parse_metadata = {
            "engine_mode": "single_prompt_llm",
            "parser_variant": self.PARSER_VARIANT,
            "catalog_size": len(self.catalog),
            "subset_enabled": self.allowed_book_ids is not None,
            "subset_size": len(self.allowed_book_ids) if self.allowed_book_ids else None,
            "prompt_char_count": len(contents),
            "prompt_bytes_utf8": len(contents.encode("utf-8")),
            "include_intro": self.include_intro,
            "intro_char_limit": self.intro_char_limit,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "request_count": int(call_metadata.get("request_count", 0) or 0),
            "retry_count": int(call_metadata.get("retry_count", 0) or 0),
            "first_attempt_success": bool(call_metadata.get("first_attempt_success", False)),
            "used_response_schema": bool(call_metadata.get("used_response_schema", False)),
            "parse_source": str(call_metadata.get("parse_source", "unknown")),
            "recovered_from_raw_text": bool(call_metadata.get("recovered_from_raw_text", False)),
            "model_id": str(call_metadata.get("model_id") or normalized_model),
            "last_retry_error": str(call_metadata.get("last_retry_error", "")),
            "raw_ranked_count": len(payload.get("ranked_ids", []) or []),
            "valid_ranked_count": len(ranked_ids),
            "invalid_ranked_count": max(0, len(payload.get("ranked_ids", []) or []) - len(ranked_ids)),
        }

        if not ranked_ids:
            return {
                "query": user_query,
                "parsed_criteria": [],
                "search_terms": user_query,
                "generated_keywords": [],
                "tag_intent": {
                    "search_terms": user_query,
                    "positive_terms": [],
                    "negative_terms": [],
                },
                "hypothetical_intro": "",
                "related_books": [],
                "reference_tags": [],
                "parse_metadata": parse_metadata,
                "query_vector": [],
                "results": [],
                "message": "No matching novels were returned by the naive single-prompt LLM baseline.",
                "engine": self.ENGINE_NAME,
                "parser_variant": self.PARSER_VARIANT,
            }

        final_results = []
        total = len(ranked_ids)
        for index, book_id in enumerate(ranked_ids, start=1):
            item = self.catalog_by_id.get(book_id)
            if not item:
                continue
            final_results.append(
                self._build_result_entry(
                    item=item,
                    rank=index,
                    total=total,
                    explanation=notes_map.get(book_id),
                )
            )

        return {
            "query": user_query,
            "parsed_criteria": [],
            "search_terms": user_query,
            "generated_keywords": [],
            "tag_intent": {
                "search_terms": user_query,
                "positive_terms": [],
                "negative_terms": [],
            },
            "hypothetical_intro": "",
            "related_books": [],
            "reference_tags": [],
            "parse_metadata": parse_metadata,
            "query_vector": [],
            "results": final_results,
            "engine": self.ENGINE_NAME,
            "parser_variant": self.PARSER_VARIANT,
        }
