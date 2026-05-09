"""
Experimental engines that bypass vector-based tag mapping.

DirectTagEngine
    Reuses the *exact same* LLM query-parsing pipeline as HybridEngine but
    replaces ``batch_map_tags`` with a simple exact-string-match against the
    tag whitelist.

SchemaConstrainedTagEngine
    Goes one step further: it makes its *own* tag-projection LLM call with a
    ``response_schema`` that uses ``enum`` to restrict outputs to the curated
    tag whitelist.  This guarantees zero hallucinations at the API level, while
    still bypassing vector tag mapping.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.core.engine import HybridEngine
from src.core.database import Database
from src.core.llm import (
    parse_query,
    _generate_json_from_contents,
    _normalize_tag_projection,
    _build_tag_projection_compact_context,
    _build_parallel_context,
    _build_tag_intent_from_projection,
    _build_structured_context_from_semantic_understanding,
    DEFAULT_PARSER_MODEL,
    DEBUG_LLM_OUTPUT,
)
from src.core.explainer import generate_explanation
from src.core.vector_store import VectorStore
from src.models.schemas import ScoringCriteria, ScoringParameters


class DirectTagEngine(HybridEngine):
    """Experimental engine that bypasses vector tag mapping.

    Inherits all infrastructure from HybridEngine (DB, VectorStore, BookMatcher,
    tag cache, scoring helpers, post-filtering, etc.) but overrides ``search``
    to replace the vector-based ``batch_map_tags`` step with a simple exact-match
    lookup against the curated tag whitelist.
    """

    # ------------------------------------------------------------------
    # Direct tag matching (replaces VectorStore.batch_map_tags)
    # ------------------------------------------------------------------

    def _build_direct_tag_mapping_weights(
        self,
        tag_terms_list: List[str],
    ) -> List[Dict[str, float]]:
        """Convert LLM-generated tag terms into exact-match weight dicts.

        For each term produced by the LLM's tag-projection step:
        * If it appears verbatim in ``self.all_tags_cache`` → ``{term: 1.0}``
        * Otherwise (hallucination / near-miss) → ``{}`` (empty dict)

        This mirrors the shape returned by ``VectorStore.batch_map_tags`` so the
        downstream ``calculate_score`` and ``_extract_recall_tags`` helpers work
        without any changes.
        """
        all_tags_set = set(self.all_tags_cache) if self.all_tags_cache else set()
        results: List[Dict[str, float]] = []
        for term in tag_terms_list:
            if term in all_tags_set:
                results.append({term: 1.0})
            else:
                results.append({})
        return results

    # ------------------------------------------------------------------
    # search() override – identical to HybridEngine except for tag step
    # ------------------------------------------------------------------

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.all_tags_cache:
            print(
                f"[DirectTagEngine] Using cached tag list with {len(self.all_tags_cache)} entries."
            )

        # ---- Book-mention extraction (identical) ----
        related_books = self.book_matcher.extract_related_books(user_query)
        related_book_context = self.book_matcher.build_related_book_context(related_books)

        # ---- LLM query parsing (identical) ----
        parse_result = parse_query(
            user_query,
            model_id=model_id,
            cache_namespace=cache_namespace,
            tag_list=self.all_tags_cache,
            reference_book_context=related_book_context,
        )

        positive_tag_terms = list(parse_result.tag_intent.positive_terms) or list(
            parse_result.generated_keywords
        )
        tag_terms_list = self._build_tag_terms_list(positive_tag_terms)

        # ============================================================
        # KEY DIFFERENCE: exact-match instead of vector mapping
        # ============================================================
        tag_mapping_weights: List[Dict[str, float]] = []
        if tag_terms_list:
            print(f"[DirectTagEngine] Direct-matching {len(tag_terms_list)} tag facets (no vector mapping).")
            tag_mapping_weights = self._build_direct_tag_mapping_weights(tag_terms_list)

            matched = sum(1 for m in tag_mapping_weights if m)
            print(
                f"[DirectTagEngine] Exact match result: "
                f"{matched}/{len(tag_terms_list)} terms hit the whitelist."
            )

        # ---- Semantic expansion (identical) ----
        base_terms = parse_result.search_terms or parse_result.original_query
        expanded_terms = base_terms

        positive_semantic = [
            criteria
            for criteria in parse_result.criteria
            if criteria.name == "semantic_similarity"
            and not getattr(criteria, "is_negative", False)
        ]
        semantic_texts = []
        normalized_base_terms = "".join(str(base_terms).split()).lower()
        for criteria in positive_semantic:
            query_text = self._criteria_params(criteria).get("query_text", "").strip()
            normalized_query_text = "".join(query_text.split()).lower()
            if query_text and normalized_query_text != normalized_base_terms:
                semantic_texts.append(query_text)
        if semantic_texts:
            semantic_expansion = " ".join(semantic_texts)
            expanded_terms = f"{expanded_terms} {semantic_expansion}".strip()

        # ---- Vector retrieval (identical) ----
        retrieval_limit = 10000
        candidates_map: Dict[str, Dict[str, Any]] = {}
        vector_score_map: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}

        vector_results, query_vector = self.vs.search(
            expanded_terms,
            limit=retrieval_limit,
            query_filter=None,
            with_payload=True,
        )
        for hit in vector_results:
            payload = hit.get("payload") or {}
            book_id = payload.get("id")
            if not book_id:
                continue
            book_id = str(book_id)
            candidates_map[book_id] = payload
            payload_map[book_id] = payload
            vector_score_map[book_id] = float(hit["score"])

        # ---- Tag recall (uses direct-match tags instead of mapped tags) ----
        if tag_terms_list and tag_mapping_weights:
            recall_tags = self._extract_recall_tags(tag_mapping_weights)
            if recall_tags:
                print(f"[DirectTagEngine] Triggering direct-tag recall for {len(recall_tags)} resolved tags.")
                tag_recall_items = self.db.search_by_tags_any(recall_tags, limit=retrieval_limit)
                for item in tag_recall_items:
                    book_id = str(item.get("id", "")).strip()
                    if not book_id:
                        continue
                    if book_id not in candidates_map:
                        candidates_map[book_id] = item
                        payload_map[book_id] = item
                        vector_score_map[book_id] = 0.0

        # ---- Candidate enrichment & filtering (identical) ----
        candidates: List[Dict[str, Any]] = []
        for book_id, item in candidates_map.items():
            if not item.get("classification") or not item.get("words_total"):
                db_item = self.db.get_item(book_id)
                if db_item:
                    item = {**db_item, **item}
                    candidates_map[book_id] = item
            if "id" not in item or not item["id"]:
                item["id"] = book_id
            has_minimum_metadata = bool(
                str(item.get("name", "")).strip()
                or str(item.get("intro", "")).strip()
                or item.get("words_total")
                or item.get("tags")
                or str(item.get("classification", "")).strip()
            )
            if not has_minimum_metadata:
                continue
            candidates.append(item)

        print(f"[DirectTagEngine] Candidate pool size: {len(candidates)}")

        # ---- Scoring (identical – reuses parent calculate_score) ----
        negative_tag_terms = self._dedupe_terms(
            list(parse_result.tag_intent.negative_terms)
        ) or self._resolve_negative_tag_terms(parse_result.criteria)
        scored_items = []
        for item in candidates:
            book_id = str(item.get("id"))
            if not book_id or book_id == "None":
                continue
            vector_score = vector_score_map.get(book_id, 0.0)
            final_score, breakdown = self.calculate_score(
                item,
                vector_score=vector_score,
                tag_terms_list=tag_terms_list,
                tag_mapping_weights=tag_mapping_weights,
            )
            scored_items.append(
                {
                    "item": item,
                    "score": float(final_score),
                    "vector_score": vector_score,
                    "breakdown": breakdown,
                    "payload": payload_map.get(book_id, {}),
                }
            )

        # ---- Post-filter & sort (identical) ----
        scored_items.sort(key=lambda result: result["score"], reverse=True)
        scored_items = self._post_filter(
            scored_items,
            parse_result.criteria,
            negative_tag_terms,
        )
        scored_items.sort(key=lambda result: result["score"], reverse=True)

        if not scored_items:
            return {
                "query": user_query,
                "parsed_criteria": [
                    self._criteria_to_dict(criteria) for criteria in parse_result.criteria
                ],
                "search_terms": parse_result.search_terms,
                "generated_keywords": parse_result.generated_keywords,
                "tag_intent": {
                    "positive_terms": list(parse_result.tag_intent.positive_terms),
                    "negative_terms": negative_tag_terms,
                },
                "query_vector": query_vector,
                "results": [],
                "message": "No matching novels were found after applying the filters.",
                "engine": "DirectTagEngine",
                "related_books": related_books,
                "reference_tags": recall_tags if 'recall_tags' in locals() else [],
                "parse_metadata": parse_result.parse_metadata,
            }

        final_results = scored_items[:limit]

        # ---- Explanation generation (identical) ----
        top_n_explain = 3 if explain else 0
        explainer_runtime_state = {
            "gemini_fail_count": 0,
            "gemini_disabled": False,
            "gemini_fail_threshold": 3,
        }
        for index, result in enumerate(final_results):
            if index >= top_n_explain:
                result["explanation"] = None
                continue

            item = result["item"]
            payload = result.get("payload", {})
            chunks_to_analyze = []
            if payload.get("content"):
                chunks_to_analyze.append(f"Retrieved content:\n{payload['content'][:500]}...")
            elif payload.get("intro"):
                chunks_to_analyze.append(f"Retrieved intro:\n{payload['intro'][:500]}...")
            if item.get("intro"):
                chunks_to_analyze.append(f"Database intro:\n{item['intro']}")

            result["explanation"] = generate_explanation(
                query=user_query,
                book_item=item,
                context_chunks=chunks_to_analyze,
                score_breakdown=result["breakdown"],
                runtime_state=explainer_runtime_state,
                model_id=model_id,
            )

        return {
            "query": user_query,
            "parsed_criteria": [
                self._criteria_to_dict(criteria) for criteria in parse_result.criteria
            ],
            "search_terms": parse_result.search_terms,
            "generated_keywords": parse_result.generated_keywords,
            "tag_intent": {
                "positive_terms": list(parse_result.tag_intent.positive_terms),
                "negative_terms": negative_tag_terms,
            },
            "hypothetical_intro": parse_result.hypothetical_intro,
            "related_books": related_books,
            "reference_tags": recall_tags if 'recall_tags' in locals() else [],
            "parse_metadata": parse_result.parse_metadata,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "DirectTagEngine",
        }


class SchemaConstrainedTagEngine(DirectTagEngine):
    """Experimental engine that uses response_schema enum to constrain LLM tag output.

    This engine makes its *own* tag-projection LLM call with a ``response_schema``
    that uses ``enum`` to restrict ``positive_terms`` and ``negative_terms`` to
    only values from the curated tag whitelist.  This eliminates tag hallucinations
    at the API level.

    Like ``DirectTagEngine``, it bypasses ``batch_map_tags`` — the tags produced
    by the schema-constrained call are used directly with weight 1.0.

    Pipeline comparison:

        HybridEngine              → free-form LLM tags → vector mapping (soft weights)
        DirectTagEngine           → free-form LLM tags → exact match (binary weights)
        SchemaConstrainedTagEngine → enum-constrained LLM tags → direct use (weight 1.0)
    """

    # ------------------------------------------------------------------
    # Schema-constrained tag projection
    # ------------------------------------------------------------------

    def _build_enum_tag_projection_schema(self) -> Dict[str, Any]:
        """Build a response_schema that constrains tag arrays to enum values."""
        tag_enum = list(self.all_tags_cache) if self.all_tags_cache else []
        return {
            "type": "object",
            "properties": {
                "positive_terms": {
                    "type": "array",
                    "items": {"type": "string", "enum": tag_enum},
                },
                "negative_terms": {
                    "type": "array",
                    "items": {"type": "string", "enum": tag_enum},
                },
            },
            "required": ["positive_terms", "negative_terms"],
        }

    def _run_schema_constrained_tag_projection(
        self,
        user_query: str,
        semantic_understanding: Dict[str, Any],
        model_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run tag projection with enum-constrained response_schema.

        Uses the same shared_context and prompt structure as the production
        tag_projection branch, but replaces the open-ended string arrays
        with enum-constrained arrays.
        """
        # Build shared context (same as production)
        shared_context = _build_parallel_context(
            tag_list=self.all_tags_cache,
        )

        instruction = f"""
{shared_context}

You are the tag projection pass.

    Return JSON with:
    - positive_terms
    - negative_terms

    Rules:
    - The input includes the original query and a compact semantic understanding summary.
    - Project only the strongest retrieval anchors into short tag-like terms.
    - You MUST select tags ONLY from the allowed enum values.
    - Be conservative. Omit weak, optional, or example-derived concepts.
    - `positive_terms` should contain 3-6 high-confidence terms only.
    - `negative_terms` should contain 0-4 explicit exclusions only.
    - Avoid near-duplicates, synonyms, and broad filler terms.
    - If a concept belongs only in semantic retrieval text and not in tags, omit it.
    - Return only these two keys. Do not emit helper fields, explanations, or notes.
""".strip()

        # Build contents (same as production)
        tag_projection_context = _build_tag_projection_compact_context(
            semantic_understanding
        )
        contents = (
            f"Original Query:\n{user_query}\n\n"
            f"Compact Semantic Understanding:\n{tag_projection_context}"
        )

        # Build enum-constrained schema
        schema = self._build_enum_tag_projection_schema()

        selected_model = str(
            model_id or DEFAULT_PARSER_MODEL
        ).strip() or DEFAULT_PARSER_MODEL

        started_at = time.perf_counter()
        try:
            raw_result, call_metadata = _generate_json_from_contents(
                contents=contents,
                task_label="tag_projection_enum",
                system_instruction=instruction,
                response_schema=schema,
                model_id=selected_model,
                sampling_temperature=0.2,
                enforce_rate_limit=False,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            print(
                f"[SchemaConstrainedTagEngine] tag_projection_enum failed "
                f"after {elapsed_ms:.0f}ms: {exc}"
            )
            raise

        if DEBUG_LLM_OUTPUT:
            print(
                f"[debug:tag_projection_enum] raw="
                f"{json.dumps(raw_result, ensure_ascii=False)}"
            )

        result = _normalize_tag_projection(raw_result)

        if DEBUG_LLM_OUTPUT:
            print(
                f"[debug:tag_projection_enum] normalized="
                f"{json.dumps(result, ensure_ascii=False)}"
            )

        latency_ms = (time.perf_counter() - started_at) * 1000
        branch_metadata = {
            "success": True,
            "latency_ms": round(latency_ms, 2),
            "request_count": int(call_metadata.get("request_count", 0) or 0),
            "retry_count": int(call_metadata.get("retry_count", 0) or 0),
            "first_attempt_success": bool(
                call_metadata.get("first_attempt_success", False)
            ),
            "used_response_schema": True,
            "parse_source": str(call_metadata.get("parse_source", "unknown")),
            "recovered_from_raw_text": bool(
                call_metadata.get("recovered_from_raw_text", False)
            ),
            "model_id": str(
                call_metadata.get("model_id") or selected_model
            ),
            "last_retry_error": str(call_metadata.get("last_retry_error", "")),
            "schema_type": "enum_constrained",
        }
        return result, branch_metadata

    # ------------------------------------------------------------------
    # search() override – replaces tag projection with enum-constrained version
    # ------------------------------------------------------------------

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.all_tags_cache:
            print(
                f"[SchemaConstrainedTagEngine] Using cached tag list "
                f"with {len(self.all_tags_cache)} entries."
            )

        # ---- Book-mention extraction (identical) ----
        related_books = self.book_matcher.extract_related_books(user_query)
        related_book_context = self.book_matcher.build_related_book_context(
            related_books
        )

        # ---- LLM query parsing (reuse for semantic + structured) ----
        parse_result = parse_query(
            user_query,
            model_id=model_id,
            cache_namespace=cache_namespace,
            tag_list=self.all_tags_cache,
            reference_book_context=related_book_context,
        )

        # ============================================================
        # KEY DIFFERENCE: run our own enum-constrained tag projection
        # ============================================================
        # Extract semantic_understanding from the parse_result metadata
        # to feed into our custom tag projection call.
        #
        # The parse_result already contains the merged semantic_understanding
        # output; we reconstruct a lightweight version from the available fields.
        semantic_understanding_for_projection = {
            "semantic_query_text": parse_result.search_terms or "",
            "intent_summary": "",
            "positive_concepts": list(parse_result.generated_keywords),
            "negative_concepts": [
                self._criteria_params(c).get("query_text", "")
                for c in parse_result.criteria
                if c.name == "semantic_similarity"
                and getattr(c, "is_negative", False)
            ],
        }

        tag_projection, tp_metadata = self._run_schema_constrained_tag_projection(
            user_query=user_query,
            semantic_understanding=semantic_understanding_for_projection,
            model_id=model_id,
        )

        # Build tag intent from the schema-constrained projection
        tag_intent = _build_tag_intent_from_projection(
            user_query=user_query,
            semantic_understanding=semantic_understanding_for_projection,
            tag_projection=tag_projection,
        )

        # CRITICAL FIX: Sync the constrained tags back into parse_result for logging
        parse_result.tag_intent = tag_intent
        # Re-build criteria list from the new tag_intent
        # This ensures parsed_criteria in the output JSON reflects the constrained tags
        new_criteria = [
            c for c in parse_result.criteria 
            if c.name != "semantic_similarity" or c.parameters.query_text == (parse_result.search_terms or user_query)
        ]
        
        for term in tag_intent.positive_terms:
            new_criteria.append(ScoringCriteria(
                name="semantic_similarity",
                is_negative=False,
                parameters=ScoringParameters(query_text=term)
            ))
        for term in tag_intent.negative_terms:
            new_criteria.append(ScoringCriteria(
                name="semantic_similarity",
                is_negative=True,
                parameters=ScoringParameters(query_text=term)
            ))
        parse_result.criteria = new_criteria

        positive_tag_terms = list(tag_intent.positive_terms) or list(
            parse_result.generated_keywords
        )
        tag_terms_list = self._build_tag_terms_list(positive_tag_terms)

        # All tags from enum-constrained output are guaranteed valid
        tag_mapping_weights: List[Dict[str, float]] = []
        if tag_terms_list:
            print(
                f"[SchemaConstrainedTagEngine] Schema-constrained "
                f"{len(tag_terms_list)} tag facets (all guaranteed valid)."
            )
            tag_mapping_weights = self._build_direct_tag_mapping_weights(
                tag_terms_list
            )
            matched = sum(1 for m in tag_mapping_weights if m)
            print(
                f"[SchemaConstrainedTagEngine] Validation: "
                f"{matched}/{len(tag_terms_list)} terms confirmed in whitelist."
            )

        # ---- Semantic expansion (identical) ----
        base_terms = parse_result.search_terms or parse_result.original_query
        expanded_terms = base_terms

        positive_semantic = [
            criteria
            for criteria in parse_result.criteria
            if criteria.name == "semantic_similarity"
            and not getattr(criteria, "is_negative", False)
        ]
        semantic_texts = []
        normalized_base_terms = "".join(str(base_terms).split()).lower()
        for criteria in positive_semantic:
            query_text = self._criteria_params(criteria).get(
                "query_text", ""
            ).strip()
            normalized_query_text = "".join(query_text.split()).lower()
            if query_text and normalized_query_text != normalized_base_terms:
                semantic_texts.append(query_text)
        if semantic_texts:
            semantic_expansion = " ".join(semantic_texts)
            expanded_terms = f"{expanded_terms} {semantic_expansion}".strip()

        # ---- Vector retrieval (identical) ----
        retrieval_limit = 10000
        candidates_map: Dict[str, Dict[str, Any]] = {}
        vector_score_map: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}

        vector_results, query_vector = self.vs.search(
            expanded_terms,
            limit=retrieval_limit,
            query_filter=None,
            with_payload=True,
        )
        for hit in vector_results:
            payload = hit.get("payload") or {}
            book_id = payload.get("id")
            if not book_id:
                continue
            book_id = str(book_id)
            candidates_map[book_id] = payload
            payload_map[book_id] = payload
            vector_score_map[book_id] = float(hit["score"])

        # ---- Tag recall ----
        if tag_terms_list and tag_mapping_weights:
            recall_tags = self._extract_recall_tags(tag_mapping_weights)
            if recall_tags:
                print(
                    f"[SchemaConstrainedTagEngine] Triggering tag recall "
                    f"for {len(recall_tags)} schema-constrained tags."
                )
                tag_recall_items = self.db.search_by_tags_any(
                    recall_tags, limit=retrieval_limit
                )
                for item in tag_recall_items:
                    book_id = str(item.get("id", "")).strip()
                    if not book_id:
                        continue
                    if book_id not in candidates_map:
                        candidates_map[book_id] = item
                        payload_map[book_id] = item
                        vector_score_map[book_id] = 0.0

        # ---- Candidate enrichment & filtering (identical) ----
        candidates: List[Dict[str, Any]] = []
        for book_id, item in candidates_map.items():
            if not item.get("classification") or not item.get("words_total"):
                db_item = self.db.get_item(book_id)
                if db_item:
                    item = {**db_item, **item}
                    candidates_map[book_id] = item
            if "id" not in item or not item["id"]:
                item["id"] = book_id
            has_minimum_metadata = bool(
                str(item.get("name", "")).strip()
                or str(item.get("intro", "")).strip()
                or item.get("words_total")
                or item.get("tags")
                or str(item.get("classification", "")).strip()
            )
            if not has_minimum_metadata:
                continue
            candidates.append(item)

        print(
            f"[SchemaConstrainedTagEngine] Candidate pool size: "
            f"{len(candidates)}"
        )

        # ---- Scoring (identical – reuses parent calculate_score) ----
        negative_tag_terms = self._dedupe_terms(
            list(tag_intent.negative_terms)
        ) or self._resolve_negative_tag_terms(parse_result.criteria)
        scored_items = []
        for item in candidates:
            book_id = str(item.get("id"))
            if not book_id or book_id == "None":
                continue
            vector_score = vector_score_map.get(book_id, 0.0)
            final_score, breakdown = self.calculate_score(
                item,
                vector_score=vector_score,
                tag_terms_list=tag_terms_list,
                tag_mapping_weights=tag_mapping_weights,
            )
            scored_items.append(
                {
                    "item": item,
                    "score": float(final_score),
                    "vector_score": vector_score,
                    "breakdown": breakdown,
                    "payload": payload_map.get(book_id, {}),
                }
            )

        # ---- Post-filter & sort (identical) ----
        scored_items.sort(key=lambda result: result["score"], reverse=True)
        scored_items = self._post_filter(
            scored_items,
            parse_result.criteria,
            negative_tag_terms,
        )
        scored_items.sort(key=lambda result: result["score"], reverse=True)

        if not scored_items:
            return {
                "query": user_query,
                "parsed_criteria": [
                    self._criteria_to_dict(criteria)
                    for criteria in parse_result.criteria
                ],
                "search_terms": parse_result.search_terms,
                "generated_keywords": parse_result.generated_keywords,
                "tag_intent": tag_intent.model_dump(),
                "query_vector": query_vector,
                "results": [],
                "message": "No matching novels were found.",
                "engine": "SchemaConstrainedTagEngine",
                "related_books": related_books,
                "reference_tags": recall_tags if 'recall_tags' in locals() else [],
                "parse_metadata": parse_result.parse_metadata,
            }

        final_results = scored_items[:limit]

        # ---- Explanation generation (identical) ----
        top_n_explain = 3 if explain else 0
        explainer_runtime_state = {
            "gemini_fail_count": 0,
            "gemini_disabled": False,
            "gemini_fail_threshold": 3,
        }
        for index, result in enumerate(final_results):
            if index >= top_n_explain:
                result["explanation"] = None
                continue

            item = result["item"]
            payload = result.get("payload", {})
            chunks_to_analyze = []
            if payload.get("content"):
                chunks_to_analyze.append(
                    f"Retrieved content:\n{payload['content'][:500]}..."
                )
            elif payload.get("intro"):
                chunks_to_analyze.append(
                    f"Retrieved intro:\n{payload['intro'][:500]}..."
                )
            if item.get("intro"):
                chunks_to_analyze.append(
                    f"Database intro:\n{item['intro']}"
                )

            result["explanation"] = generate_explanation(
                query=user_query,
                book_item=item,
                context_chunks=chunks_to_analyze,
                score_breakdown=result["breakdown"],
                runtime_state=explainer_runtime_state,
                model_id=model_id,
            )

        # Merge parse_metadata with the extra tag_projection_enum branch info
        merged_parse_metadata = dict(parse_result.parse_metadata or {})
        branches = dict(merged_parse_metadata.get("branches", {}))
        branches["tag_projection_enum"] = tp_metadata
        merged_parse_metadata["branches"] = branches

        return {
            "query": user_query,
            "parsed_criteria": [
                self._criteria_to_dict(criteria)
                for criteria in parse_result.criteria
            ],
            "search_terms": parse_result.search_terms,
            "generated_keywords": parse_result.generated_keywords,
            "tag_intent": tag_intent.model_dump(),
            "hypothetical_intro": parse_result.hypothetical_intro,
            "related_books": related_books,
            "reference_tags": recall_tags if 'recall_tags' in locals() else [],
            "parse_metadata": merged_parse_metadata,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "SchemaConstrainedTagEngine",
        }
