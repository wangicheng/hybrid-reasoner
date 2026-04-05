import json
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.core.book_matcher import BookMatcher
from src.core.database import Database
from src.core.explainer import generate_explanation
from src.core.llm import parse_query
from src.core.query_synonym_normalizer import normalize_query_after_book_lookup
from src.core.vector_store import VectorStore
from src.core.tag_context import build_tag_context_text, load_tag_descriptions


class HybridEngine:
    """Production search engine using the selected tag-processing strategy."""

    def __init__(
        self,
        db: Optional[Database] = None,
        vs: Optional[VectorStore] = None,
        semantic_weight: Optional[float] = None,
        attribute_weight: Optional[float] = None,
        use_tag_descriptions: bool = False,
        embed_generated_keywords: bool = True,
        tag_descriptions_path: Optional[str] = None,
    ):
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        self.book_matcher = BookMatcher(self.db)
        self.use_tag_descriptions = use_tag_descriptions
        self.embed_generated_keywords = embed_generated_keywords
        self.semantic_weight = (
            semantic_weight if semantic_weight is not None else settings.SEMANTIC_WEIGHT
        )
        self.attribute_weight = (
            attribute_weight
            if attribute_weight is not None
            else settings.ATTRIBUTE_WEIGHT
        )
        self.all_tags_cache: Optional[Tuple[str, ...]] = None
        self.tag_descriptions_cache: Optional[Dict[str, str]] = None
        self.tag_context_cache: Optional[str] = None
        self._load_tags_cache()
        if not self.all_tags_cache:
            raise RuntimeError(
                "Tag metadata file 'data/all_tags.json' is missing or empty."
            )

        # Keep the tag embedding collection aligned with the curated whitelist.
        self.vs.sync_tag_collection(self.all_tags_cache)

        if self.use_tag_descriptions:
            self._load_tag_context_cache(tag_descriptions_path)

        if not self.vs.collection_exists("novel_tags"):
            raise RuntimeError("Qdrant collection 'novel_tags' is missing.")

    def _load_tags_cache(self) -> None:
        tags_path = "data/all_tags.json"
        try:
            with open(tags_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(tags_path, "r", encoding="utf-16") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Tag metadata file '{tags_path}' not found.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load tag metadata from '{tags_path}': {exc}") from exc

        if isinstance(data, list) and data:
            self.all_tags_cache = tuple(str(tag) for tag in data if tag)
            return

        raise RuntimeError(
            f"Tag metadata file '{tags_path}' is empty or has an unexpected format."
        )

    def _load_tag_context_cache(self, tag_descriptions_path: Optional[str]) -> None:
        descriptions = load_tag_descriptions(tag_descriptions_path)
        self.tag_descriptions_cache = descriptions
        self.tag_context_cache = build_tag_context_text(self.all_tags_cache or (), descriptions)
        if not self.tag_context_cache.strip():
            raise RuntimeError("Tag context cache could not be built from descriptions.")

    @staticmethod
    def _criteria_params(criteria: Any) -> Dict[str, Any]:
        params = getattr(criteria, "parameters", {})
        if hasattr(params, "model_dump"):
            return params.model_dump()
        if hasattr(params, "dict"):
            return params.dict()
        return dict(params)

    @staticmethod
    def _criteria_to_dict(criteria: Any) -> Dict[str, Any]:
        if hasattr(criteria, "model_dump"):
            return criteria.model_dump()
        if hasattr(criteria, "dict"):
            return criteria.dict()
        return dict(criteria)

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> List[str]:
        if isinstance(raw_tags, str):
            try:
                raw_tags = json.loads(raw_tags)
            except Exception:
                return []
        if isinstance(raw_tags, list):
            return [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        return []

    @staticmethod
    def _dedupe_terms(terms: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for term in terms:
            normalized = term.replace(" ", "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _build_tag_terms_list(
        self,
        generated_keywords: List[str],
    ) -> List[str]:
        terms: List[str] = []
        if self.embed_generated_keywords:
            terms.extend(generated_keywords)
        return self._dedupe_terms(terms)

    @staticmethod
    def _normalize_status(status_value: str) -> Optional[str]:
        raw_value = str(status_value or "").strip()
        lowered = raw_value.lower()
        completed_keywords = ["complet", "finish", "ended", "done", "完結", "已完結"]
        ongoing_keywords = ["ongoing", "serializ", "running", "active", "連載", "連載中"]

        if any(keyword in lowered or keyword in raw_value for keyword in completed_keywords):
            return "completed"
        if any(keyword in lowered or keyword in raw_value for keyword in ongoing_keywords):
            return "ongoing"
        return None


    def calculate_score(
        self,
        item: Dict[str, Any],
        vector_score: float,
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        breakdown: List[Dict[str, Any]] = []

        semantic_score = 0.1 + 0.9 * vector_score
        breakdown.append(
            {
                "criteria": "semantic_track",
                "label": "Semantic Track",
                "raw_score": vector_score,
                "weighted_score": semantic_score,
                "is_filter": False,
                "reason": f"semantic score {semantic_score:.4f} (raw {vector_score:.4f})",
            }
        )

        attribute_score = 1.0
        has_tag_scoring = False
        if tag_terms_list and tag_mapping_weights:
            book_tags = self._normalize_tags(item.get("tags", []))
            total_facet_score = 0.0
            matched_details = []

            for index, facet_map in enumerate(tag_mapping_weights):
                target_term = (
                    tag_terms_list[index] if index < len(tag_terms_list) else f"facet_{index}"
                )
                best_score = 0.0
                best_tag = None
                for book_tag in book_tags:
                    similarity = facet_map.get(book_tag, 0.0)
                    if similarity > best_score:
                        best_score = similarity
                        best_tag = book_tag

                total_facet_score += best_score
                if best_tag is not None and best_score > 0:
                    matched_details.append(f"{target_term}->{best_tag}({best_score:.2f})")

            average_similarity = total_facet_score / len(tag_mapping_weights)
            attribute_score = 0.1 + 0.9 * average_similarity
            has_tag_scoring = True
            breakdown.append(
                {
                    "criteria": "attribute_track",
                    "label": "Attribute Track",
                    "raw_score": average_similarity,
                    "weighted_score": attribute_score,
                    "is_filter": False,
                    "reason": (
                        f"facet avg {average_similarity:.4f}; "
                        f"matches: {', '.join(matched_details) if matched_details else 'none'}"
                    ),
                }
            )

        if has_tag_scoring:
            total_score = (
                semantic_score * self.semantic_weight
                + attribute_score * self.attribute_weight
            )
            fusion_reason = (
                f"({semantic_score:.4f} * {self.semantic_weight}) + "
                f"({attribute_score:.4f} * {self.attribute_weight})"
            )
        else:
            total_score = semantic_score
            fusion_reason = f"semantic only: {semantic_score:.4f}"

        breakdown.append(
            {
                "criteria": "global_fusion",
                "label": "Global Fusion",
                "raw_score": total_score,
                "weighted_score": total_score,
                "is_filter": False,
                "reason": fusion_reason,
            }
        )

        return total_score, breakdown

    def _post_filter(
        self,
        scored_items: List[Dict[str, Any]],
        criteria_list: List[Any],
        negative_tag_terms: List[str],
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []

        status_filter = None
        author_filter = None
        words_min = None
        words_max = None

        for criteria in criteria_list:
            params = self._criteria_params(criteria)
            if criteria.name == "status_check":
                status_filter = self._normalize_status(params.get("target_status", ""))
            elif criteria.name == "author_match":
                author_filter = params.get("author_name", "").strip()
            elif criteria.name == "numeric_range" and params.get("field") == "words_total":
                words_min = params.get("min_val")
                words_max = params.get("max_val")

        for result in scored_items:
            item = result["item"]
            excluded = False
            book_tags = self._normalize_tags(item.get("tags", []))

            for negative_term in negative_tag_terms:
                if any(
                    negative_term in book_tag or book_tag in negative_term
                    for book_tag in book_tags
                ):
                    excluded = True
                    break

            if not excluded and status_filter:
                item_status = self._normalize_status(item.get("publish_status", ""))
                if item_status != status_filter:
                    excluded = True

            if not excluded and author_filter:
                author = item.get("author", "")
                if not (author_filter in author or author in author_filter):
                    excluded = True

            if not excluded and (words_min is not None or words_max is not None):
                actual_words = item.get("words_total", 0) or 0
                if words_min is not None and actual_words < words_min:
                    excluded = True
                if words_max is not None and actual_words > words_max:
                    excluded = True

            if not excluded:
                filtered.append(result)

        print(
            f"[PostFilter] {len(scored_items)} -> {len(filtered)} "
            f"(removed {len(scored_items) - len(filtered)})"
        )
        return filtered

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
                f"[Engine] Using cached tag list with {len(self.all_tags_cache)} entries."
            )

        related_books = self.book_matcher.extract_related_books(user_query)
        related_book_context = self.book_matcher.build_related_book_context(related_books)

        normalized_query, query_replacements = normalize_query_after_book_lookup(user_query)
        if query_replacements:
            print(f"[SynonymNormalize] Added {len(query_replacements)} canonical hints: {query_replacements}")

        parse_result = parse_query(
            normalized_query,
            model_id=model_id,
            cache_namespace=cache_namespace,
            tag_list=self.all_tags_cache,
            tag_context=self.tag_context_cache if self.use_tag_descriptions else None,
            reference_book_context=related_book_context,
        )

        tag_terms_list = self._build_tag_terms_list(
            list(parse_result.generated_keywords),
        )

        tag_mapping_weights: List[Dict[str, float]] = []
        if tag_terms_list:
            print(f"[Engine] Pre-mapping {len(tag_terms_list)} tag facets.")
            tag_mapping_weights = self.vs.batch_map_tags(tag_terms_list)

        base_terms = parse_result.search_terms or parse_result.original_query
        expanded_terms = base_terms

        positive_semantic = [
            criteria
            for criteria in parse_result.criteria
            if criteria.name == "semantic_similarity"
            and not getattr(criteria, "is_negative", False)
        ]
        semantic_texts = []
        for criteria in positive_semantic:
            query_text = self._criteria_params(criteria).get("query_text", "").strip()
            if query_text:
                semantic_texts.append(query_text)
        if semantic_texts:
            semantic_expansion = " ".join(semantic_texts)
            expanded_terms = f"{expanded_terms} {semantic_expansion}".strip()

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

        if tag_terms_list:
            print(f"[Engine] Triggering mapped-tag recall for {len(tag_terms_list)} terms.")
            tag_queries = [f"tag: {term}" for term in tag_terms_list]
            tag_results = self.vs.search_individual(
                tag_queries,
                limit=retrieval_limit,
                collection_name="novel_tags",
            )
            for hit in tag_results:
                payload = hit.get("payload") or {}
                book_id = payload.get("id", hit.get("id"))
                if book_id is None:
                    continue
                book_id = str(book_id)
                if book_id not in candidates_map:
                    candidates_map[book_id] = payload
                    payload_map[book_id] = payload
                    vector_score_map[book_id] = 0.0

        candidates: List[Dict[str, Any]] = []
        for book_id, item in candidates_map.items():
            if not item.get("classification") or not item.get("words_total"):
                db_item = self.db.get_item(book_id)
                if db_item:
                    item = {**db_item, **item}
                    candidates_map[book_id] = item
            if "id" not in item or not item["id"]:
                item["id"] = book_id
            candidates.append(item)

        print(f"[Engine] Candidate pool size: {len(candidates)}")

        negative_tag_terms: List[str] = []
        negative_criteria = [
            criteria
            for criteria in parse_result.criteria
            if criteria.name == "semantic_similarity"
            and getattr(criteria, "is_negative", False)
        ]
        for criteria in negative_criteria:
            query_text = self._criteria_params(criteria).get("query_text", "").strip()
            if not query_text:
                continue

            try:
                mapped = self.vs.search_tags(
                    f"tag: {query_text}",
                    limit=1,
                    similarity_threshold=0.7,
                )
            except Exception as exc:
                print(f"[Engine] Warning: negative tag mapping failed: {exc}")
                mapped = []

            if mapped:
                negative_tag_terms.extend(result["tag"] for result in mapped)
            else:
                negative_tag_terms.append(query_text)
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
                "query_vector": query_vector,
                "results": [],
                "message": "No matching novels were found after applying the filters.",
                "engine": "HybridEngine",
                "related_books": related_books,
                "reference_tags": [],
            }

        final_results = scored_items[:limit]

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
            "hypothetical_intro": parse_result.hypothetical_intro,
            "related_books": related_books,
            "reference_tags": [],
            "query_vector": query_vector,
            "results": final_results,
            "engine": "HybridEngine",
        }
