import json
import math
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.core.book_matcher import BookMatcher
from src.core.database import Database
from src.core.explainer import generate_explanation
from src.core.llm import parse_query
from src.core.vector_store import VectorStore
from src.core.lexical_store import LexicalStore


class HybridEngine:
    """Production search engine using the fixed production tag-processing path."""

    # Supported BM25 fusion modes:
    #   "multiplicative" : total = base × (1 + bonus_max × bm25_metric)  [original]
    #   "additive"       : total = base + bonus_max × bm25_metric
    #   "log_dampened"   : total = base + bonus_max × log(1 + bm25_metric)
    #   "tiebreaker"     : total = base + bonus_max × bm25_metric  (bonus_max should be ~0.001)
    VALID_FUSION_MODES = {"multiplicative", "additive", "log_dampened", "tiebreaker"}

    def __init__(
        self,
        db: Optional[Database] = None,
        vs: Optional[VectorStore] = None,
        lexical_store: Optional[LexicalStore] = None,
        semantic_weight: Optional[float] = None,
        attribute_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        enable_bm25: Optional[bool] = None,
        bm25_bonus_max: Optional[float] = None,
        bm25_fusion_mode: Optional[str] = None,
    ):
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        
        self.enable_bm25 = enable_bm25 if enable_bm25 is not None else settings.ENABLE_BM25
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.BM25_WEIGHT
        self.bm25_bonus_max = bm25_bonus_max if bm25_bonus_max is not None else settings.BM25_BONUS_MAX
        self.bm25_fusion_mode = bm25_fusion_mode or settings.BM25_FUSION_MODE
        if self.bm25_fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                f"Invalid bm25_fusion_mode '{self.bm25_fusion_mode}'. "
                f"Must be one of: {self.VALID_FUSION_MODES}"
            )
        if self.enable_bm25:
            self.lexical_store = lexical_store if lexical_store is not None else LexicalStore(self.db)
        else:
            self.lexical_store = None
            
        self.book_matcher = BookMatcher(self.db)
        self.semantic_weight = (
            semantic_weight if semantic_weight is not None else settings.SEMANTIC_WEIGHT
        )
        self.attribute_weight = (
            attribute_weight
            if attribute_weight is not None
            else settings.ATTRIBUTE_WEIGHT
        )
        self.all_tags_cache: Optional[Tuple[str, ...]] = None
        self._load_tags_cache()
        if not self.all_tags_cache:
            raise RuntimeError(
                "Tag metadata file 'data/all_tags.json' is missing or empty."
            )

        # Keep the tag embedding collection aligned with the curated whitelist.
        self.vs.sync_tag_collection(self.all_tags_cache)

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

    @staticmethod
    def _extract_recall_tags(
        tag_mapping_weights: List[Dict[str, float]],
        min_score: float = 0.7,
        max_tags_per_term: int = 3,
    ) -> List[str]:
        recall_tags: List[str] = []
        seen = set()

        for mapping in tag_mapping_weights:
            ranked = sorted(mapping.items(), key=lambda item: item[1], reverse=True)
            accepted = 0
            for tag_name, score in ranked:
                if score < min_score:
                    continue
                if tag_name in seen:
                    continue
                seen.add(tag_name)
                recall_tags.append(tag_name)
                accepted += 1
                if accepted >= max_tags_per_term:
                    break

        return recall_tags

    @staticmethod
    def _normalize_bm25_scores(bm25_score_map: Dict[str, float]) -> Dict[str, float]:
        """Normalize positive BM25 scores into [0, 1] using percentile-robust scaling.

        Uses 5th and 95th percentile as boundaries instead of raw min/max to
        prevent a single extreme outlier from compressing all other scores
        towards zero.
        """
        positive_scores = sorted(
            [score for score in bm25_score_map.values() if score > 0]
        )
        if not positive_scores:
            return {}

        if len(positive_scores) == 1:
            return {
                book_id: 1.0 if score > 0 else 0.0
                for book_id, score in bm25_score_map.items()
            }

        # Percentile-robust boundaries (P5 / P95)
        def _percentile(sorted_vals: List[float], pct: float) -> float:
            idx = (len(sorted_vals) - 1) * pct
            lo = int(idx)
            hi = min(lo + 1, len(sorted_vals) - 1)
            frac = idx - lo
            return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

        p5 = _percentile(positive_scores, 0.05)
        p95 = _percentile(positive_scores, 0.95)

        if p95 <= p5:
            # All positive scores are effectively the same
            return {
                book_id: 1.0 if score > 0 else 0.0
                for book_id, score in bm25_score_map.items()
            }

        scale = p95 - p5
        normalized: Dict[str, float] = {}
        for book_id, score in bm25_score_map.items():
            if score <= 0:
                normalized[book_id] = 0.0
                continue
            metric = (score - p5) / scale
            normalized[book_id] = max(0.0, min(1.0, float(metric)))
        return normalized

    def _build_tag_terms_list(
        self,
        generated_keywords: List[str],
    ) -> List[str]:
        return self._dedupe_terms(generated_keywords)

    def _resolve_negative_tag_terms(self, criteria_list: List[Any]) -> List[str]:
        """Negative semantic criteria are only used to resolve blocked tag terms."""
        negative_tag_terms: List[str] = []
        negative_criteria = [
            criteria
            for criteria in criteria_list
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

        return negative_tag_terms

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

    def _extract_hard_constraints(
        self,
        criteria_list: List[Any],
        negative_tag_terms: List[str],
    ) -> Dict[str, Any]:
        """Extract hard constraint filters from parsed criteria.

        Returns a dict with keys:
          - status_filter: Optional[str] ("completed" / "ongoing")
          - author_filter: Optional[str]
          - words_min: Optional[int]
          - words_max: Optional[int]
          - negative_tag_terms: List[str]
        """
        constraints: Dict[str, Any] = {
            "status_filter": None,
            "author_filter": None,
            "words_min": None,
            "words_max": None,
            "negative_tag_terms": list(negative_tag_terms),
        }

        for criteria in criteria_list:
            params = self._criteria_params(criteria)
            if criteria.name == "status_check":
                constraints["status_filter"] = self._normalize_status(
                    params.get("target_status", "")
                )
            elif criteria.name == "author_match":
                val = params.get("author_name", "").strip()
                if val:
                    constraints["author_filter"] = val
            elif criteria.name == "numeric_range" and params.get("field") == "words_total":
                constraints["words_min"] = params.get("min_val")
                constraints["words_max"] = params.get("max_val")

        return constraints

    def _item_violates_hard_constraints(
        self,
        item: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> bool:
        """Check if a single item violates any hard constraint.

        Returns True if the item should be EXCLUDED.
        If metadata is missing, returns False (benefit of the doubt;
        the post-filter safety net will catch it after DB enrichment).
        """
        # 1. Negative tag check
        negative_terms = constraints.get("negative_tag_terms", [])
        if negative_terms:
            book_tags = self._normalize_tags(item.get("tags", []))
            for neg_term in negative_terms:
                if any(
                    neg_term in tag or tag in neg_term
                    for tag in book_tags
                ):
                    return True

        # 2. Status check
        status_filter = constraints.get("status_filter")
        if status_filter:
            item_status_raw = item.get("publish_status", "")
            if item_status_raw:  # Only filter if metadata is present
                item_status = self._normalize_status(item_status_raw)
                if item_status and item_status != status_filter:
                    return True

        # 3. Word count check
        words_min = constraints.get("words_min")
        words_max = constraints.get("words_max")
        if words_min is not None or words_max is not None:
            actual_words = item.get("words_total", 0) or 0
            if actual_words > 0:  # Only filter if metadata is present
                if words_min is not None and actual_words < words_min:
                    return True
                if words_max is not None and actual_words > words_max:
                    return True

        # 4. Author check
        author_filter = constraints.get("author_filter")
        if author_filter:
            author = item.get("author", "")
            if author:  # Only filter if metadata is present
                if not (author_filter in author or author in author_filter):
                    return True

        return False


    def calculate_score(
        self,
        item: Dict[str, Any],
        vector_score: float,
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
        bm25_metric: float = 0.0,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        breakdown: List[Dict[str, Any]] = []

        semantic_score = vector_score
        breakdown.append(
            {
                "criteria": "semantic_track",
                "label": "Semantic Track",
                "raw_score": vector_score,
                "weighted_score": semantic_score,
                "is_filter": False,
                "reason": f"semantic score {semantic_score:.4f}",
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
            attribute_score = average_similarity
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
            base_score = (
                semantic_score * self.semantic_weight
                + attribute_score * self.attribute_weight
            )
            base_fusion_reason = (
                f"({semantic_score:.4f} * {self.semantic_weight}) + "
                f"({attribute_score:.4f} * {self.attribute_weight})"
            )
        else:
            base_score = semantic_score
            base_fusion_reason = f"semantic only: {semantic_score:.4f}"

        safe_bm25_metric = max(0.0, min(1.0, float(bm25_metric)))
        bonus_max = self.bm25_bonus_max
        mode = self.bm25_fusion_mode

        if semantic_score <= 0 or bonus_max <= 0 or safe_bm25_metric <= 0:
            # No BM25 influence: gate on semantic presence
            total_score = base_score
            fusion_reason = (
                f"base={base_fusion_reason}; "
                f"bm25: no boost (mode={mode}, sem={semantic_score:.4f}, "
                f"β={bonus_max:.4f}, metric={safe_bm25_metric:.4f})"
            )
        elif mode == "multiplicative":
            bm25_multiplier = 1.0 + bonus_max * safe_bm25_metric
            total_score = base_score * bm25_multiplier
            fusion_reason = (
                f"base={base_fusion_reason}; "
                f"bm25[mult]=(1+{bonus_max:.4f}*{safe_bm25_metric:.4f})="
                f"{bm25_multiplier:.4f}; final={base_score:.4f}*{bm25_multiplier:.4f}"
            )
        elif mode in ("additive", "tiebreaker"):
            bm25_addend = bonus_max * safe_bm25_metric
            total_score = base_score + bm25_addend
            fusion_reason = (
                f"base={base_fusion_reason}; "
                f"bm25[{mode}]=+{bonus_max:.4f}*{safe_bm25_metric:.4f}="
                f"+{bm25_addend:.6f}; final={total_score:.6f}"
            )
        elif mode == "log_dampened":
            bm25_addend = bonus_max * math.log(1.0 + safe_bm25_metric)
            total_score = base_score + bm25_addend
            fusion_reason = (
                f"base={base_fusion_reason}; "
                f"bm25[log]=+{bonus_max:.4f}*log(1+{safe_bm25_metric:.4f})="
                f"+{bm25_addend:.6f}; final={total_score:.6f}"
            )
        else:
            total_score = base_score
            fusion_reason = f"base={base_fusion_reason}; bm25: unknown mode '{mode}'"

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
        normalized_base_terms = "".join(str(base_terms).split()).lower()
        for criteria in positive_semantic:
            query_text = self._criteria_params(criteria).get("query_text", "").strip()
            normalized_query_text = "".join(query_text.split()).lower()
            if query_text and normalized_query_text != normalized_base_terms:
                semantic_texts.append(query_text)
        if semantic_texts:
            semantic_expansion = " ".join(semantic_texts)
            expanded_terms = f"{expanded_terms} {semantic_expansion}".strip()

        # ── Extract hard constraints early for BM25 pre-filtering ──
        negative_tag_terms = self._dedupe_terms(
            list(parse_result.tag_intent.negative_terms)
        ) or self._resolve_negative_tag_terms(parse_result.criteria)

        hard_constraints = self._extract_hard_constraints(
            parse_result.criteria, negative_tag_terms
        )

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

        bm25_score_map: Dict[str, float] = {}
        bm25_metric_map: Dict[str, float] = {}
        bm25_new_count = 0
        bm25_prefiltered_count = 0
        if self.enable_bm25 and self.lexical_store:
            bm25_results = self.lexical_store.search(expanded_terms, limit=getattr(settings, "TOP_K_BM25", 1000))
            for res in bm25_results:
                item = res["item"]
                book_id = str(item.get("id"))
                if not book_id:
                    continue
                bm25_score_map[book_id] = float(res["score"])
                if book_id not in candidates_map:
                    # Pre-filter: skip BM25-only candidates that violate hard constraints
                    if self._item_violates_hard_constraints(item, hard_constraints):
                        bm25_prefiltered_count += 1
                        continue
                    candidates_map[book_id] = item
                    payload_map[book_id] = item
                    vector_score_map[book_id] = 0.0
                    bm25_new_count += 1
            # Normalize once after collecting all BM25 scores
            bm25_metric_map = self._normalize_bm25_scores(bm25_score_map)
            print(
                f"[Engine] BM25: {len(bm25_results)} results, "
                f"{bm25_new_count} new candidates added, "
                f"{bm25_prefiltered_count} pre-filtered by hard constraints "
                f"(already {len(bm25_score_map) - bm25_new_count - bm25_prefiltered_count} were in pool)"
            )

        if tag_terms_list and tag_mapping_weights:
            recall_tags = self._extract_recall_tags(tag_mapping_weights)
            if recall_tags:
                print(f"[Engine] Triggering mapped-tag recall for {len(recall_tags)} resolved tags.")
                tag_recall_items = self.db.search_by_tags_any(recall_tags, limit=retrieval_limit)
                for item in tag_recall_items:
                    book_id = str(item.get("id", "")).strip()
                    if not book_id:
                        continue
                    if book_id not in candidates_map:
                        candidates_map[book_id] = item
                        payload_map[book_id] = item
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

        print(f"[Engine] Candidate pool size: {len(candidates)}")

        # negative_tag_terms already extracted above (before BM25 search)
        scored_items = []
        for item in candidates:
            book_id = str(item.get("id"))
            if not book_id or book_id == "None":
                continue
            vector_score = vector_score_map.get(book_id, 0.0)
            raw_bm25_score = bm25_score_map.get(book_id, 0.0) if self.enable_bm25 else 0.0
            bm25_metric = bm25_metric_map.get(book_id, 0.0) if self.enable_bm25 else 0.0
            
            final_score, breakdown = self.calculate_score(
                item,
                vector_score,
                tag_terms_list,
                tag_mapping_weights,
                bm25_metric=bm25_metric,
            )
            scored_items.append(
                {
                    "item": item,
                    "score": float(final_score),
                    "vector_score": vector_score,
                    "bm25_score": raw_bm25_score,
                    "bm25_metric": bm25_metric,
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
                "search_terms": parse_result.search_terms,
                "generated_keywords": parse_result.generated_keywords,
                "tag_intent": parse_result.tag_intent.model_dump(),
                "query_vector": query_vector,
                "results": [],
                "message": "No matching novels were found after applying the filters.",
                "engine": "HybridEngine",
                "related_books": related_books,
                "reference_tags": [],
                "parse_metadata": parse_result.parse_metadata,
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
            "tag_intent": parse_result.tag_intent.model_dump(),
            "hypothetical_intro": parse_result.hypothetical_intro,
            "related_books": related_books,
            "reference_tags": [],
            "parse_metadata": parse_result.parse_metadata,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "HybridEngine",
        }
