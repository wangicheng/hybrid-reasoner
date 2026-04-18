import json
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.core.book_matcher import BookMatcher
from src.core.database import Database, BM25Index
from src.core.explainer import generate_explanation
from src.core.llm import parse_query
from src.core.vector_store import VectorStore


class HybridEngine:
    """Production search engine using the fixed production tag-processing path."""

    def __init__(
        self,
        db: Optional[Database] = None,
        vs: Optional[VectorStore] = None,
        semantic_weight: Optional[float] = None,
        attribute_weight: Optional[float] = None,
    ):
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        self.bm25_index = BM25Index(self.db)
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

    def _build_tag_terms_list(
        self,
        generated_keywords: List[str],
    ) -> List[str]:
        return self._dedupe_terms(generated_keywords)

    def _build_semantic_retrieval_text(self, parse_result: Any) -> str:
        base_terms = str(getattr(parse_result, "search_terms", "") or "").strip()
        if not base_terms:
            base_terms = str(getattr(parse_result, "original_query", "") or "").strip()

        positive_semantic = [
            criteria
            for criteria in getattr(parse_result, "criteria", [])
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
            return f"{base_terms} {semantic_expansion}".strip()

        return base_terms

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

    @staticmethod
    def _rrf_fuse(
        vector_rank_map: Dict[str, int],
        bm25_rank_map: Dict[str, int],
        k: int = 60,
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion (RRF).
        Combines two rank maps into a single score map.
        score(d) = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))
        """
        all_ids = set(vector_rank_map.keys()) | set(bm25_rank_map.keys())
        fused: Dict[str, float] = {}
        default_rank = 99999
        for doc_id in all_ids:
            v_rank = vector_rank_map.get(doc_id, default_rank)
            b_rank = bm25_rank_map.get(doc_id, default_rank)
            fused[doc_id] = 1.0 / (k + v_rank) + 1.0 / (k + b_rank)
        return fused

    @staticmethod
    def _rrf_fuse_multi(
        *rank_maps: Dict[str, int],
        k: int = 60,
    ) -> Dict[str, float]:
        """Multi-way Reciprocal Rank Fusion."""
        all_ids: set = set()
        for rank_map in rank_maps:
            all_ids |= set(rank_map.keys())
        fused: Dict[str, float] = {}
        default_rank = 99999
        for doc_id in all_ids:
            score = sum(1.0 / (k + rm.get(doc_id, default_rank)) for rm in rank_maps)
            fused[doc_id] = score
        return fused


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
            final_score = (
                semantic_score * self.semantic_weight
                + attribute_score * self.attribute_weight
            )
            fusion_reason = (
                f"({semantic_score:.4f} * {self.semantic_weight}) + "
                f"({attribute_score:.4f} * {self.attribute_weight})"
            )
        else:
            final_score = semantic_score
            fusion_reason = f"semantic only: {semantic_score:.4f}"

        # 精準標籤獎勵 (Exact Tag Bonus)
        query_words = set(tag_terms_list) if tag_terms_list else set()
        item_tags = set(self._normalize_tags(item.get("tags", [])))
        exact_matches = [w for w in query_words if w in item_tags and len(w) > 1]

        if exact_matches:
            bonus = min(0.3, len(exact_matches) * 0.1)
            final_score += bonus
            breakdown.append({
                "criteria": "tag_exact_match",
                "label": "精準標籤獎勵",
                "reason": f"精準命中標籤: {', '.join(exact_matches)}",
                "raw_score": float(len(exact_matches)),
                "weighted_score": bonus,
                "is_filter": False,
            })
            fusion_reason += f" | Tag Bonus +{bonus:.2f}"

        breakdown.append(
            {
                "criteria": "global_fusion",
                "label": "Global Fusion",
                "raw_score": final_score,
                "weighted_score": final_score,
                "is_filter": False,
                "reason": fusion_reason,
            }
        )

        return final_score, breakdown

    def _post_filter(
        self,
        scored_items: List[Dict[str, Any]],
        criteria_list: List[Any],
        negative_tag_terms: List[str],
        positive_tag_terms: Optional[List[str]] = None,
        required_tag_terms: Optional[List[str]] = None,
        result_limit: int = 5,
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

        # Fix 3: Pre-compute exact whitelist blocked tags (invariant across results)
        blocked_exact = set(negative_tag_terms) & set(self.all_tags_cache or [])

        for result in scored_items:
            item = result["item"]
            excluded = False
            book_tags = self._normalize_tags(item.get("tags", []))

            # Exact whitelist match for negative tags
            if blocked_exact:
                book_tag_set = set(book_tags)
                if blocked_exact & book_tag_set:
                    excluded = True

            # Substring fallback for non-whitelist negative terms
            if not excluded:
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
            f"[PostFilter] Hard filters: {len(scored_items)} -> {len(filtered)} "
            f"(removed {len(scored_items) - len(filtered)})"
        )

        # ── 1. Strict required tags filter ──
        if required_tag_terms and self.all_tags_cache:
            required_set = set(required_tag_terms) & set(self.all_tags_cache)
            if required_set:
                strictly_matched = []
                for result in filtered:
                    book_tags = set(self._normalize_tags(result["item"].get("tags", [])))
                    if required_set.issubset(book_tags):
                        strictly_matched.append(result)
                print(f"[PostFilter] Required tags {required_set} filtered: {len(filtered)} -> {len(strictly_matched)}")
                if len(strictly_matched) > 0:
                    filtered = strictly_matched

        # ── 2. Positive tag coverage-based ranking ──
        target_tag_set: set = set()
        if positive_tag_terms and self.all_tags_cache:
            target_tag_set = set(positive_tag_terms) & set(self.all_tags_cache)

        if target_tag_set:
            # Sort by number of positive tag hits
            for result in filtered:
                book_tags = set(self._normalize_tags(result["item"].get("tags", [])))
                matched_count = len(target_tag_set & book_tags)
                result["_tag_coverage"] = matched_count
            
            # Sort descending by coverage, but preserve original RRF order for ties
            filtered.sort(key=lambda r: r.get("_tag_coverage", 0), reverse=True)

            print(
                f"[PostFilter] Sorted {len(filtered)} results by positive tag coverage (out of {len(target_tag_set)} tags)"
            )

        return filtered[:result_limit]

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

        semantic_retrieval_text = self._build_semantic_retrieval_text(parse_result)

        retrieval_limit = 10000
        candidates_map: Dict[str, Dict[str, Any]] = {}
        vector_score_map: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}

        vector_results, query_vector = self.vs.search(
            semantic_retrieval_text,
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

        # ── HyDE vector recall (secondary channel) ──
        hyde_score_map: Dict[str, float] = {}
        hypothetical_intro = str(getattr(parse_result, "hypothetical_intro", "") or "").strip()
        if hypothetical_intro:
            try:
                hyde_text = f"Intro: {hypothetical_intro}"
                hyde_results, _ = self.vs.search(
                    hyde_text,
                    limit=200,
                    query_filter=None,
                    with_payload=True,
                )
                for hit in hyde_results:
                    payload = hit.get("payload") or {}
                    book_id = payload.get("id")
                    if not book_id:
                        continue
                    book_id = str(book_id)
                    hyde_score_map[book_id] = float(hit["score"])
                    if book_id not in candidates_map:
                        candidates_map[book_id] = payload
                        payload_map[book_id] = payload
                        vector_score_map[book_id] = 0.0
                print(f"[Engine] HyDE recall: {len(hyde_score_map)} candidates")
            except Exception as exc:
                print(f"[Engine] HyDE recall failed: {exc}")

        # ── BM25 查詢淨化：從查詢中移除負向詞，避免汙染 BM25 計分 ──
        tag_intent = parse_result.tag_intent
        bm25_base = parse_result.search_terms or parse_result.original_query
        bm25_query = bm25_base
        if tag_intent.negative_terms:
            for neg_term in tag_intent.negative_terms:
                bm25_query = bm25_query.replace(neg_term, "")
            bm25_query = " ".join(bm25_query.split()).strip() or bm25_base
        print(
            f"[Engine] BM25 查詢: \"{bm25_query}\" "
            f"(原始: \"{bm25_base}\")"
        )

        # ── BM25 intro recall ──
        bm25_intro_results = self.bm25_index.search_intro(bm25_query, top_k=200)
        bm25_score_map: Dict[str, float] = {}
        for result in bm25_intro_results:
            bm25_item = result['item']
            book_id = str(bm25_item.get('id', ''))
            if not book_id or book_id == 'None':
                continue
            bm25_score_map[book_id] = result['bm25_score']
            if book_id not in candidates_map:
                candidates_map[book_id] = bm25_item
                payload_map[book_id] = bm25_item
                vector_score_map[book_id] = 0.0

        # ── RRF fusion of vector + HyDE + BM25 ranks ──
        vector_sorted = sorted(
            vector_score_map.items(), key=lambda x: x[1], reverse=True
        )
        vector_rank_map = {
            doc_id: rank for rank, (doc_id, _) in enumerate(vector_sorted)
        }

        bm25_sorted = sorted(
            bm25_score_map.items(), key=lambda x: x[1], reverse=True
        )
        bm25_rank_map = {
            doc_id: rank for rank, (doc_id, _) in enumerate(bm25_sorted)
        }

        if hyde_score_map:
            hyde_sorted = sorted(
                hyde_score_map.items(), key=lambda x: x[1], reverse=True
            )
            hyde_rank_map = {
                doc_id: rank for rank, (doc_id, _) in enumerate(hyde_sorted)
            }
            rrf_scores = self._rrf_fuse_multi(vector_rank_map, hyde_rank_map, bm25_rank_map)
        else:
            rrf_scores = self._rrf_fuse(vector_rank_map, bm25_rank_map)

        print(
            f"[Engine] Retrieval: {len(vector_score_map)} vector, "
            f"{len(hyde_score_map)} HyDE, "
            f"{len(bm25_score_map)} BM25, "
            f"{len(rrf_scores)} RRF merged"
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

        negative_tag_terms = self._dedupe_terms(
            list(parse_result.tag_intent.negative_terms)
        ) or self._resolve_negative_tag_terms(parse_result.criteria)
        scored_items = []
        for item in candidates:
            book_id = str(item.get("id"))
            if not book_id or book_id == "None":
                continue
            vector_score = vector_score_map.get(book_id, 0.0)
            rrf_bonus = rrf_scores.get(book_id, 0.0)

            final_score, breakdown = self.calculate_score(
                item,
                vector_score=vector_score,
                tag_terms_list=tag_terms_list,
                tag_mapping_weights=tag_mapping_weights,
            )

            # 將 RRF 作為加分項附加
            final_score += rrf_bonus
            breakdown.append({
                "criteria": "rrf_fusion",
                "label": "RRF Fusion Bonus",
                "reason": f"RRF排名加分: {rrf_bonus:.6f}",
                "raw_score": rrf_bonus,
                "weighted_score": rrf_bonus,
                "is_filter": False,
            })

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
        # Separate required_tags for hard filtering and positive_terms for soft sorting
        scored_items = self._post_filter(
            scored_items,
            parse_result.criteria,
            negative_tag_terms,
            positive_tag_terms=list(parse_result.tag_intent.positive_terms),
            required_tag_terms=list(parse_result.tag_intent.required_tags),
            result_limit=limit,
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
                "hypothetical_intro": parse_result.hypothetical_intro,
                "semantic_retrieval_text": semantic_retrieval_text,
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
            "semantic_retrieval_text": semantic_retrieval_text,
            "related_books": related_books,
            "reference_tags": [],
            "parse_metadata": parse_result.parse_metadata,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "HybridEngine",
        }
