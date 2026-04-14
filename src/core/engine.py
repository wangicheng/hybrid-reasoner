import json
import jieba
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.core.book_matcher import BookMatcher
from src.core.database import Database, BM25Index
from src.core.explainer import generate_explanation
from src.core.llm import parse_query
from src.core.query_preprocessor import (
    build_query_intent,
    apply_negative_boost,
    apply_hard_exclusions,
)
from src.core.vector_store import VectorStore


class HybridEngine:
    """Production search engine using the selected tag-processing strategy."""

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
        self.book_matcher = BookMatcher(self.db, bm25_index=self.bm25_index)
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
    def _normalize_status(target_status: str) -> Optional[str]:
        lowered = target_status.lower()
        completed_keywords = ["complet", "finish", "ended", "done", "完結", "已完結"]
        ongoing_keywords = ["ongoing", "serializ", "running", "active", "連載", "連載中"]

        if any(keyword in lowered or keyword in target_status for keyword in completed_keywords):
            return "完結"
        if any(keyword in lowered or keyword in target_status for keyword in ongoing_keywords):
            return "連載中"
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
        # For items missing from one list, use a very large rank (low contribution)
        default_rank = 99999
        for doc_id in all_ids:
            v_rank = vector_rank_map.get(doc_id, default_rank)
            b_rank = bm25_rank_map.get(doc_id, default_rank)
            fused[doc_id] = 1.0 / (k + v_rank) + 1.0 / (k + b_rank)
        return fused


    def calculate_score(
        self,
        item: Dict[str, Any],
        vector_score: float,
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
        parsed=None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        # 1. 恢復語意相似度作為主權重
        semantic_score = 0.1 + 0.9 * vector_score
        final_score = semantic_score
        breakdown: List[Dict[str, Any]] = [
            {
                "criteria": "semantic_track",
                "label": "Semantic Track",
                "raw_score": vector_score,
                "weighted_score": semantic_score,
                "is_filter": False,
                "reason": f"semantic score {semantic_score:.4f} (raw {vector_score:.4f})",
            }
        ]

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

        # 2. 精準標籤獎勵 (Exact Tag Bonus)
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
                if item.get("publish_status", "") != status_filter:
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
    ) -> Dict[str, Any]:
        if self.all_tags_cache:
            print(
                f"[Engine] Using cached tag list with {len(self.all_tags_cache)} entries."
            )

        parse_result = parse_query(
            user_query,
            model_id=model_id,
            tag_list=self.all_tags_cache,
        )

        # ── 階段一：Pre-Retrieval 意圖解析與查詢重構 ──
        query_intent = build_query_intent(parse_result, user_query)
        # 將結構化意圖回寫到 parse_result（便於下游使用）
        parse_result = parse_result.model_copy(
            update={"query_intent": query_intent}
        )

        reference_tags = self.book_matcher.extract_reference_tags(
            user_query,
            search_terms=parse_result.search_terms,
            reference_books=parse_result.reference_books,
            query_intent=query_intent,
        )

        tag_terms_list = self._dedupe_terms(
            list(parse_result.generated_keywords) + reference_tags[:8]
        )

        tag_mapping_weights: List[Dict[str, float]] = []
        if tag_terms_list:
            print(f"[Engine] Pre-mapping {len(tag_terms_list)} tag facets.")
            tag_mapping_weights = self.vs.batch_map_tags(tag_terms_list)

        # ── 使用淨化後的正向詞取代原始查詢送入 BM25 ──
        # 語意搜尋仍用完整 expanded_terms（向量模型能處理否定語意）
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

        # BM25 查詢使用淨化後的正向詞，避免否定詞汙染 BM25 計分
        sanitized_bm25_terms = query_intent.sanitized_bm25_query or expanded_terms
        print(
            f"[Engine] BM25 淨化查詢: \"{sanitized_bm25_terms}\" "
            f"(原始: \"{base_terms}\")"
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

        # ── BM25 intro recall：使用淨化後的查詢 ──
        bm25_intro_results = self.bm25_index.search_intro(
            sanitized_bm25_terms, top_k=200
        )
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

        # ── RRF fusion of vector + BM25 ranks ──
        # Build rank maps (0-indexed, sorted by score desc)
        vector_sorted = sorted(vector_score_map.items(), key=lambda x: x[1], reverse=True)
        vector_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(vector_sorted)}

        bm25_sorted = sorted(bm25_score_map.items(), key=lambda x: x[1], reverse=True)
        bm25_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_sorted)}

        rrf_scores = self._rrf_fuse(vector_rank_map, bm25_rank_map)

        print(
            f"[Engine] Retrieval: {len(vector_score_map)} vector, "
            f"{len(bm25_score_map)} BM25, "
            f"{len(rrf_scores)} RRF merged"
        )

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

        # ── 從 query_intent 建立負向約束（取代舊的 negative_criteria 邏輯）──
        negative_tag_terms: List[str] = []

        # 從 query_intent.hard_exclusions 提取硬排除標籤
        if query_intent.hard_exclusions:
            for exc in query_intent.hard_exclusions:
                try:
                    mapped = self.vs.search_tags(
                        f"tag: {exc.term}",
                        limit=5,
                        similarity_threshold=0.6,
                    )
                except Exception as e:
                    print(f"[Engine] Warning: hard exclusion tag mapping failed: {e}")
                    mapped = []
                if mapped:
                    negative_tag_terms.extend(result["tag"] for result in mapped)
                else:
                    negative_tag_terms.append(exc.term)

        # 也從舊的 criteria is_negative 補充（向下相容）
        negative_criteria = [
            criteria
            for criteria in parse_result.criteria
            if criteria.name == "semantic_similarity"
            and getattr(criteria, "is_negative", False)
        ]
        for criteria in negative_criteria:
            query_text = self._criteria_params(criteria).get("query_text", "").strip()
            if not query_text or query_text in negative_tag_terms:
                continue
            try:
                mapped = self.vs.search_tags(
                    f"tag: {query_text}",
                    limit=5,
                    similarity_threshold=0.6,
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
            rrf_bonus = rrf_scores.get(book_id, 0.0)
            
            final_score, breakdown = self.calculate_score(
                item,
                vector_score=vector_score,
                tag_terms_list=tag_terms_list,
                tag_mapping_weights=tag_mapping_weights,
                parsed=parse_result,
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

        # ── 階段一延伸：負權重機制 (Negative Boosting) ──
        # 對軟排除命中的文件進行扣分
        if query_intent.soft_exclusions:
            scored_items = apply_negative_boost(
                scored_items,
                query_intent.soft_exclusions,
                self._normalize_tags,
            )

        # ── 硬排除過濾（結合 query_intent + 舊有 post_filter）──
        # 依指示：在結果排序前就徹底剔除，確保硬排除具有最優先的阻斷力

        # 先用 query_intent 的硬排除進行 tag 過濾
        if query_intent.hard_exclusions:
            scored_items = apply_hard_exclusions(
                scored_items,
                query_intent.hard_exclusions,
                self._normalize_tags,
            )

        # 再走舊有的 post_filter（處理 status/author/words/legacy negative_tag_terms）
        scored_items = self._post_filter(
            scored_items,
            parse_result.criteria,
            negative_tag_terms,
        )
        
        # 過濾乾淨後，才進行最終排序
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
            "reference_tags": reference_tags,
            "query_intent": query_intent.model_dump() if query_intent else None,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "HybridEngine",
        }
