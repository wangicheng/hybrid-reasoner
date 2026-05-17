import asyncio
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from src.config import settings
from src.core.book_matcher import BookMatcher
from src.core.database import Database
from src.core.llm import parse_query, route_query_with_llm
from src.core.query_compiler import CompiledQuery, compile_query
from src.core.vector_store import VectorStore
from src.core.lexical_store import LexicalStore


class HybridEngine:
    """Production search engine using the fixed production tag-processing path."""

    # Supported BM25 fusion modes (used when fusion_strategy='weighted'):
    #   "multiplicative" : total = base × (1 + bonus_max × bm25_metric)  [original]
    #   "additive"       : total = base + bonus_max × bm25_metric
    #   "log_dampened"   : total = base + bonus_max × log(1 + bm25_metric)
    #   "tiebreaker"     : total = base + bonus_max × bm25_metric  (bonus_max should be ~0.001)
    VALID_FUSION_MODES = {"multiplicative", "additive", "log_dampened", "tiebreaker"}
    VALID_FUSION_STRATEGIES = {"weighted", "rrf", "auto", "auto_llm"}

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
        fusion_strategy: Optional[str] = None,
        rrf_k: int = 60,
        # ── Dynamic routing parameters (only used when fusion_strategy='auto') ──
        routing_tag_threshold: int = 1,
        routing_weighted_ws: float = 0.35,
        routing_weighted_wa: float = 0.65,
        routing_weighted_bm25: bool = True,
        routing_rrf_bm25: bool = False,
        rerank: Optional[bool] = None,
    ):
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        
        self.fusion_strategy = fusion_strategy or getattr(settings, 'FUSION_STRATEGY', 'auto')
        if self.fusion_strategy not in self.VALID_FUSION_STRATEGIES:
            raise ValueError(
                f"Invalid fusion_strategy '{self.fusion_strategy}'. "
                f"Must be one of: {self.VALID_FUSION_STRATEGIES}"
            )
        self.rrf_k = rrf_k

        # Dynamic routing knobs
        self.routing_tag_threshold = routing_tag_threshold
        self.routing_weighted_ws = routing_weighted_ws
        self.routing_weighted_wa = routing_weighted_wa
        self.routing_weighted_bm25 = routing_weighted_bm25
        self.routing_rrf_bm25 = routing_rrf_bm25

        self.enable_bm25 = enable_bm25 if enable_bm25 is not None else settings.ENABLE_BM25
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.BM25_WEIGHT
        self.bm25_bonus_max = bm25_bonus_max if bm25_bonus_max is not None else settings.BM25_BONUS_MAX
        self.bm25_fusion_mode = bm25_fusion_mode or settings.BM25_FUSION_MODE
        if self.fusion_strategy in ('weighted', 'auto', 'auto_llm') and self.bm25_fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                f"Invalid bm25_fusion_mode '{self.bm25_fusion_mode}'. "
                f"Must be one of: {self.VALID_FUSION_MODES}"
            )
        # Always initialize lexical_store so 'auto'/'auto_llm' mode can toggle BM25 per-query
        if self.enable_bm25 or self.fusion_strategy in ('auto', 'auto_llm'):
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
        self.rerank_enabled = rerank if rerank is not None else settings.RERANK_ENABLED
        self._reranker = None
        self.max_tags_per_term = 3

        # DAT (Dynamic Alpha Tuning)
        self.enable_dat = settings.ENABLE_DAT
        self._dat_router = None  # Lazy init
        self.all_tags_cache: Optional[Tuple[str, ...]] = None
        self.tag_descriptions_cache: Optional[Dict[str, str]] = None
        self._load_tags_cache()
        if not self.all_tags_cache:
            raise RuntimeError(
                "Tag metadata file 'data/all_tags.json' is missing or empty."
            )

        # Keep the tag embedding collection aligned with the curated whitelist.
        self.vs.sync_tag_collection(self.all_tags_cache, tag_descriptions=self.tag_descriptions_cache)

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
        else:
            raise RuntimeError(
                f"Tag metadata file '{tags_path}' is empty or has an unexpected format."
            )

        # Also load descriptions for semantic enhancement (Strategy C)
        desc_path = "data/tag_descriptions.json"
        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                self.tag_descriptions_cache = json.load(f)
            print(f"[Engine] Loaded {len(self.tag_descriptions_cache)} tag descriptions.")
        except Exception as exc:
            print(f"[Engine] Warning: Failed to load tag descriptions from '{desc_path}': {exc}")
            self.tag_descriptions_cache = None

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

    def _extract_recall_tags(
        self,
        tag_mapping_weights: List[Dict[str, float]],
        min_score: float = 0.7,
    ) -> List[str]:
        max_tags_per_term = self.max_tags_per_term

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
        """Normalize positive BM25 scores into [0, 1] using Batch Max Normalization.
        
        Calculates the maximum score in the batch and divides all positive scores
        by this maximum. This preserves the relative score ratios and is standard
        in hybrid search engines.
        """
        if not bm25_score_map:
            return {}

        max_score = max(bm25_score_map.values())
        
        if max_score <= 0:
            return {book_id: 0.0 for book_id in bm25_score_map.keys()}

        normalized: Dict[str, float] = {}
        for book_id, score in bm25_score_map.items():
            if score <= 0:
                normalized[book_id] = 0.0
            else:
                normalized[book_id] = float(score) / max_score
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
                    f"這部作品的類型偏向{query_text}",
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

    def _apply_degradation_step(
        self,
        original_constraints: Dict[str, Any],
        attempt: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Apply graceful degradation strategy based on attempt number.
        
        Returns: (adjusted_constraints, similarity_threshold)
        
        Degradation strategy:
        - Attempt 1 (嚴格模式): All constraints active, similarity_threshold=0.6
        - Attempt 2 (一階放寬): Convert word constraints to soft match (should), similarity_threshold=0.4
        - Attempt 3 (終極放寬): Convert status/author to soft match, keep only semantic + negative tags hard
        """
        constraints = dict(original_constraints)  # Make a copy
        
        if attempt == 1:
            # ── Attempt 1: Strict mode ──
            # All constraints: words, status, author, negative tags
            similarity_threshold = 0.6
            print(f"[Engine] Degradation Attempt 1 (嚴格模式): All constraints active, threshold=0.6")
            
        elif attempt == 2:
            # ── Attempt 2: First relaxation ──
            # Move word counts to soft match
            constraints["soft_words_min"] = constraints.pop("words_min", None)
            constraints["soft_words_max"] = constraints.pop("words_max", None)
            similarity_threshold = 0.4
            print(f"[Engine] Degradation Attempt 2 (一階放寬): Word constraints soft-matched, threshold=0.4")
            
        elif attempt == 3:
            # ── Attempt 3: Ultimate relaxation ──
            constraints["soft_words_min"] = constraints.pop("words_min", None)
            constraints["soft_words_max"] = constraints.pop("words_max", None)
            constraints["soft_status_filter"] = constraints.pop("status_filter", None)
            constraints["soft_author_filter"] = constraints.pop("author_filter", None)
            # Transfer required_tags to soft_required_tags for Python memory Tag Rescue Bonus
            constraints["soft_required_tags"] = constraints.pop("required_tags", [])
            constraints["required_tags"] = []
            similarity_threshold = 0.4
            print(f"[Engine] Degradation Attempt 3 (終極放寬): Author/Status/Words/Tags soft-matched (Tag Rescue active), threshold=0.4")
            
        else:
            # Fallback
            similarity_threshold = 0.4
            
        return constraints, similarity_threshold


    def calculate_score(
        self,
        item: Dict[str, Any],
        vector_score: float,
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
        bm25_metric: float = 0.0,
        iteration_constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        breakdown: List[Dict[str, Any]] = []

        # --- [軌道 A] 內文向量：全域錨點 + 指數放大 ---
        GLOBAL_MIN_VEC = 0.60
        GLOBAL_MAX_VEC = 0.85
        VEC_RANGE = GLOBAL_MAX_VEC - GLOBAL_MIN_VEC
        TAU = 0.1

        norm_text_base = max(0.0, min(1.0, (vector_score - GLOBAL_MIN_VEC) / VEC_RANGE))
        semantic_score = math.exp(norm_text_base / TAU) / math.exp(1.0 / TAU)

        breakdown.append(
            {
                "criteria": "semantic_track",
                "label": "Semantic Track",
                "raw_score": vector_score,
                "weighted_score": semantic_score,
                "is_filter": False,
                "reason": f"semantic raw {vector_score:.4f} -> exp_norm {semantic_score:.4f} (base {norm_text_base:.4f})",
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
            # --- [軌道 C] 標籤向量：全域錨點 + 指數放大 ---
            norm_tag_base = max(0.0, min(1.0, (average_similarity - GLOBAL_MIN_VEC) / VEC_RANGE))
            attribute_score = math.exp(norm_tag_base / TAU) / math.exp(1.0 / TAU)
            
            has_tag_scoring = True
            breakdown.append(
                {
                    "criteria": "attribute_track",
                    "label": "Attribute Track",
                    "raw_score": average_similarity,
                    "weighted_score": attribute_score,
                    "is_filter": False,
                    "reason": (
                        f"facet avg {average_similarity:.4f} -> exp_norm {attribute_score:.4f} (base {norm_tag_base:.4f}); "
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

        # ── Tag Rescue Bonus (Memory Softening) ──
        tag_rescue_bonus = 0.0
        rescue_details = []
        if iteration_constraints and "soft_required_tags" in iteration_constraints:
            soft_tags = iteration_constraints["soft_required_tags"]
            # Gate on semantic relevance: prevent junk/tag-stuffed items from getting rescue bonus
            if soft_tags and semantic_score >= 0.50:
                book_tags = self._normalize_tags(item.get("tags", []))
                for stag in soft_tags:
                    if any(stag in t or t in stag for t in book_tags):
                        tag_rescue_bonus += 0.4
                        rescue_details.append(stag)

        if tag_rescue_bonus > 0:
            total_score += tag_rescue_bonus
            breakdown.append(
                {
                    "criteria": "tag_rescue_bonus",
                    "label": "Tag Rescue Bonus",
                    "raw_score": tag_rescue_bonus,
                    "weighted_score": tag_rescue_bonus,
                    "is_filter": False,
                    "reason": f"soft-matched required tags: {', '.join(rescue_details)} (+{tag_rescue_bonus:.2f})",
                }
            )

        # ── Score Capping Removed ──
        # Removed min(total_score, 1.0) capping (Issue 4) to prevent ranking signal
        # compression when tag rescue bonus and BM25 bonus stack.

        return total_score, breakdown

    def calculate_score_v2(
        self,
        item: Dict[str, Any],
        vector_score: float,
        bm25_metric: float,
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
        alpha: float,
        tag_vector_score: float = 0.0,
        required_tags: Optional[List[str]] = None,
        penalty_multiplier: float = 1.0,
        beta: Optional[float] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """3+1 Layer Scoring Pipeline.

        Layer 1: Base = α × Norm_S_plot(τ=0.1) + (1-α) × Norm_S_BM25
        Layer 2: Bonus = Base × (1 + β × Norm_S_tag(τ=TAG_TAU))
                 β auto-scales dynamically via DAT or fallback logic
                 Batch Map and Track C normalized independently, then fused via max
        Layer 3: Boost = Bonus × (1 + (MULTIPLIER-1) × match_ratio)
                 Multiplicative: preserves L1+L2 ranking signal
        Layer 0: Final = Boost × penalty_multiplier
        """
        breakdown: List[Dict[str, Any]] = []
        
        # Dynamic β: use passed DAT beta, otherwise fallback to heuristic
        if beta is None:
            beta = settings.TAG_BONUS_BETA
            if required_tags and len(required_tags) >= 4:
                beta = max(beta, 0.35)
            elif required_tags and len(required_tags) >= 2:
                beta = max(beta, 0.25)
        req_tag_multiplier = settings.REQUIRED_TAG_MULTIPLIER

        # ── Normalization constants (shared with legacy calculate_score) ──
        GLOBAL_MIN_VEC = 0.60
        GLOBAL_MAX_VEC = 0.85
        VEC_RANGE = GLOBAL_MAX_VEC - GLOBAL_MIN_VEC
        TAU = 0.1
        TAU_TAG = settings.TAG_TAU  # Gentler temperature for Layer 2 tag scoring

        # ══════════════════════════════════════════════════════════════
        # Layer 1: Base Relevance — α × Plot + (1-α) × BM25
        # ══════════════════════════════════════════════════════════════
        norm_plot_base = max(0.0, min(1.0, (vector_score - GLOBAL_MIN_VEC) / VEC_RANGE))
        norm_plot = math.exp(norm_plot_base / TAU) / math.exp(1.0 / TAU)

        norm_bm25 = max(0.0, min(1.0, float(bm25_metric)))

        base_score = alpha * norm_plot + (1.0 - alpha) * norm_bm25

        breakdown.append({
            "criteria": "layer1_base", "label": "L1: Base Relevance",
            "raw_score": base_score, "weighted_score": base_score, "is_filter": False,
            "reason": (
                f"α={alpha:.3f} × plot_exp={norm_plot:.4f} (raw={vector_score:.4f}) + "
                f"(1-α)={1-alpha:.3f} × bm25={norm_bm25:.4f}"
            ),
        })

        # ══════════════════════════════════════════════════════════════
        # Layer 2: Tag Vector Bonus — Base × (1 + β × Norm_S_tag)
        #
        # Batch Map (tag-to-tag similarity) and Track C (tag-query-to-
        # document similarity) operate in different embedding spaces.
        # Normalize each independently before fusing to prevent scale
        # mismatch that would cause one source to dominate unfairly.
        # ══════════════════════════════════════════════════════════════
        # Independent anchors for tag-to-tag batch map space
        BATCH_MAP_MIN = 0.55   # Tag-to-tag floor (below 0.6 threshold after facet averaging)
        BATCH_MAP_MAX = 0.90   # Tag-to-tag ceiling (exact matches reach higher)
        BATCH_MAP_RANGE = BATCH_MAP_MAX - BATCH_MAP_MIN

        norm_tag = 0.0
        tag_detail = "no tag scoring"

        # Source 1: Batch Map — tag-to-tag embedding space
        batch_map_norm = 0.0
        raw_batch_sim = 0.0
        if tag_terms_list and tag_mapping_weights:
            raw_batch_sim = self._compute_attribute_score_for_item(
                item, tag_terms_list, tag_mapping_weights
            )
            if raw_batch_sim > 0:
                batch_base = max(0.0, min(1.0, (raw_batch_sim - BATCH_MAP_MIN) / BATCH_MAP_RANGE))
                batch_map_norm = math.exp(batch_base / TAU_TAG) / math.exp(1.0 / TAU_TAG)
            tag_detail = f"batch_map={raw_batch_sim:.4f}→norm={batch_map_norm:.4f}"

        # Source 2: Track C — tag-query-to-document space (same as Track A)
        track_c_norm = 0.0
        if tag_vector_score > 0:
            tc_base = max(0.0, min(1.0, (tag_vector_score - GLOBAL_MIN_VEC) / VEC_RANGE))
            track_c_norm = math.exp(tc_base / TAU_TAG) / math.exp(1.0 / TAU_TAG)
            tag_detail += f" | track_c={tag_vector_score:.4f}→norm={track_c_norm:.4f}"

        # Fuse: max of independently normalized scores
        norm_tag = max(batch_map_norm, track_c_norm)
        if norm_tag > 0:
            tag_detail += f" → fused={norm_tag:.4f} (τ_tag={TAU_TAG})"

        tag_multiplier = 1.0 + beta * norm_tag
        bonus_score = base_score * tag_multiplier

        breakdown.append({
            "criteria": "layer2_tag_bonus", "label": "L2: Tag Bonus",
            "raw_score": norm_tag, "weighted_score": bonus_score, "is_filter": False,
            "reason": f"base × (1 + β={beta:.2f} × tag={norm_tag:.4f}) = {base_score:.4f} × {tag_multiplier:.4f}; {tag_detail}",
        })

        # ══════════════════════════════════════════════════════════════
        # Layer 3: Required Tag Boost — +BOOST if has required tag
        # ══════════════════════════════════════════════════════════════
        boost_mult = 1.0
        boost_reason = "no required tags"
        if required_tags:
            book_tags = self._normalize_tags(item.get("tags", []))
            matched_req = [
                rt for rt in required_tags
                if any(rt in t or t in rt for t in book_tags)
            ]
            if matched_req:
                match_ratio = len(matched_req) / len(required_tags)
                boost_mult = 1.0 + (req_tag_multiplier - 1.0) * match_ratio
                boost_reason = f"matched: {', '.join(matched_req)} ({len(matched_req)}/{len(required_tags)}) → ×{boost_mult:.3f}"

        boosted_score = bonus_score * boost_mult
        breakdown.append({
            "criteria": "layer3_boost", "label": "L3: Required Tag Boost",
            "raw_score": boost_mult, "weighted_score": boosted_score, "is_filter": False,
            "reason": boost_reason,
        })

        # ══════════════════════════════════════════════════════════════
        # Layer 0: Penalty & Finalization
        # ══════════════════════════════════════════════════════════════
        final_score = boosted_score * penalty_multiplier

        if penalty_multiplier < 1.0:
            breakdown.append({
                "criteria": "layer0_penalty", "label": "L0: Penalty",
                "raw_score": penalty_multiplier, "weighted_score": final_score, "is_filter": False,
                "reason": f"penalty_multiplier={penalty_multiplier:.3f}",
            })

        return final_score, breakdown


    def _compute_attribute_score_for_item(
        self,
        item: Dict[str, Any],
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
    ) -> float:
        """Compute attribute (tag-matching) score for a single item."""
        if not tag_terms_list or not tag_mapping_weights:
            return 0.0
        book_tags = self._normalize_tags(item.get("tags", []))
        if not book_tags:
            return 0.0
        total_facet_score = 0.0
        for facet_map in tag_mapping_weights:
            best_score = 0.0
            for book_tag in book_tags:
                similarity = facet_map.get(book_tag, 0.0)
                if similarity > best_score:
                    best_score = similarity
            total_facet_score += best_score
        return total_facet_score / len(tag_mapping_weights)

    def _rrf_fuse(
        self,
        candidates: List[Dict[str, Any]],
        vector_score_map: Dict[str, float],
        bm25_score_map: Dict[str, float],
        tag_terms_list: List[str],
        tag_mapping_weights: List[Dict[str, float]],
        payload_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion across semantic, attribute, and BM25 channels.

        RRF score = sum_channel( 1 / (k + rank_in_channel) )
        where k is self.rrf_k (default 60).
        """
        k = self.rrf_k

        # 1. Compute per-item raw scores for each channel
        book_ids = []
        sem_scores: Dict[str, float] = {}
        attr_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        for item in candidates:
            book_id = str(item.get("id", ""))
            if not book_id or book_id == "None":
                continue
            book_ids.append(book_id)
            sem_scores[book_id] = vector_score_map.get(book_id, 0.0)
            attr_scores[book_id] = self._compute_attribute_score_for_item(
                item, tag_terms_list, tag_mapping_weights
            )
            bm25_scores[book_id] = bm25_score_map.get(book_id, 0.0)

        # 2. Build per-channel rank maps (1-indexed, ties get same rank)
        def _build_rank_map(scores: Dict[str, float]) -> Dict[str, int]:
            sorted_ids = sorted(scores.keys(), key=lambda bid: scores[bid], reverse=True)
            rank_map: Dict[str, int] = {}
            for rank_idx, bid in enumerate(sorted_ids):
                rank_map[bid] = rank_idx + 1
            return rank_map

        sem_rank = _build_rank_map(sem_scores)
        attr_rank = _build_rank_map(attr_scores)
        bm25_rank = _build_rank_map(bm25_scores)

        absent_rank = len(book_ids) + 1  # For items missing from a channel

        # 3. Compute RRF score
        rrf_scores: Dict[str, float] = {}
        for book_id in book_ids:
            r_sem = sem_rank.get(book_id, absent_rank)
            r_attr = attr_rank.get(book_id, absent_rank)
            r_bm25 = bm25_rank.get(book_id, absent_rank)
            rrf_scores[book_id] = (
                1.0 / (k + r_sem)
                + 1.0 / (k + r_attr)
                + 1.0 / (k + r_bm25)
            )

        # 4. Build scored_items in the same format as the weighted path
        item_map = {str(item.get("id", "")): item for item in candidates}
        scored_items = []
        for book_id in book_ids:
            item = item_map.get(book_id)
            if not item:
                continue
            rrf_score = rrf_scores[book_id]
            r_sem = sem_rank.get(book_id, absent_rank)
            r_attr = attr_rank.get(book_id, absent_rank)
            r_bm25 = bm25_rank.get(book_id, absent_rank)

            breakdown = [
                {
                    "criteria": "rrf_fusion",
                    "label": "RRF Fusion",
                    "raw_score": rrf_score,
                    "weighted_score": rrf_score,
                    "is_filter": False,
                    "reason": (
                        f"RRF(k={k}): sem_rank={r_sem} attr_rank={r_attr} bm25_rank={r_bm25} | "
                        f"sem={sem_scores.get(book_id, 0):.4f} "
                        f"attr={attr_scores.get(book_id, 0):.4f} "
                        f"bm25={bm25_scores.get(book_id, 0):.2f}"
                    ),
                }
            ]
            scored_items.append({
                "item": item,
                "score": rrf_score,
                "vector_score": sem_scores.get(book_id, 0.0),
                "bm25_score": bm25_scores.get(book_id, 0.0),
                "bm25_metric": 0.0,
                "breakdown": breakdown,
                "payload": payload_map.get(book_id, {}),
            })

        scored_items.sort(key=lambda r: r["score"], reverse=True)
        print(
            f"[Engine] RRF fusion complete: {len(scored_items)} candidates, "
            f"k={k}, top score={scored_items[0]['score']:.6f}" if scored_items else
            f"[Engine] RRF fusion: no candidates"
        )
        return scored_items

    def _tag_matches_blocked(self, blocked_term: str, book_tag: str) -> bool:
        """檢查 book_tag 是否違反 blocked_term，使用詞邊界感知的匹配。
        
        匹配優先順序:
        1. 精確匹配 (case-insensitive)
        2. 子字符串匹配 (僅在完整詞邊界上)
        
        詞邊界包括: 開始/結束位置、空格、連字符、下劃線
        """
        blocked = str(blocked_term).strip()
        tag = str(book_tag).strip()
        
        if not blocked or not tag:
            return False
        
        blocked_lower = blocked.lower()
        tag_lower = tag.lower()
        
        # 1. 精確匹配 (首選)
        if blocked_lower == tag_lower:
            return True
        
        # 2. 子字符串匹配 (詞邊界感知)
        if blocked_lower in tag_lower:
            idx = tag_lower.find(blocked_lower)
            if idx < 0:
                return False
            
            # 檢查左邊界: 必須在開始或分隔符後
            if idx > 0 and tag_lower[idx - 1] not in ' -_/\\，。':
                return False
            
            # 檢查右邊界: 必須在結束或分隔符前
            end_idx = idx + len(blocked_lower)
            if end_idx < len(tag_lower) and tag_lower[end_idx] not in ' -_/\\，。':
                return False
            
            return True
        
        return False

    def _post_filter(
        self,
        scored_items: List[Dict[str, Any]],
        iteration_constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """HyST Safety-Net Post-Filter.

        Under HyST, Qdrant pre-filters enforce status/words/negative-tags at
        the database level. This post-filter exists as a safety net for:
        1. Author matching (Qdrant lacks substring index)
        2. Boundary-aware negative tag matching (Qdrant MatchValue is exact-only)
        3. Required tag enforcement
        4. Catch any BM25/tag-recall candidates that bypassed Qdrant pre-filter
        
        It relies on iteration_constraints so that if Graceful Degradation
        removes a constraint, the post-filter correctly relaxes as well.
        """
        filtered: List[Dict[str, Any]] = []

        author_filter = iteration_constraints.get("author_filter")
        status_filter = iteration_constraints.get("status_filter")
        words_min = iteration_constraints.get("words_min")
        words_max = iteration_constraints.get("words_max")
        negative_tag_terms = iteration_constraints.get("negative_tag_terms", [])
        required_tags = iteration_constraints.get("required_tags", [])

        for result in scored_items:
            item = result["item"]
            excluded = False
            book_tags = self._normalize_tags(item.get("tags", []))

            # Check required_tags: book must contain ALL required tags
            if not excluded and required_tags:
                for req_tag in required_tags:
                    if not any(req_tag == tag for tag in book_tags):
                        excluded = True
                        break

            # Check negative_tags (blocked tags) using improved matching
            # NOTE: This uses boundary-aware matching which is stricter than
            # Qdrant's exact MatchValue — catches partial/fuzzy violations
            if not excluded and negative_tag_terms:
                for negative_term in negative_tag_terms:
                    # Use improved boundary-aware matching
                    if any(
                        self._tag_matches_blocked(negative_term, book_tag)
                        for book_tag in book_tags
                    ):
                        excluded = True
                        break

            # Author check — primary post-filter duty (Qdrant can't do this)
            if not excluded and author_filter:
                author = item.get("author", "")
                if not (author_filter in author or author in author_filter):
                    excluded = True

            # Safety-net: status & word count for non-Qdrant candidates
            if not excluded and status_filter:
                item_status = self._normalize_status(item.get("publish_status", ""))
                if item_status != status_filter:
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
            f"[PostFilter:HyST] {len(scored_items)} -> {len(filtered)} "
            f"(removed {len(scored_items) - len(filtered)})"
        )
        return filtered

    def _determine_routing_strategy(
        self,
        parse_result: Any,
        hard_constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Intent-Aware Dynamic Routing (HyST: filter/score separated).

        Hard constraints (status, words, author, negative tags) are already
        enforced as Qdrant payload pre-filters — they do NOT influence the
        fusion strategy decision.

        Routing is purely based on semantic/tag intent:
        - Tag-heavy queries -> Weighted (倚天劍模式)
        - Atmosphere/semantic queries -> RRF (屠龍刀模式)

        Returns a config dict consumed by the scoring pipeline.
        """
        pos_tags_count = len(parse_result.tag_intent.positive_terms)
        neg_tags_count = len(parse_result.tag_intent.negative_terms)

        # HyST: routing decision is based ONLY on tag/semantic signal,
        # NOT on the presence of hard constraints (which are pre-filtered).
        is_tag_heavy = (
            pos_tags_count >= self.routing_tag_threshold
            or neg_tags_count > 0
        )

        if is_tag_heavy:
            strategy = {
                "fusion": "weighted",
                "ws": self.routing_weighted_ws,
                "wa": self.routing_weighted_wa,
                "enable_bm25": self.routing_weighted_bm25,
                "reason": (
                    f"倚天劍 (Weighted ws={self.routing_weighted_ws} wa={self.routing_weighted_wa}): "
                    f"pos_tags={pos_tags_count}, "
                    f"neg_tags={neg_tags_count}, tag-heavy query"
                ),
            }
        else:
            strategy = {
                "fusion": "rrf",
                "rrf_k": self.rrf_k,
                "enable_bm25": self.routing_rrf_bm25,
                "reason": (
                    f"屠龍刀 (RRF k={self.rrf_k} bm25={self.routing_rrf_bm25}): "
                    f"pos_tags={pos_tags_count}, "
                    f"neg_tags={neg_tags_count}, semantic-dominant query"
                ),
            }

        print(f"[Router] {strategy['reason']}")
        return strategy

    def _determine_routing_strategy_llm(
        self,
        user_query: str,
        parse_result: Any,
        hard_constraints: Dict[str, Any],
        model_id: Optional[str] = None,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """LLM-as-Router: let the LLM decide the fusion strategy.

        Calls `route_query_with_llm` for an independent routing judgment,
        then maps the result to the engine config format.
        """
        llm_decision = route_query_with_llm(
            user_query,
            model_id=model_id,
            cache_namespace=cache_namespace,
        )

        chosen_strategy = llm_decision["strategy"]
        confidence = llm_decision.get("confidence", 0.5)
        reasoning = llm_decision.get("reasoning", "")

        if chosen_strategy == "weighted":
            strategy = {
                "fusion": "weighted",
                "ws": self.routing_weighted_ws,
                "wa": self.routing_weighted_wa,
                "enable_bm25": self.routing_weighted_bm25,
                "reason": (
                    f"倚天劍 (LLM Router → Weighted ws={self.routing_weighted_ws} "
                    f"wa={self.routing_weighted_wa}): "
                    f"confidence={confidence:.2f}, {reasoning[:60]}"
                ),
                "llm_routing": llm_decision,
            }
        else:
            strategy = {
                "fusion": "rrf",
                "rrf_k": self.rrf_k,
                "enable_bm25": self.routing_rrf_bm25,
                "reason": (
                    f"屠龍刀 (LLM Router → RRF k={self.rrf_k} "
                    f"bm25={self.routing_rrf_bm25}): "
                    f"confidence={confidence:.2f}, {reasoning[:60]}"
                ),
                "llm_routing": llm_decision,
            }

        print(f"[Router:LLM] {strategy['reason']}")
        return strategy

    def _get_reranker(self):
        """Lazy-initialise the PermSC reranker on first use."""
        if self._reranker is None:
            from src.core.reranker import PermSCReranker
            self._reranker = PermSCReranker(
                model_id=settings.RERANK_MODEL_ID,
                n_permutations=settings.RERANK_PERMUTATIONS,
                shuffle_seed=getattr(settings, "RERANK_SHUFFLE_SEED", None),
                max_attempts_per_permutation=getattr(settings, "RERANK_MAX_ATTEMPTS_PER_PERM", 5),
            )
        return self._reranker

    async def _rerank_results(
        self,
        scored_items: List[Dict[str, Any]],
        user_query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Apply PermSC reranking to the post-filtered candidate pool."""
        reranker = self._get_reranker()
        candidates = []
        for rank, result in enumerate(scored_items):
            item = result["item"]
            book_id = str(item.get("id", "")).strip()
            if not book_id:
                continue
            candidates.append({
                "book_id": book_id,
                "name": item.get("name", ""),
                "author": item.get("author") or item.get("user", {}).get("name", ""),
                "tags": item.get("tags", []),
                "intro": item.get("intro", ""),
                "words_total": item.get("words_total", 0),
                "publish_status": item.get("publish_status", ""),
                "original_rank": rank + 1,
            })

        print(f"[Reranker] PermSC reranking {len(candidates)} candidates...")
        reranked = await reranker.rerank(user_query, candidates)

        # Rebuild scored_items in reranked order
        id_to_result = {str(r["item"].get("id", "")).strip(): r for r in scored_items}
        reranked_results = []
        for c in reranked:
            original = id_to_result.get(c["book_id"])
            if original:
                reranked_results.append(original)
        return reranked_results

    # ── Tri-track Architecture Constants ──
    SOFT_SCORE_ALPHA = 0.25   # Tag Match Ratio multiplier (legacy, kept for _soft_score_candidates)
    FAST_PATH_RETRIEVAL_LIMIT = 300  # Track A (Vector) & B (BM25) each recall this many
    TAG_VECTOR_RETRIEVAL_LIMIT = 300  # Track C (Tag Vector) recall limit
    EXCEPTION_L1_EXTRA = 1000         # Level 1: wider recall budget (same filters)
    EXCEPTION_L2_THRESHOLD_MULT = 3   # Exception path triggers when results < limit × this
    NEGATIVE_TAG_PENALTY = 0.1        # Legacy constant, kept for backward compat

    def _run_dat_scout(
        self,
        query: str,
        expanded_terms: str,
        metadata_filter: Any,
        tag_terms_list: List[str],
        model_id: Optional[str] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """DAT Scout Phase: fetch Top-1 from Vector + BM25 + TagVector, ask LLM for α and β.

        Returns (alpha, beta, dat_info_dict).
        """
        from src.core.dat_router import DATRouter

        # Lazy-init the router
        if self._dat_router is None:
            self._dat_router = DATRouter(
                default_alpha=settings.DAT_DEFAULT_ALPHA,
                timeout_ms=settings.DAT_TIMEOUT_MS,
                short_query_threshold=settings.DAT_SHORT_QUERY_THRESHOLD,
                short_query_alpha=settings.DAT_SHORT_QUERY_ALPHA,
                model_id=settings.DAT_MODEL_ID,
            )

        fallback_beta = settings.TAG_BONUS_BETA
        default_info = {"enabled": True, "alpha": None, "beta": None, "source": "fallback", "latency_ms": 0.0}

        try:
            # Scout: Vector Top-1
            vec_results, _ = self.vs.search(
                expanded_terms, limit=1, query_filter=metadata_filter, with_payload=True
            )
            if not vec_results:
                print("[DAT:Scout] No vector results, using default α")
                alpha = settings.DAT_DEFAULT_ALPHA
                default_info["source"] = "no_vector_results"
                default_info["alpha"] = alpha
                default_info["beta"] = fallback_beta
                return alpha, fallback_beta, default_info

            # Scout: BM25 Top-1
            bm25_results = self.lexical_store.search(expanded_terms, limit=1) if self.lexical_store else []
            if not bm25_results:
                print("[DAT:Scout] No BM25 results, using default α")
                alpha = settings.DAT_DEFAULT_ALPHA
                default_info["source"] = "no_bm25_results"
                default_info["alpha"] = alpha
                default_info["beta"] = fallback_beta
                return alpha, fallback_beta, default_info

            # Scout: Tag Vector Top-1
            tag_results = []
            if tag_terms_list:
                tag_query = " ".join(tag_terms_list)
                tag_results, _ = self.vs.search(
                    tag_query, limit=1, query_filter=metadata_filter, with_payload=True
                )
            if not tag_results:
                tag_results = [{}] # Dummy empty result so unpacking works

            # Call DAT Router
            alpha, beta, info = self._dat_router.get_dynamic_alpha_and_beta(
                query, vec_results[0], bm25_results[0], tag_results[0], model_id=model_id
            )
            info["alpha"] = alpha
            info["beta"] = beta
            return alpha, beta, info

        except Exception as exc:
            print(f"[DAT:Scout] Error: {exc}, using default α")
            alpha = settings.DAT_DEFAULT_ALPHA
            default_info["source"] = "error"
            default_info["alpha"] = alpha
            default_info["beta"] = fallback_beta
            return alpha, fallback_beta, default_info


    def _soft_score_candidates(self, scored_items: List[Dict[str, Any]], positive_tags: List[str]) -> List[Dict[str, Any]]:
        """In-Memory Soft Scoring: Final_Score = Base × (1 + α × MatchRatio)."""
        if not positive_tags:
            return scored_items
        total = len(positive_tags)
        for r in scored_items:
            book_tags = self._normalize_tags(r["item"].get("tags", []))
            matched = [pt for pt in positive_tags if any(pt in t or t in pt for t in book_tags)]
            ratio = len(matched) / total
            if ratio > 0:
                base = r["score"]
                mult = 1.0 + self.SOFT_SCORE_ALPHA * ratio
                r["score"] = min(base * mult, 1.0)
                r["breakdown"].append({"criteria": "soft_tag_bonus", "label": "Soft Tag Bonus", "raw_score": ratio, "weighted_score": r["score"] - base, "is_filter": False, "reason": f"MatchRatio={ratio:.2f} ({len(matched)}/{total}), α={self.SOFT_SCORE_ALPHA}, ×{mult:.3f}"})
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items

    def _run_retrieval_pipeline(self, expanded_terms, metadata_filter, constraint_dict, tag_terms_list, tag_mapping_weights, active_bm25, active_fusion, active_ws, active_wa, active_rrf_k, active_alpha=0.5, vector_limit=500, bm25_limit=500, tag_vector_limit=300, positive_tags=None, active_beta=None, recall_tags_override: Optional[List[str]] = None):
        """Tri-track retrieval + scoring pipeline.

        Track A: Content vector search (semantic similarity)
        Track B: BM25 lexical search (keyword matching)
        Track C: Tag vector search (attribute affinity)

        Returns (scored_items, query_vector, recall_tags).
        """
        candidates_map, vector_score_map, payload_map = {}, {}, {}
        tag_vector_score_map: Dict[str, float] = {}  # Track C scores

        # ── Track A: Content Vector Search ──
        vector_results, query_vector = self.vs.search(expanded_terms, limit=vector_limit, query_filter=metadata_filter, with_payload=True)
        for hit in vector_results:
            payload = hit.get("payload") or {}
            bid = payload.get("id")
            if not bid: continue
            bid = str(bid)
            candidates_map[bid] = payload; payload_map[bid] = payload; vector_score_map[bid] = float(hit["score"])
        print(f"[Engine:TrackA] Vector recall: {len(vector_results)} hits")

        # ── Track B: BM25 Lexical Search ──
        bm25_score_map, bm25_metric_map = {}, {}
        if active_bm25 and self.lexical_store:
            bm25_results = self.lexical_store.search(expanded_terms, limit=bm25_limit)
            for res in bm25_results:
                item = res["item"]; bid = str(item.get("id"))
                if not bid: continue
                bm25_score_map[bid] = float(res["score"])
                if bid not in candidates_map:
                    if self._item_violates_hard_constraints(item, constraint_dict): continue
                    candidates_map[bid] = item; payload_map[bid] = item; vector_score_map[bid] = 0.0
            bm25_metric_map = self._normalize_bm25_scores(bm25_score_map)
            print(f"[Engine:TrackB] BM25 recall: {len(bm25_results)} hits")

        # ── Track C: Tag Vector Search (Tri-track) ──
        tag_query_terms = positive_tags or tag_terms_list
        if tag_query_terms and tag_vector_limit > 0:
            try:
                tag_queries = [f"這部作品的類型偏向{t}" for t in tag_query_terms]
                tag_vector_results = self.vs.search_individual(
                    tag_queries, limit=tag_vector_limit,
                    query_filter=metadata_filter,
                )
                tag_new_count = 0
                for hit in tag_vector_results:
                    bid = str(hit.get("id", "")).strip()
                    if not bid: continue
                    tag_vector_score_map[bid] = float(hit.get("score", 0.0))
                    hit_payload = hit.get("payload") or {}
                    if bid not in candidates_map:
                        item = hit_payload
                        if not item.get("id"): item["id"] = bid
                        if self._item_violates_hard_constraints(item, constraint_dict): continue
                        candidates_map[bid] = item; payload_map[bid] = item
                        vector_score_map[bid] = 0.0
                        tag_new_count += 1
                print(f"[Engine:TrackC] Tag vector recall: {len(tag_vector_results)} hits, {tag_new_count} new candidates")
            except Exception as exc:
                print(f"[Engine:TrackC] Tag vector search failed, skipping: {exc}")

        # ── Tag-based exact recall (legacy augmentation) ──
        recall_tags = recall_tags_override or []
        if not recall_tags and tag_terms_list and tag_mapping_weights:
            recall_tags = self._extract_recall_tags(tag_mapping_weights)
        if recall_tags:
            for item in self.db.search_by_tags_any(recall_tags, limit=vector_limit):
                bid = str(item.get("id", "")).strip()
                if not bid or bid in candidates_map: continue
                if self._item_violates_hard_constraints(item, constraint_dict): continue
                candidates_map[bid] = item; payload_map[bid] = item; vector_score_map[bid] = 0.0

        # ── Enrich & validate candidates ──
        candidates = []
        for bid, item in candidates_map.items():
            if not item.get("classification") or not item.get("words_total"):
                db_item = self.db.get_item(bid)
                if db_item: item = {**db_item, **item}; candidates_map[bid] = item
            if self._item_violates_hard_constraints(item, constraint_dict): continue
            if "id" not in item or not item["id"]: item["id"] = bid
            if not (str(item.get("name","")).strip() or str(item.get("intro","")).strip() or item.get("words_total") or item.get("tags") or str(item.get("classification","")).strip()): continue
            candidates.append(item)
        print(f"[Engine] Candidate pool (tri-track merged): {len(candidates)}")

        # ── Cross-track imputation: candidates missing from a track get 0 ──
        # Using 0 (not batch-min) prevents "ghost scores" where BM25-only
        # candidates receive an artificially high semantic score from the
        # batch minimum (typically 0.5+), which produces a non-trivial
        # plot contribution after exponential normalization.

        # ── Score candidates (3+1 Layer Pipeline for weighted, RRF unchanged) ──
        if active_fusion == "rrf":
            scored = self._rrf_fuse(candidates, vector_score_map, bm25_score_map, tag_terms_list, tag_mapping_weights, payload_map)
        else:
            scored = []
            for item in candidates:
                bid = str(item.get("id"))
                if not bid or bid == "None": continue
                vs = vector_score_map.get(bid, 0.0)
                bm_metric = bm25_metric_map.get(bid, 0.0) if active_bm25 else 0.0
                tv_score = tag_vector_score_map.get(bid, 0.0)

                fs, bd = self.calculate_score_v2(
                    item, vs, bm_metric,
                    tag_terms_list, tag_mapping_weights,
                    alpha=active_alpha,
                    tag_vector_score=tv_score,
                    required_tags=positive_tags,
                    beta=active_beta,
                )

                bm_score = bm25_score_map.get(bid, 0.0) if active_bm25 else 0.0
                scored.append({"item": item, "score": float(fs), "vector_score": vs, "bm25_score": bm_score, "bm25_metric": bm_metric, "tag_vector_score": tv_score, "breakdown": bd, "payload": payload_map.get(bid, {})})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored, query_vector, recall_tags

    def _exception_path_triage(self, scored_items, negative_tag_terms, positive_tags, constraint_dict=None, min_results=1000):
        """Tiered triage with violation-specific penalties.

        Tier1: Filter out books with negative tags (clean pool).
        Tier2: If not enough clean results, re-admit rejected books with
               violation-specific penalty multipliers applied to their scores.

        Penalty multipliers (from config, multiplicative stacking):
          - blocked_tags:     ×0.1 (死罪)
          - required_status:  ×0.5 (中罪)
          - required_tags:    ×0.8 (輕罪)

        Returns (items, level).
        """
        tier1, rejected = [], []
        for r in scored_items:
            tags = self._normalize_tags(r["item"].get("tags", []))
            if any(any(self._tag_matches_blocked(nt, bt) for bt in tags) for nt in negative_tag_terms):
                rejected.append(r)
            else:
                tier1.append(r)
        print(f"[ExceptionPath:Tier1] {len(tier1)} clean, {len(rejected)} rejected")
        if len(tier1) >= min_results:
            return tier1, 1

        # ── Tier 2: Re-admit with violation-specific penalties ──
        print(f"[ExceptionPath:Tier2] Applying tiered penalty to {len(rejected)} items")
        p_blocked = settings.PENALTY_BLOCKED_TAGS
        p_status = settings.PENALTY_REQUIRED_STATUS
        p_tags = settings.PENALTY_REQUIRED_TAGS
        status_filter = (constraint_dict or {}).get("status_filter")
        required_tags_list = positive_tags or []

        for r in rejected:
            item = r["item"]
            penalty = 1.0
            violations = []

            # Check: blocked tags (always true for rejected items)
            penalty *= p_blocked
            violations.append(f"blocked_tags(×{p_blocked})")

            # Check: status violation
            if status_filter:
                item_status = self._normalize_status(item.get("publish_status", ""))
                if item_status and item_status != status_filter:
                    penalty *= p_status
                    violations.append(f"required_status(×{p_status})")

            # Check: required tags missing
            if required_tags_list:
                book_tags = self._normalize_tags(item.get("tags", []))
                if not any(any(rt in t or t in rt for t in book_tags) for rt in required_tags_list):
                    # ── Semantic Immunity (語意免疫動態防護網) ──
                    # 若語意向量分數極高，視為「良性遺漏」，動態減輕缺標籤的懲罰乘數
                    # 聯合判斷: Track A (劇情語意) + Track C (標籤向量) 取最高
                    vec_score = r.get("vector_score", 0.0)
                    tag_vec_score = r.get("tag_vector_score", 0.0)
                    combined_immunity = max(vec_score, tag_vec_score * 0.9)
                    current_p_tags = p_tags
                    if combined_immunity >= 0.85:
                        current_p_tags = max(p_tags, 0.95)
                    elif combined_immunity >= 0.80:
                        current_p_tags = max(p_tags, 0.90)
                        
                    penalty *= current_p_tags
                    violations.append(f"required_tags(×{current_p_tags}, immunity={combined_immunity:.3f})")

            r["score"] *= penalty
            r["breakdown"].append({
                "criteria": "violation_penalty", "label": "Violation Penalty",
                "raw_score": penalty, "weighted_score": r["score"], "is_filter": False,
                "reason": f"Tier 2: score × {penalty:.4f} [{', '.join(violations)}]",
            })

        all_items = tier1 + rejected
        all_items.sort(key=lambda r: r["score"], reverse=True)
        return all_items, 2

    async def search(
        self,
        user_query: str,
        limit: int = 100,
        model_id: Optional[str] = None,
        explain: bool = True,
        cache_namespace: Optional[str] = None,
        progress_callback: Optional[Callable[[str, Any], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """HyST v2 Dual-Path Search: Fast Path → Exception Path."""
        if self.all_tags_cache:
            print(f"[Engine] Using cached tag list with {len(self.all_tags_cache)} entries.")

        related_books = self.book_matcher.extract_related_books(user_query)
        related_book_context = self.book_matcher.build_related_book_context(related_books)
        
        # Define a thread-safe sync wrapper for progress_callback to emit events to the main event loop
        loop = asyncio.get_running_loop()
        def sync_progress_callback(step: str, data: Any):
            if progress_callback:
                asyncio.run_coroutine_threadsafe(
                    progress_callback(step, data),
                    loop
                )

        from src.core.llm import register_parser_callback, unregister_parser_callback
        register_parser_callback(user_query, sync_progress_callback)
        try:
            parse_result = await asyncio.to_thread(
                parse_query,
                user_query,
                model_id=model_id,
                cache_namespace=cache_namespace,
                tag_list=self.all_tags_cache,
                reference_book_context=related_book_context
            )
        finally:
            unregister_parser_callback(user_query)

        # ── Step 2.5: Query Compiler ──
        exact_neg_terms = self._dedupe_terms(list(parse_result.tag_intent.negative_terms)) if hasattr(parse_result, "tag_intent") else []
        fuzzy_neg_terms = self._dedupe_terms(list(parse_result.tag_intent.fuzzy_negative_terms)) if hasattr(parse_result, "tag_intent") else []

        if progress_callback:
            await progress_callback("planner", {
                "search_terms": parse_result.search_terms,
                "generated_keywords": parse_result.generated_keywords,
                "positive_terms": list(parse_result.tag_intent.positive_terms) if hasattr(parse_result, "tag_intent") and parse_result.tag_intent else [],
                "negative_terms": list(parse_result.tag_intent.negative_terms) if hasattr(parse_result, "tag_intent") and parse_result.tag_intent else []
            })

        # Map fuzzy_neg_terms to real tags via Vector Mapping, since Qdrant filter only works with exact tags.
        mapped_fuzzy_neg = []
        for term in fuzzy_neg_terms:
            try:
                mapped = self.vs.search_tags(
                    f"這部作品的類型偏向{term}",
                    limit=1,
                    similarity_threshold=0.7,
                )
                if mapped:
                    mapped_fuzzy_neg.extend(result["tag"] for result in mapped)
            except Exception as exc:
                print(f"[Engine] Warning: fuzzy_negative tag mapping failed: {exc}")

        combined_negative_terms = self._dedupe_terms(exact_neg_terms + mapped_fuzzy_neg) or self._resolve_negative_tag_terms(parse_result.criteria)
        compiled = compile_query(parse_result, combined_negative_terms)
        hard_constraint_dict = compiled.hard_filters.to_constraint_dict()
        tag_terms_list = compiled.soft_factors.tag_terms_list
        positive_tags = compiled.soft_factors.positive_tags
        print(f"[QueryCompiler] Hard: status={compiled.hard_filters.status_filter}, words=[{compiled.hard_filters.words_min}, {compiled.hard_filters.words_max}], neg_tags={compiled.hard_filters.negative_tag_terms} | Soft: pos_tags={positive_tags}")

        # ── Expand search terms ──
        base_terms = parse_result.search_terms or parse_result.original_query
        expanded_terms = base_terms
        positive_semantic = [c for c in parse_result.criteria if c.name == "semantic_similarity" and not getattr(c, "is_negative", False)]
        sem_texts = []
        norm_base = "".join(str(base_terms).split()).lower()
        for c in positive_semantic:
            qt = self._criteria_params(c).get("query_text", "").strip()
            if qt and "".join(qt.split()).lower() != norm_base:
                sem_texts.append(qt)
        if sem_texts:
            expanded_terms = f"{expanded_terms} {' '.join(sem_texts)}".strip()

        # ── Dynamic Routing ──
        if self.fusion_strategy == "auto":
            routing = self._determine_routing_strategy(parse_result, hard_constraint_dict)
        elif self.fusion_strategy == "auto_llm":
            routing = self._determine_routing_strategy_llm(user_query, parse_result, hard_constraint_dict, model_id=model_id, cache_namespace=cache_namespace)
        else:
            routing = {"fusion": self.fusion_strategy}
        active_fusion = routing.get("fusion", self.fusion_strategy)
        active_ws = routing.get("ws", self.semantic_weight)
        active_wa = routing.get("wa", self.attribute_weight)
        active_rrf_k = routing.get("rrf_k", self.rrf_k)
        active_bm25 = routing.get("enable_bm25", self.enable_bm25)

        # ── Build Metadata Filter (Hard Constraints) ──
        metadata_filter = VectorStore.build_metadata_filter(hard_constraint_dict)

        # ── DAT: Dynamic Alpha Tuning ──
        active_alpha = settings.DAT_DEFAULT_ALPHA
        active_beta = settings.TAG_BONUS_BETA
        dat_info = {"enabled": self.enable_dat, "alpha": None, "beta": None, "source": "static", "latency_ms": 0.0}
        if self.enable_dat and active_fusion == "weighted":
            dat_start = time.perf_counter()
            active_alpha, active_beta, dat_info = self._run_dat_scout(
                user_query, expanded_terms, metadata_filter, tag_terms_list, model_id
            )
            dat_info["enabled"] = True
            print(f"[DAT:Result] α={active_alpha:.3f}, β={active_beta:.3f} (source={dat_info.get('source', 'unknown')})")

        # ── Compute tag mappings (Dual-Track: exact=1.0, fuzzy=batch_map×0.4) ──
        tag_mapping_weights: List[Dict[str, float]] = []
        tag_mapping_info: List[Dict[str, Any]] = []
        if tag_terms_list:
            for tag in tag_terms_list:
                # Dual-track scoring strategy:
                # If the tag is an exact tag from the schema constrained output, it gets weight 1.0 directly.
                # If it's a fuzzy tag (freely generated), run it through batch mapping, and reduce its weight to 0.4.
                is_exact_tag = tag in parse_result.tag_intent.positive_terms if hasattr(parse_result, "tag_intent") else False

                if is_exact_tag:
                    tag_mapping_weights.append({tag: 1.0})
                    tag_mapping_info.append({"term": tag, "is_exact": True, "mappings": [{"tag": tag, "raw_score": 1.0, "scaled_score": 1.0}]})
                else:
                    try:
                        raw_map = self.vs.batch_map_tags([tag], similarity_threshold=0.6)[0]
                    except Exception as exc:
                        print(f"[Engine] Warning: tag batch mapping failed for '{tag}': {exc}")
                        raw_map = {}

                    # Scale down fuzzy mappings for scoring
                    fuzzy_mapped = {k: v * 0.4 for k, v in raw_map.items()}
                    tag_mapping_weights.append(fuzzy_mapped)

                    mappings = []
                    for mt, raw_score in raw_map.items():
                        mappings.append({"tag": mt, "raw_score": raw_score, "scaled_score": raw_score * 0.4})
                    tag_mapping_info.append({"term": tag, "is_exact": False, "mappings": mappings})

        # Build raw tag mapping weights for recall selection (use raw_score cutoff = 0.7)
        raw_tag_mapping_weights: List[Dict[str, float]] = []
        if tag_mapping_info:
            for entry in tag_mapping_info:
                if entry.get("is_exact"):
                    raw_tag_mapping_weights.append({entry.get("term"): 1.0})
                else:
                    mapping_map = {m.get("tag"): m.get("raw_score") for m in entry.get("mappings", [])}
                    raw_tag_mapping_weights.append(mapping_map)

        # Select recall tags using raw mapping scores (>= 0.7)
        recall_tags_override = self._extract_recall_tags(raw_tag_mapping_weights, min_score=0.7) if raw_tag_mapping_weights else []

        # ════════════════════════════════════════════════════════════════
        # ── FAST PATH: Tri-track recall (Vector 500 + BM25 500 + TagVec 300) ──
        # ════════════════════════════════════════════════════════════════
        print(f"[FastPath] Tri-track recall (Vec={self.FAST_PATH_RETRIEVAL_LIMIT}, BM25={self.FAST_PATH_RETRIEVAL_LIMIT}, TagVec={self.TAG_VECTOR_RETRIEVAL_LIMIT}, filter={'ON' if metadata_filter else 'OFF'})...")
        scored_items, query_vector, recall_tags = self._run_retrieval_pipeline(
            expanded_terms, metadata_filter, hard_constraint_dict,
            tag_terms_list, tag_mapping_weights,
            active_bm25, active_fusion, active_ws, active_wa, active_rrf_k,
            active_alpha=active_alpha,
            vector_limit=self.FAST_PATH_RETRIEVAL_LIMIT, bm25_limit=self.FAST_PATH_RETRIEVAL_LIMIT,
            tag_vector_limit=self.TAG_VECTOR_RETRIEVAL_LIMIT, positive_tags=positive_tags,
            active_beta=active_beta,
            recall_tags_override=recall_tags_override,
        )
        if progress_callback:
            await asyncio.sleep(0.8)  # Cinematic pacing: allow UI to render previous stage smoothly
            await progress_callback("retrieval", {
                "candidate_count": len(scored_items),
                "recall_tags": recall_tags if 'recall_tags' in locals() else []
            })
        scored_items = self._post_filter(scored_items, hard_constraint_dict)
        scored_items.sort(key=lambda r: r["score"], reverse=True)

        degradation_level = 0
        system_message = None
        passed_count = len(scored_items)

        # Dynamic threshold: need enough candidates for quality Top-K selection
        exception_threshold = max(limit * self.EXCEPTION_L2_THRESHOLD_MULT, 30)

        if passed_count >= exception_threshold:
            print(f"[FastPath] ✅ {passed_count} results ≥ {exception_threshold}, returning directly.")
        else:
            # ════════════════════════════════════════════════════════════
            # ── EXCEPTION PATH: Relax filters + Triage ──
            # Triggers when Fast Path can't fill the Exception Threshold
            # ════════════════════════════════════════════════════════════
            print(
                f"[ExceptionPath] ⚠️ {passed_count} < {exception_threshold}, "
                f"relaxing hard constraints for rescue retrieval..."
            )

            # Build relaxed constraints: only keep negative tags
            relaxed_constraints = {
                "status_filter": None,
                "author_filter": None,
                "words_min": None,
                "words_max": None,
                "negative_tag_terms": hard_constraint_dict.get("negative_tag_terms", []),
            }
            relaxed_filter = VectorStore.build_metadata_filter(relaxed_constraints)

            l2_scored, _, l2_recall = self._run_retrieval_pipeline(
                expanded_terms, relaxed_filter, relaxed_constraints,
                tag_terms_list, tag_mapping_weights,
                active_bm25, active_fusion, active_ws, active_wa, active_rrf_k,
                active_alpha=active_alpha,
                vector_limit=self.FAST_PATH_RETRIEVAL_LIMIT,
                bm25_limit=self.FAST_PATH_RETRIEVAL_LIMIT,
                tag_vector_limit=self.TAG_VECTOR_RETRIEVAL_LIMIT,
                positive_tags=positive_tags,
                active_beta=active_beta,
                recall_tags_override=recall_tags_override,
            )
            if l2_recall:
                recall_tags = list(set(recall_tags + l2_recall))

            # Apply violation-specific penalties to rescued items
            p_status = settings.PENALTY_REQUIRED_STATUS
            p_tags = settings.PENALTY_REQUIRED_TAGS
            status_filter = hard_constraint_dict.get("status_filter")
            words_min_orig = hard_constraint_dict.get("words_min")
            words_max_orig = hard_constraint_dict.get("words_max")
            author_filter = hard_constraint_dict.get("author_filter")

            for r in l2_scored:
                item = r["item"]
                penalty = 1.0
                violations = []

                # Status violation
                if status_filter:
                    item_status = self._normalize_status(item.get("publish_status", ""))
                    if item_status and item_status != status_filter:
                        penalty *= p_status
                        violations.append(f"status(×{p_status})")

                # Word count violation
                if words_min_orig is not None or words_max_orig is not None:
                    actual_words = item.get("words_total", 0) or 0
                    if actual_words > 0:
                        if (words_min_orig and actual_words < words_min_orig) or \
                           (words_max_orig and actual_words > words_max_orig):
                            penalty *= p_status
                            violations.append(f"words(×{p_status})")

                # Author violation
                if author_filter:
                    author = item.get("author", "")
                    if author and not (author_filter in author or author in author_filter):
                        penalty *= p_status
                        violations.append(f"author(×{p_status})")

                # Required tags missing — with Semantic Immunity
                if positive_tags:
                    book_tags = self._normalize_tags(item.get("tags", []))
                    if not any(any(rt in t or t in rt for t in book_tags) for rt in positive_tags):
                        vec_score = r.get("vector_score", 0.0)
                        tv_score = r.get("tag_vector_score", 0.0)
                        combined_immunity = max(vec_score, tv_score * 0.9)
                        effective_p = p_tags
                        if combined_immunity >= 0.85:
                            effective_p = max(p_tags, 0.95)
                        elif combined_immunity >= 0.80:
                            effective_p = max(p_tags, 0.90)
                        penalty *= effective_p
                        violations.append(f"tags(×{effective_p:.2f}, immunity={combined_immunity:.3f})")

                if penalty < 1.0:
                    r["score"] *= penalty
                    r["breakdown"].append({
                        "criteria": "l2_violation_penalty",
                        "label": "L2: Violation Penalty",
                        "raw_score": penalty,
                        "weighted_score": r["score"],
                        "is_filter": False,
                        "reason": f"Relaxed rescue: score × {penalty:.4f} [{', '.join(violations)}]",
                    })

            # Merge rescued items (only new ones)
            existing_ids = {str(r["item"].get("id")) for r in scored_items}
            l2_new = 0
            for r in l2_scored:
                bid = str(r["item"].get("id"))
                if bid not in existing_ids:
                    scored_items.append(r)
                    existing_ids.add(bid)
                    l2_new += 1

            scored_items.sort(key=lambda r: r["score"], reverse=True)
            degradation_level = 1
            system_message = (
                f"放寬約束條件救援：補充 {l2_new} 本候選 "
                f"(含違規懲罰降權)"
            )
            print(
                f"[ExceptionPath] Rescued {l2_new} new candidates, "
                f"total={len(scored_items)}, degradation_level=1"
            )

        if progress_callback:
            await asyncio.sleep(0.8)  # Cinematic pacing: allow UI to render Rule Filter & Scoring Fusion stage smoothly
            await progress_callback("post_filter", {
                "filtered_count": len(scored_items)
            })

        if not scored_items:
            return {"query": user_query, "parsed_criteria": [self._criteria_to_dict(c) for c in parse_result.criteria], "search_terms": parse_result.search_terms, "generated_keywords": parse_result.generated_keywords, "tag_intent": parse_result.tag_intent.model_dump(), "query_vector": query_vector, "results": [], "message": "No matching novels were found.", "engine": "HybridEngine", "related_books": related_books, "reference_tags": [], "parse_metadata": parse_result.parse_metadata, "degradation_level": degradation_level, "system_message": system_message}

        # ── Optional PermSC Reranking ──
        if self.rerank_enabled:
            candidate_limit = settings.RERANK_CANDIDATE_LIMIT
            # Only rerank the top N candidates to avoid overwhelming the LLM
            to_rerank = scored_items[:candidate_limit]
            print(f"[Engine] Limiting rerank pool from {len(scored_items)} to {len(to_rerank)} candidates.")

            # If a progress_callback is provided (streaming UI usage), avoid
            # performing multiple PermSC permutations to reduce latency/cost.
            # Temporarily set the reranker's permutation count to 1, then
            # restore the original value after reranking.
            reranker_instance = self._get_reranker()
            orig_perms = getattr(reranker_instance, "n_permutations", None)
            if orig_perms is None:
                orig_perms = None
            try:
                if progress_callback:
                    try:
                        reranker_instance.n_permutations = 1
                    except Exception:
                        # If the reranker implementation does not expose this
                        # attribute, silently continue (no behavior change).
                        pass

                reranked = await self._rerank_results(to_rerank, user_query, limit)
            finally:
                if orig_perms is not None:
                    try:
                        reranker_instance.n_permutations = orig_perms
                    except Exception:
                        pass

            # Combine reranked items with the rest of the unranked items
            scored_items = reranked + scored_items[candidate_limit:]

        if progress_callback:
            await progress_callback("rerank", {
                "top_results": [
                    {"name": r["item"].get("name"), "score": r["score"]} 
                    for r in scored_items[:3]
                ]
            })

        # Return all scored items so that the front-end can display the rest behind the top 10
        final_results = scored_items
        for result in final_results:
            result["explanation"] = None

        return {
            "query": user_query,
            "parsed_criteria": [self._criteria_to_dict(c) for c in parse_result.criteria],
            "search_terms": parse_result.search_terms,
            "generated_keywords": parse_result.generated_keywords,
            "tag_intent": {
                "positive_terms": list(parse_result.tag_intent.positive_terms),
                "negative_terms": combined_negative_terms,
                "fuzzy_positive_terms": list(parse_result.tag_intent.fuzzy_positive_terms) if hasattr(parse_result.tag_intent, "fuzzy_positive_terms") else [],
                "fuzzy_negative_terms": list(parse_result.tag_intent.fuzzy_negative_terms) if hasattr(parse_result.tag_intent, "fuzzy_negative_terms") else [],
            },
            "tag_mapping": tag_mapping_info,
            "hypothetical_intro": parse_result.hypothetical_intro,
            "related_books": related_books,
            "reference_tags": recall_tags,
            "parse_metadata": parse_result.parse_metadata,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "HybridEngine",
            "degradation_level": degradation_level,
            "system_message": system_message,
            "dat_info": dat_info
        }
