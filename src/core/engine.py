import asyncio
from typing import List, Dict, Any, Tuple, Optional
from src.core.llm import parse_query
from src.models.schemas import QueryParseResult
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.logic.registry import ScoringRegistry
from qdrant_client.http import models as rest
import src.logic.scoring_functions 
from src.core.explainer import generate_explanation 
from src.core.reranker import Reranker
from src.core.llm_reranker import LLMReranker
from src.config import settings

class BaseEngine:
    def __init__(self, db=None, vs=None):
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        self.reranker = Reranker()

    def _build_qdrant_filter(self, criteria_list: List[Any]) -> Optional[rest.Filter]:
        """
        Converts parsed criteria into Qdrant Filter for logic push-down.
        """
        must_conditions = []
        must_not_conditions = []

        for criteria in criteria_list:
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            elif hasattr(criteria.parameters, 'dict'):
                params = criteria.parameters.dict()
            else:
                params = criteria.parameters if isinstance(criteria.parameters, dict) else {}

            name = criteria.name
            is_negative = getattr(criteria, 'is_negative', False)

            current_cond = None

            # 1. Numeric Range
            if name == "numeric_range":
                field = params.get("field")
                min_val = params.get("min_val")
                max_val = params.get("max_val")
                
                if field in ["words_total", "bookmark_count", "rating_score", "total_recommendations"]:
                    range_params = {}
                    if min_val is not None:
                        range_params["gte"] = float(min_val)
                    if max_val is not None:
                        range_params["lte"] = float(max_val)
                    if range_params:
                        current_cond = rest.FieldCondition(key=field, range=rest.Range(**range_params))

            # 2. Status Check
            elif name == "status_check":
                target_status = params.get("target_status")
                if target_status:
                    if target_status.lower() in ["finished", "completed"]:
                        possible_values = ["completed", "已完結", "完結"]
                    elif target_status.lower() in ["ongoing", "serializing"]:
                        possible_values = ["ongoing", "連載中", "連載"]
                    else:
                        possible_values = [target_status]
                    
                    # Create an OR condition for all possible status mappings
                    should_conds = [
                        rest.FieldCondition(key="publish_status", match=rest.MatchValue(value=v))
                        for v in possible_values
                    ]
                    current_cond = rest.Filter(should=should_conds)

            # 3. Keyword Match (Smart Filter: classification OR tags 聯防)
            elif name == "keyword_match":
                field = params.get("field")
                keyword = params.get("keyword")
                # 清理 LLM 可能產生的空格 (e.g. "網 遊" -> "網遊")
                if field in ["classification", "tags"] and keyword and isinstance(keyword, str):
                    keyword = keyword.replace(" ", "")
                
                # 【核心修改】：如果是找分類或標籤，採取寬鬆策略 (Classification OR Tags)
                if field in ["classification", "tags"] and keyword:
                    should_conditions = [
                        rest.FieldCondition(key="classification", match=rest.MatchValue(value=keyword)),
                        rest.FieldCondition(key="tags", match=rest.MatchValue(value=keyword))
                    ]
                    current_cond = rest.Filter(should=should_conditions)
                
                elif field and keyword:
                    current_cond = rest.FieldCondition(key=field, match=rest.MatchValue(value=keyword))

            if current_cond:
                if is_negative:
                    must_not_conditions.append(current_cond)
                else:
                    must_conditions.append(current_cond)

        if not must_conditions and not must_not_conditions:
            return None
            
        filter_kwargs = {}
        if must_conditions: filter_kwargs["must"] = must_conditions
        if must_not_conditions: filter_kwargs["must_not"] = must_not_conditions
        
        return rest.Filter(**filter_kwargs)

    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None, explain: bool = True) -> Dict[str, Any]:
        """
        Base search interface. Derived classes must implement this.
        """
        raise NotImplementedError


class ExactMatchEngine(BaseEngine):
    """
    1. 傳統字面檢索 (Lexical / Exact Match Baseline)
    純粹使用 LLM 解析出的硬性條件，轉化為 Qdrant Filter 進行過濾。
    這代表了最傳統的「標籤勾選+字數過濾」網站體驗。不比較語意，純看是否命中條件。
    """
    def __init__(self, db=None, vs=None):
        super().__init__(db, vs)

    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None, explain: bool = True) -> Dict[str, Any]:
        parse_result = parse_query(user_query, model_id=model_id)
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        try:
            results = []
            if qdrant_filter:
                # 傳統過濾檢索：因為沒有文字向量搜尋(或傳統不依賴)，我們單純使用 scroll 取回符合 filter 的資料
                response, _ = self.vs.client.scroll(
                    collection_name=self.vs.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
                for hit in response:
                    item = self.db.get_item(hit.id)
                    if item:
                        results.append({
                            "item": item,
                            "score": 1.0,  # 基礎命中給予 1.0 (Exact Match)
                            "vector_score": 0.0,
                            "breakdown": [{"criteria": "exact_match", "reason": "符合過濾條件"}],
                            "payload": hit.payload
                        })
            
            return {
                "query": user_query,
                "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
                "query_vector": [], # 無語意向量
                "results": results,
                "is_relaxed": False,
                "engine": "ExactMatchEngine"
            }
        except Exception as e:
            return {
                "query": user_query,
                "error": str(e),
                "engine": "ExactMatchEngine"
            }


class PureVectorEngine(BaseEngine):
    """
    2. 純語意向量搜尋 (Pure Vector Search Baseline)
    不使用 LLM，完全用使用者的原句產生 Embedding，進行 Qdrant Cosine Similarity 搜尋。
    """
    def __init__(self, db=None, vs=None):
        super().__init__(db, vs)

    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None, explain: bool = True) -> Dict[str, Any]:
        """
        Executes the full search pipeline.
        
        Args:
            user_query: The natural language query from user.
            limit: Maximum number of results to return.
            model_id: Optional LLM model ID to use for parsing. If not provided, uses default from env or config.
            explain: Whether to generate an explanation.
        """
        try:
            vector_results, query_vector = self.vs.search(
                user_query,
                limit=limit,
                query_filter=None,
                with_payload=True,
                vector_name="content"
            )
            
            scored_items = []
            for hit in vector_results:
                item = self.db.get_item(hit["id"])
                if item:
                    scored_items.append({
                        "item": item,
                        "score": hit["score"], # 純 Qdrant 的 Cosine Score
                        "vector_score": hit["score"],
                        "breakdown": [{"criteria": "vector_similarity", "reason": f"原始 Cosine Score {hit['score']:.3f}"}],
                        "payload": hit.get("payload", {})
                    })
                    
            return {
                "query": user_query,
                "parsed_criteria": [], # 無 LLM 解析
                "query_vector": query_vector,
                "results": scored_items,
                "is_relaxed": False,
                "engine": "PureVectorEngine"
            }
        except Exception as e:
            return {
                "query": user_query,
                "error": str(e),
                "engine": "PureVectorEngine"
            }


class FilteredVectorEngine(BaseEngine):
    """
    3. 混合式向量過濾搜尋 (Filtered Vector Search)
    使用 LLM 萃取硬性條件 (Filter)，並結合查詢內容的向量與 Qdrant 搜尋。
    取出 Top-K 後，不執行我們自訂的「多維度條件（Criteria）逐條計分」。
    """
    def __init__(self, db=None, vs=None):
        super().__init__(db, vs)

    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None, explain: bool = True) -> Dict[str, Any]:
        parse_result = parse_query(user_query, model_id=model_id)
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        # 組裝擴展關鍵字
        base_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        
        try:
            vector_results, query_vector = self.vs.search(
                base_terms, 
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
                vector_name="content"
            )
            
            scored_items = []
            for hit in vector_results:
                item = self.db.get_item(hit["id"])
                if item:
                    scored_items.append({
                        "item": item,
                        "score": hit["score"], # 混合過濾後的 Qdrant Score
                        "vector_score": hit["score"],
                        "breakdown": [{"criteria": "filtered_vector", "reason": "經過 Hard Filter 後的向量分數"}],
                        "payload": hit.get("payload", {})
                    })
                    
            return {
                "query": user_query,
                "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
                "query_vector": query_vector,
                "results": scored_items,
                "is_relaxed": False,
                "engine": "FilteredVectorEngine"
            }
        except Exception as e:
            return {
                "query": user_query,
                "error": str(e),
                "engine": "FilteredVectorEngine"
            }


class HybridReasonerEngine(BaseEngine):
    """
    4. 深度推理混合引擎 (Hybrid Reasoner Engine) —— 現有的實驗組 (Proposed Method)
    包含 Filter、向量、局部取回後的多維度條件逐條加權計分，與可解釋性生成。
    """
    def __init__(self, db=None, vs=None):
        super().__init__(db, vs)
        self.llm_reranker = LLMReranker()

    @staticmethod
    def _minmax_normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 1e-9:
            return [0.5 for _ in values]
        return [(v - min_v) / (max_v - min_v) for v in values]

    def _normalize_vector_score(self, raw_score: float) -> float:
        """將 Qdrant 的 Cosine Score 拉伸到 0.0~1.0"""
        min_threshold = 0.35
        max_threshold = 0.65  
        
        if raw_score <= min_threshold:
            return 0.0
        if raw_score >= max_threshold:
            return 1.0
            
        return (raw_score - min_threshold) / (max_threshold - min_threshold)

    def calculate_score(
        self,
        item: Dict[str, Any],
        criteria_list: List[Any],
        vector_score: float = 0.0,
        normalized_vector_score: Optional[float] = None,
        tag_vector_score: float = 0.0,
        normalized_tag_vector_score: Optional[float] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        total_score = 0.0
        breakdown = []
        
        # --- 1. 處理內容向量分數 ---
        semantic_criteria = next((c for c in criteria_list if c.name == "semantic_similarity"), None)
        
        is_sem_negative = getattr(semantic_criteria, 'is_negative', False) if semantic_criteria else False
        sem_weight = -1.0 if is_sem_negative else 1.0
        reason_suffix = "(反向權重)" if is_sem_negative else "(固定權重)"

        normalized_v_score = (
            normalized_vector_score
            if normalized_vector_score is not None
            else self._normalize_vector_score(vector_score)
        )
        v_score_contrib = normalized_v_score * sem_weight
        total_score += v_score_contrib
        
        breakdown.append({
            "criteria": "semantic_similarity",
            "label": "語意與內容相似度",
            "weight": sem_weight,
            "raw_score": vector_score,       
            "normalized_score": normalized_v_score, 
            "weighted_score": v_score_contrib,
            "reason": f"語意相似度 {vector_score:.3f} -> Norm {normalized_v_score:.2f} {reason_suffix}"
        })

        # --- 1b. 處理標籤向量分數 (Named Vectors) ---
        normalized_t_score = (
            normalized_tag_vector_score
            if normalized_tag_vector_score is not None
            else self._normalize_vector_score(tag_vector_score)
        )
        t_score_contrib = normalized_t_score * sem_weight  # Same direction as semantic
        total_score += t_score_contrib
        
        breakdown.append({
            "criteria": "tag_vector_similarity",
            "label": "標籤語意相似度",
            "weight": sem_weight,
            "raw_score": tag_vector_score,
            "normalized_score": normalized_t_score,
            "weighted_score": t_score_contrib,
            "reason": f"標籤向量 {tag_vector_score:.3f} -> Norm {normalized_t_score:.2f}"
        })

        # --- 2. 處理其他規則分數 ---
        for criteria in criteria_list:
            func_name = criteria.name
            
            if func_name == "semantic_similarity":
                continue
                
            is_negative = getattr(criteria, 'is_negative', False)
            weight = -1.0 if is_negative else 1.0
            
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            else:
                params = criteria.parameters.dict()
            
            func = ScoringRegistry.get(func_name)
            if not func:
                continue
                
            result = func(item, params)
            
            if isinstance(result, tuple):
                raw_score, reason_msg = result
            else:
                raw_score = float(result)
                reason_msg = f"評分: {raw_score:.2f}"

            score_contrib = raw_score * weight
            total_score += score_contrib
            
            label = func_name
            if func_name == "keyword_match":
                field = params.get("field", "")
                field_str = "分類" if field == "classification" else ("標籤" if field == "tags" else field)
                label = f"{field_str}: {params.get('keyword', '關鍵字')}"
            elif func_name == "numeric_range":
                min_v = params.get("min_val")
                max_v = params.get("max_val")
                if min_v and max_v:
                    label = f"字數: {int(min_v/10000)}萬-{int(max_v/10000)}萬"
                elif min_v:
                    label = f"字數 > {int(min_v/10000)}萬"
                elif max_v:
                    label = f"字數 < {int(max_v/10000)}萬"
                else:
                    label = "字數範圍"
            elif func_name == "status_check":
                label = f"狀態: {params.get('target_status', '狀態')}"
            elif func_name == "author_match":
                label = f"作者: {params.get('author_name', '作者')}"

            if is_negative:
                label = f"[排除] {label}"

            breakdown.append({
                "criteria": func_name,
                "label": label,
                "weight": weight,
                "raw_score": raw_score,
                "weighted_score": score_contrib,
                "params": params,
                "reason": reason_msg
            })
            
        return total_score, breakdown

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
        rerank_strategy: Optional[str] = None,
        rerank_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full search pipeline.
        """
        # 1. Parse Query
        parse_result = parse_query(user_query, model_id=model_id)
        
        # 2. Build Qdrant Filter
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        base_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        
        expanded_terms = base_terms
        if parse_result.generated_keywords:
            cleaned_keywords = [kw.replace(" ", "") for kw in parse_result.generated_keywords]
            expansion_str = " ".join(cleaned_keywords)
            print(f"[Engine] 🤖 LLM 動態擴展關鍵字: {expansion_str}")
            expanded_terms += f" {expansion_str}"
        
        if parse_result.hypothetical_intro:
            print(f"[Engine] 🔮 HyDE 假想簡介: {parse_result.hypothetical_intro[:80]}...")
            expanded_terms += f" {parse_result.hypothetical_intro}"
        
        # --- Content Vector Search ---
        vector_results, query_vector = self.vs.search(
            expanded_terms, 
            limit=50,
            query_filter=qdrant_filter,
            with_payload=True,
            vector_name="content",
            task_type="RETRIEVAL_QUERY"
        )
        
        # --- Tag Vector Search ---
        tag_query = " ".join(parse_result.search_terms) or parse_result.original_query
        if parse_result.generated_keywords:
            tag_query += " " + " ".join(parse_result.generated_keywords[:5])  # Top-5 keywords only
        
        print(f"[Engine] 🏷️ Tag vector query: {tag_query[:60]}...")
        tag_results, _ = self.vs.search(
            tag_query,
            limit=30,
            query_filter=qdrant_filter,
            with_payload=True,
            vector_name="tags",
            task_type="SEMANTIC_SIMILARITY"
        )
        
        is_relaxed = False
        if len(vector_results) < 3 and qdrant_filter is not None:
            print(f"[Engine] ⚠️ 搜尋結果過少 ({len(vector_results)} 筆)，啟動自動放寬機制 (移除 Hard Filter)...")
            vector_results, query_vector = self.vs.search(
                expanded_terms, 
                limit=50,
                query_filter=None,
                with_payload=True,
                vector_name="content",
                task_type="RETRIEVAL_QUERY"
            )
            tag_results, _ = self.vs.search(
                tag_query,
                limit=30,
                query_filter=None,
                with_payload=True,
                vector_name="tags",
                task_type="SEMANTIC_SIMILARITY"
            )
            is_relaxed = True
        
        candidates_map = {} 
        vector_score_map = {}    # Content vector scores
        tag_score_map = {}       # Tag vector scores
        payload_map = {} 

        # Collect content vector results
        for hit in vector_results:
            item = self.db.get_item(hit["id"])
            if item:
                bid = str(item["id"])
                candidates_map[bid] = item
                vector_score_map[bid] = hit["score"]
                if hit.get('payload'):
                    payload_map[bid] = hit['payload']
        
        # Collect tag vector results (merge into candidates)
        for hit in tag_results:
            bid_raw = hit["id"]
            item = self.db.get_item(bid_raw)
            if item:
                bid = str(item["id"])
                if bid not in candidates_map:
                    candidates_map[bid] = item
                    print(f"[Engine] 🏷️ Tag search added: {item.get('name', 'N/A')}")
                tag_score_map[bid] = hit["score"]
                if hit.get('payload') and bid not in payload_map:
                    payload_map[bid] = hit['payload']

        # 3. Structural Retrieval (Title & Author Match)
        for term in parse_result.search_terms:
            if len(term) < 2: continue
            title_matches = self.db.search_by_title_fuzzy(term)
            for book in title_matches:
                b_id = str(book["id"])
                if b_id not in candidates_map:
                    print(f"[Engine] 📖 Title Match found: {book['name']} (term: {term})")
                    candidates_map[b_id] = book
                    vector_score_map[b_id] = 1.0 

        for criterion in parse_result.criteria:
            if criterion.name == "author_match":
                if hasattr(criterion.parameters, 'author_name'):
                    author_name = criterion.parameters.author_name
                elif isinstance(criterion.parameters, dict):
                    author_name = criterion.parameters.get("author_name")
                else: 
                    author_name = None
                
                if author_name:
                    author_books = self.db.search_by_author(author_name)
                    for book in author_books:
                        b_id = str(book["id"])
                        if b_id not in candidates_map:
                            candidates_map[b_id] = book
                            vector_score_map[b_id] = 0.5 

        candidates = list(candidates_map.values())

        # 4. Scoring — normalize both content and tag vector scores
        vector_norm_map: Dict[str, float] = {}
        if vector_score_map:
            vector_values = [float(v) for v in vector_score_map.values()]
            min_vector = min(vector_values)
            max_vector = max(vector_values)

            if max_vector - min_vector > 1e-9:
                for bid, raw_v in vector_score_map.items():
                    vector_norm_map[bid] = (float(raw_v) - min_vector) / (max_vector - min_vector)
            else:
                for bid, raw_v in vector_score_map.items():
                    vector_norm_map[bid] = self._normalize_vector_score(float(raw_v))

        tag_norm_map: Dict[str, float] = {}
        if tag_score_map:
            tag_values = [float(v) for v in tag_score_map.values()]
            min_tag = min(tag_values)
            max_tag = max(tag_values)

            if max_tag - min_tag > 1e-9:
                for bid, raw_t in tag_score_map.items():
                    tag_norm_map[bid] = (float(raw_t) - min_tag) / (max_tag - min_tag)
            else:
                for bid, raw_t in tag_score_map.items():
                    tag_norm_map[bid] = self._normalize_vector_score(float(raw_t))

        scored_items = []
        for item in candidates:
            bid = str(item["id"])
            v_score = vector_score_map.get(bid, 0.0)
            v_norm = vector_norm_map.get(bid, self._normalize_vector_score(float(v_score)))
            t_score = tag_score_map.get(bid, 0.0)
            t_norm = tag_norm_map.get(bid, self._normalize_vector_score(float(t_score)))
            
            score_val, breakdown = self.calculate_score(
                item,
                parse_result.criteria,
                vector_score=v_score,
                normalized_vector_score=v_norm,
                tag_vector_score=t_score,
                normalized_tag_vector_score=t_norm,
            )

            final_score = float(score_val)

            scored_items.append({
                "item": item,
                "score": final_score,
                "vector_score": v_score,
                "tag_vector_score": t_score,
                "breakdown": breakdown,
                "payload": payload_map.get(str(item["id"]), {}) 
            })

        strategy = (rerank_strategy or settings.RERANK_STRATEGY or "score_only").lower().strip()
        allowed_strategies = {"score_only", "rerank_only", "hybrid_fusion", "original_llm_reranker_top10"}
        if strategy not in allowed_strategies:
            strategy = "score_only"

        alpha = rerank_alpha if rerank_alpha is not None else settings.RERANK_FUSION_ALPHA
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.3
        alpha = max(0.0, min(1.0, alpha))
            
        # 5. Rerank (Optional)
        if self.reranker and scored_items and strategy in {"rerank_only", "hybrid_fusion"}:
            rerank_query = user_query
            if parse_result.search_terms:
                rerank_query = " ".join(parse_result.search_terms)

            base_candidates = [entry['item'] for entry in scored_items]
            reranked_items = self.reranker.rerank(rerank_query, base_candidates, top_k=len(base_candidates))

            rerank_map = {}
            for rank_idx, item in enumerate(reranked_items):
                item_id = str(item.get("id"))
                rerank_map[item_id] = {
                    "rerank_score": float(item.get("rerank_score", 0.0)),
                    "rerank_rank": rank_idx + 1,
                }

            for entry in scored_items:
                item_id = str(entry["item"].get("id"))
                rerank_info = rerank_map.get(item_id)
                if rerank_info:
                    entry["rerank_score"] = rerank_info["rerank_score"]
                    entry["rerank_rank"] = rerank_info["rerank_rank"]
                else:
                    entry["rerank_score"] = 0.0
                    entry["rerank_rank"] = len(scored_items) + 1

        # 5b. LLM Rerank Top-10 (Only for explicit mode)
        if self.llm_reranker and scored_items and strategy == "original_llm_reranker_top10":
            rerank_query = user_query
            if parse_result.search_terms:
                rerank_query = " ".join(parse_result.search_terms)

            llm_ranked = self.llm_reranker.rerank(
                query=rerank_query,
                candidates=[entry["item"] for entry in scored_items],
                top_k=10,
            )
            llm_rank_map = {str(r.get("id")): r for r in llm_ranked}
            for entry in scored_items:
                item_id = str(entry["item"].get("id"))
                llm_info = llm_rank_map.get(item_id)
                if llm_info:
                    entry["llm_rerank_score"] = float(llm_info.get("llm_rerank_score", 0.0))
                    entry["llm_rerank_rank"] = int(llm_info.get("llm_rerank_rank", 999))
                else:
                    entry["llm_rerank_score"] = 0.0
                    entry["llm_rerank_rank"] = 999

        if strategy == "original_llm_reranker_top10" and scored_items and any("llm_rerank_score" in e for e in scored_items):
            base_scores = [float(e["score"]) for e in scored_items]
            llm_scores = [float(e.get("llm_rerank_score", 0.0)) for e in scored_items]
            norm_base_scores = self._minmax_normalize(base_scores)
            norm_llm_scores = self._minmax_normalize(llm_scores)

            for idx, entry in enumerate(scored_items):
                entry["normalized_score"] = norm_base_scores[idx]
                entry["normalized_llm_rerank_score"] = norm_llm_scores[idx]
                entry["final_sort_score"] = (1.0 - alpha) * norm_base_scores[idx] + alpha * norm_llm_scores[idx]
        elif scored_items and any("rerank_score" in e for e in scored_items):
            base_scores = [float(e["score"]) for e in scored_items]
            rerank_scores = [float(e.get("rerank_score", 0.0)) for e in scored_items]
            norm_base_scores = self._minmax_normalize(base_scores)
            norm_rerank_scores = self._minmax_normalize(rerank_scores)

            for idx, entry in enumerate(scored_items):
                entry["normalized_score"] = norm_base_scores[idx]
                entry["normalized_rerank_score"] = norm_rerank_scores[idx]

                if strategy == "rerank_only":
                    entry["final_sort_score"] = norm_rerank_scores[idx]
                elif strategy == "hybrid_fusion":
                    entry["final_sort_score"] = (1.0 - alpha) * norm_base_scores[idx] + alpha * norm_rerank_scores[idx]
                else:
                    entry["final_sort_score"] = float(entry["score"])
        else:
            for entry in scored_items:
                entry["final_sort_score"] = float(entry["score"])

        if strategy == "rerank_only" and not any("rerank_score" in e for e in scored_items):
            strategy = "score_only"
        if strategy == "hybrid_fusion" and not any("rerank_score" in e for e in scored_items):
            strategy = "score_only"
        if strategy == "original_llm_reranker_top10" and not any("llm_rerank_score" in e for e in scored_items):
            strategy = "score_only"
        
        # 6. Final Rank
        # Sort by selected strategy score
        scored_items.sort(key=lambda x: float(x.get("final_sort_score", x["score"])), reverse=True)
        
        # 放寬搜尋的最低語意門檻：過濾掉向量分數太低的雜訊
        if scored_items and scored_items[0].get("rerank_score") is not None:
             threshold = 0.01 
        else:
             threshold = 0.6
        
        if is_relaxed:
            scored_items = [r for r in scored_items if float(r['vector_score']) > threshold]
            
        if not scored_items:
            print("[Engine] ℹ️ 無足夠相關結果，回傳空結果。")
            return {
                "query": user_query,
                "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
                "query_vector": query_vector,
                "results": [],
                "is_relaxed": is_relaxed,
                "message": "資料庫中無相關書籍，請嘗試其他搜尋條件。",
                "engine": "HybridReasonerEngine"
            }
        final_results = scored_items[:limit]

        top_n_explain = 3 if explain else 0 
        explainer_runtime_state = {
            "gemini_fail_count": 0,
            "gemini_disabled": False,
            "gemini_fail_threshold": 3,
        }
        for i, res in enumerate(final_results):
            if i < top_n_explain:
                item = res['item']
                breakdown = res['breakdown']
                payload = res.get('payload', {})
                
                chunks_to_analyze = []
                if payload.get('content'): 
                    chunks_to_analyze.append(f"【檢索命中的內文片段】\n{payload['content'][:500]}...")
                elif payload.get('intro'):
                     chunks_to_analyze.append(f"【檢索命中的片段】\n{payload['intro'][:500]}...")
                
                if item.get('intro'):
                    chunks_to_analyze.append(f"【書籍簡介】\n{item['intro']}")

                explanation = generate_explanation(
                    query=user_query,
                    book_item=item,
                    context_chunks=chunks_to_analyze,
                    score_breakdown=breakdown,
                    runtime_state=explainer_runtime_state,
                )
                res['explanation'] = explanation
            else:
                res['explanation'] = None
        
        return {
            "query": user_query,
            "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
            "query_vector": query_vector,
            "results": final_results,
            "is_relaxed": is_relaxed,
            "engine": "HybridReasonerEngine",
            "rerank_strategy": strategy,
            "rerank_alpha": alpha,
            "llm_rerank_top_k": 10 if strategy == "original_llm_reranker_top10" else 0,
        }

# 為了維持對既有程式碼 (`web_api.py`) 的向後相容性，將 HybridEngine 指向 HybridReasonerEngine
HybridEngine = HybridReasonerEngine
