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

class BaseEngine:
    def __init__(self):
        self.db = Database()
        self.vs = VectorStore(collection_name="novels")

    def _build_qdrant_filter(self, criteria_list: List[Any]) -> Optional[rest.Filter]:
        """
        Converts parsed criteria into Qdrant Filter for logic push-down.
        """
        conditions = []

        for criteria in criteria_list:
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            elif hasattr(criteria.parameters, 'dict'):
                params = criteria.parameters.dict()
            else:
                params = criteria.parameters if isinstance(criteria.parameters, dict) else {}

            name = criteria.name
            weight = criteria.weight

            # 1. Numeric Range
            if name == "numeric_range":
                field = params.get("field")
                min_val = params.get("min_val")
                max_val = params.get("max_val")
                
                if field == "words_total":
                    range_params = {}
                    if min_val is not None:
                        range_params["gte"] = float(min_val)
                    if max_val is not None:
                        range_params["lte"] = float(max_val)
                    if range_params:
                        conditions.append(rest.FieldCondition(key="words_total", range=rest.Range(**range_params)))

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
                    conditions.append(rest.Filter(should=should_conds))

            # 3. Keyword Match (Smart Filter: classification OR tags 聯防)
            elif name == "keyword_match":
                if weight >= 0.8:
                    field = params.get("field")
                    keyword = params.get("keyword")
                    # 清理 LLM 可能產生的空格 (e.g. "網 遊" -> "網遊")
                    if field in ["classification", "tags"] and keyword and isinstance(keyword, str):
                        keyword = keyword.replace(" ", "")
                    
                    # 【核心修改】：如果是找分類或標籤，採取寬鬆策略 (Classification OR Tags)
                    if field in ["classification", "tags"] and keyword:
                        should_conditions = [
                            rest.FieldCondition(key="classification.name", match=rest.MatchValue(value=keyword)),
                            rest.FieldCondition(key="tags", match=rest.MatchValue(value=keyword))
                        ]
                        conditions.append(rest.Filter(should=should_conditions))
                    
                    elif field and keyword:
                        conditions.append(rest.FieldCondition(key=field, match=rest.MatchValue(value=keyword)))

        if not conditions:
            return None
        return rest.Filter(must=conditions)

    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None) -> Dict[str, Any]:
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
    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None) -> Dict[str, Any]:
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
    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            vector_results, query_vector = self.vs.search(
                user_query, # 完全使用使用者的原始字串
                limit=limit,
                query_filter=None, # 無任何硬性限制
                with_payload=True 
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
    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None) -> Dict[str, Any]:
        parse_result = parse_query(user_query, model_id=model_id)
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        # 組裝擴展關鍵字
        base_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        
        try:
            vector_results, query_vector = self.vs.search(
                base_terms, 
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True 
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
    def _normalize_vector_score(self, raw_score: float) -> float:
        """將 Qdrant 的 Cosine Score 拉伸到 0.0~1.0"""
        min_threshold = 0.35
        max_threshold = 0.65  
        
        if raw_score <= min_threshold:
            return 0.0
        if raw_score >= max_threshold:
            return 1.0
            
        return (raw_score - min_threshold) / (max_threshold - min_threshold)

    def calculate_score(self, item: Dict[str, Any], criteria_list: List[Any], vector_score: float = 0.0) -> Tuple[float, List[Dict[str, Any]]]:
        total_score = 0.0
        breakdown = []
        
        # --- 1. 處理向量分數 ---
        semantic_criteria = next((c for c in criteria_list if c.name == "semantic_similarity"), None)
        
        if semantic_criteria:
            sem_weight = semantic_criteria.weight
            reason_suffix = "(LLM 指定)"
        else:
            sem_weight = 1.0  
            reason_suffix = "(系統預設)"

        normalized_v_score = self._normalize_vector_score(vector_score)
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

        # --- 2. 處理其他規則分數 ---
        for criteria in criteria_list:
            func_name = criteria.name
            
            if func_name == "semantic_similarity":
                continue
                
            weight = criteria.weight
            
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

    def search(self, user_query: str, limit: int = 5, model_id: Optional[str] = None) -> Dict[str, Any]:
        parse_result = parse_query(user_query, model_id=model_id)
        
        try:
            return self._execute_search(user_query, parse_result, limit)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "query": user_query,
                "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
                "query_vector": [],
                "results": [],
                "is_relaxed": False,
                "error": str(e),
                "engine": "HybridReasonerEngine"
            }

    def _execute_search(self, user_query: str, parse_result: Any, limit: int) -> Dict[str, Any]:
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
        
        vector_results, query_vector = self.vs.search(
            expanded_terms, 
            limit=50,
            query_filter=qdrant_filter,
            with_payload=True 
        )
        
        is_relaxed = False
        if len(vector_results) < 3 and qdrant_filter is not None:
            print(f"[Engine] ⚠️ 搜尋結果過少 ({len(vector_results)} 筆)，啟動自動放寬機制 (移除 Hard Filter)...")
            vector_results, query_vector = self.vs.search(
                expanded_terms, 
                limit=50,
                query_filter=None,
                with_payload=True 
            )
            is_relaxed = True
        
        candidates_map = {} 
        vector_score_map = {}
        payload_map = {} 

        for hit in vector_results:
            item = self.db.get_item(hit["id"])
            if item:
                bid = str(item["id"])
                candidates_map[bid] = item
                vector_score_map[bid] = hit["score"]
                if hit.get('payload'):
                    payload_map[bid] = hit['payload']
        
        for criterion in parse_result.criteria:
            if criterion.name == "author_match":
                if hasattr(criterion.parameters, 'author_name'):
                    author_name = criterion.parameters.author_name
                else:
                    author_name = criterion.parameters.get("author_name")
                
                if author_name:
                    author_books = self.db.search_by_author(author_name)
                    for book in author_books:
                        b_id = str(book["id"])
                        if b_id not in candidates_map:
                            candidates_map[b_id] = book
                            vector_score_map[b_id] = 0.5 

        candidates = list(candidates_map.values())

        scored_items = []
        for item in candidates:
            v_score = vector_score_map.get(str(item["id"]), 0.0)
            score, breakdown = self.calculate_score(item, parse_result.criteria, vector_score=v_score)

            scored_items.append({
                "item": item,
                "score": score,
                "vector_score": v_score,
                "breakdown": breakdown,
                "payload": payload_map.get(str(item["id"]), {}) 
            })
            
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        if is_relaxed:
            scored_items = [r for r in scored_items if r['vector_score'] > 0.6]
            if not scored_items:
                print("[Engine] ℹ️ 放寬搜尋後仍無足夠相關結果，回傳空結果。")
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

        top_n_explain = 3 
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
                    score_breakdown=breakdown
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
            "engine": "HybridReasonerEngine"
        }

# 為了維持對既有程式碼 (`web_api.py`) 的向後相容性，將 HybridEngine 指向 HybridReasonerEngine
HybridEngine = HybridReasonerEngine
