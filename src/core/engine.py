import asyncio
from typing import List, Dict, Any, Tuple, Optional
from src.core.llm import parse_query
from src.models.schemas import QueryParseResult
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.logic.registry import ScoringRegistry
import src.logic.scoring_functions 
from src.core.explainer import generate_explanation 
from src.config import settings

class HybridEngine:
    """
    深度推理混合引擎 (Hybrid Reasoner Engine)
    包含 Filter、向量、局部取回後的多維度條件逐條加權計分，與可解釋性生成。
    """
    def __init__(self, db=None, vs=None):
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")

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
    ) -> Tuple[float, List[Dict[str, Any]]]:
        total_score = 0.0
        breakdown = []
        
        # --- 1. 處理向量分數 ---
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
    ) -> Dict[str, Any]:
        """
        Executes the full search pipeline.
        """
        # 1. Parse Query
        parse_result = parse_query(user_query, model_id=model_id)
        
        # 2. 準備檢索字詞 (不使用 Qdrant 硬過濾，讓所有書籍進入多維度加權)
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
        
        # 取回全部資料進行純 Python 多維度軟評分
        retrieval_limit = 10000 
        vector_results, query_vector = self.vs.search(
            expanded_terms, 
            limit=retrieval_limit,
            query_filter=None,  # 移除硬過濾
            with_payload=True 
        )
        
        is_relaxed = True  # 標記為放寬，因為我們沒有用任何硬條件
        
        candidates_map = {} 
        vector_score_map = {}
        payload_map = {} 

        for hit in vector_results:
            item = self.db.get_item(hit["id"])
            if item and item.get("name"):
                bid = str(item["id"])
                candidates_map[bid] = item
                vector_score_map[bid] = hit["score"]
                if hit.get('payload'):
                    payload_map[bid] = hit['payload']
        
        # 3. Structural Retrieval (Title & Author Match)
        # 3a. Title Match (Newly Added for Exact Title Search)
        # Utilizes search_terms to find exact book matches
        for term in parse_result.search_terms:
            if len(term) < 2: continue # Skip single chars
            title_matches = self.db.search_by_title_fuzzy(term)
            for book in title_matches:
                b_id = str(book["id"])
                # Only add if robust match (e.g. term is significant part of title)
                # For now, trust the keyword search but assign high score
                if b_id not in candidates_map:
                    print(f"[Engine] 📖 Title Match found: {book['name']} (term: {term})")
                    candidates_map[b_id] = book
                    # Assign max vector score for direct title match
                    vector_score_map[b_id] = 1.0 

        # 3b. Author Match
        for criterion in parse_result.criteria:
            if criterion.name == "author_match":
                # Handle parameter extraction more safely
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

        # 4. Scoring
        # Dynamic normalization within current query candidates (restores discrimination)
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

        scored_items = []
        for item in candidates:
            bid = str(item["id"])
            v_score = vector_score_map.get(bid, 0.0)
            v_norm = vector_norm_map.get(bid, self._normalize_vector_score(float(v_score)))
            
            # Calculate final hybrid score (Rule + Vector)
            score_val, breakdown = self.calculate_score(
                item,
                parse_result.criteria,
                vector_score=v_score,
                normalized_vector_score=v_norm,
            )

            final_score = float(score_val)

            scored_items.append({
                "item": item,
                "score": final_score,
                "vector_score": v_score,
                "breakdown": breakdown,
                "payload": payload_map.get(str(item["id"]), {}) 
            })

        # 最終排序
        scored_items.sort(key=lambda x: float(x["score"]), reverse=True)
        
        # 放寬搜尋的最低語意門檻：過濾掉向量分數太低的雜訊
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
                "engine": "HybridEngine"
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
                    model_id=model_id,
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
            "engine": "HybridEngine",
        }
