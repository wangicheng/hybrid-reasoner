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

class HybridEngine:
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
                    if keyword and isinstance(keyword, str):
                        keyword = keyword.replace(" ", "")
                    
                    # 【核心修改】：如果是找分類或標籤，採取寬鬆策略 (Classification OR Tags)
                    # 意思：要嘛分類對，要嘛標籤對，只要中一個就行
                    if field in ["classification", "tags"] and keyword:
                        should_conditions = [
                            rest.FieldCondition(key="classification.name", match=rest.MatchValue(value=keyword)),
                            rest.FieldCondition(key="tags", match=rest.MatchValue(value=keyword))
                        ]
                        # 用 should (OR) 包裝後放進 must 裡
                        conditions.append(rest.Filter(should=should_conditions))
                    
                    # 其他欄位 (如 author) 還是維持精確匹配
                    elif field and keyword:
                        conditions.append(rest.FieldCondition(key=field, match=rest.MatchValue(value=keyword)))

        if not conditions:
            return None
        return rest.Filter(must=conditions)

    def _normalize_vector_score(self, raw_score: float) -> float:
        """
        將 Qdrant 的 Cosine Score (通常 0.35~0.7) 拉伸到 0.0~1.0
        根據觀察：
        - 0.35 以下：極低相關
        - 0.60 以上：高相關
        """
        min_threshold = 0.35
        max_threshold = 0.65  # 設定一個合理的上限，超過算滿分
        
        if raw_score <= min_threshold:
            return 0.0
        if raw_score >= max_threshold:
            return 1.0
            
        # 線性拉伸
        return (raw_score - min_threshold) / (max_threshold - min_threshold)

    def calculate_score(self, item: Dict[str, Any], criteria_list: List[Any], vector_score: float = 0.0) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculates the total score.
        Fix: 強制加入 Normalized Vector Score 作為基礎分。
        """
        total_score = 0.0
        breakdown = []
        
        # --- 1. 處理向量分數 (強制生效) ---
        # 檢查 LLM 是否有指定 semantic_similarity 的權重
        semantic_criteria = next((c for c in criteria_list if c.name == "semantic_similarity"), None)
        
        if semantic_criteria:
            sem_weight = semantic_criteria.weight
            reason_suffix = "(LLM 指定)"
        else:
            sem_weight = 1.0  # 預設權重，確保它總是佔有一席之地
            reason_suffix = "(系統預設)"

        # 進行正規化拉伸
        normalized_v_score = self._normalize_vector_score(vector_score)
        
        # 計算向量得分
        v_score_contrib = normalized_v_score * sem_weight
        total_score += v_score_contrib
        
        breakdown.append({
            "criteria": "semantic_similarity",
            "weight": sem_weight,
            "raw_score": vector_score,       # 顯示原始分供參考
            "normalized_score": normalized_v_score, # 顯示拉伸後的分數
            "weighted_score": v_score_contrib,
            "reason": f"語意相似度 {vector_score:.3f} -> Norm {normalized_v_score:.2f} {reason_suffix}"
        })

        # --- 2. 處理其他規則分數 ---
        for criteria in criteria_list:
            func_name = criteria.name
            
            # 跳過已經處理過的 semantic_similarity
            if func_name == "semantic_similarity":
                continue
                
            weight = criteria.weight
            
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            else:
                params = criteria.parameters.dict()
            
            # 呼叫規則函數
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
            
            breakdown.append({
                "criteria": func_name,
                "weight": weight,
                "raw_score": raw_score,
                "weighted_score": score_contrib,
                "params": params,
                "reason": reason_msg
            })
            
        return total_score, breakdown

    def search(self, user_query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Executes the full search pipeline.
        """
        # 1. Parse Query
        parse_result = parse_query(user_query)
        
        # 2. Build Qdrant Filter
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        # 3. Retrieval (with Dynamic Query Expansion)
        base_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        
        # 加入 LLM 生成的領域關鍵字
        expanded_terms = base_terms
        if parse_result.generated_keywords:
            # 清理空格 (與既有 keyword 清理邏輯一致)
            cleaned_keywords = [kw.replace(" ", "") for kw in parse_result.generated_keywords]
            expansion_str = " ".join(cleaned_keywords)
            print(f"[Engine] 🤖 LLM 動態擴展關鍵字: {expansion_str}")
            expanded_terms += f" {expansion_str}"
        
        # 擴展 B：HyDE 假簡介 (針對劇情與氛圍匹配)
        if parse_result.hypothetical_intro:
            print(f"[Engine] 🔮 HyDE 假想簡介: {parse_result.hypothetical_intro[:80]}...")
            expanded_terms += f" {parse_result.hypothetical_intro}"
        
        # 要求 Qdrant 回傳 payload，以便後續解釋使用
        vector_results, query_vector = self.vs.search(
            expanded_terms, 
            limit=50,
            query_filter=qdrant_filter,
            with_payload=True 
        )
        
        # 【新增】自動放寬機制 (Auto-Relaxation)
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
        payload_map = {} # 新增：儲存 Qdrant 的 payload (含原文片段)

        # 2a. Process Vector Results
        for hit in vector_results:
            item = self.db.get_item(hit["id"])
            if item:
                bid = str(item["id"])
                candidates_map[bid] = item
                vector_score_map[bid] = hit["score"]
                if hit.get('payload'):
                    payload_map[bid] = hit['payload']
        
        # 2b. Structural Retrieval (Author Match)
        # ... (Author match logic remains same) ...
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
                            # 給予一個基礎向量分數，避免被拉伸邏輯變成 0 分
                            vector_score_map[b_id] = 0.5 

        candidates = list(candidates_map.values())

        # 3. Scoring
        scored_items = []
        for item in candidates:
            v_score = vector_score_map.get(str(item["id"]), 0.0)
            score, breakdown = self.calculate_score(item, parse_result.criteria, vector_score=v_score)

            scored_items.append({
                "item": item,
                "score": score,
                "vector_score": v_score,
                "breakdown": breakdown,
                "payload": payload_map.get(str(item["id"]), {}) # 傳遞 payload
            })
            
        # 4. Rank
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # 【新增】放寬搜尋的最低語意門檻：過濾掉向量分數太低的雜訊
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
                    "message": "資料庫中無相關書籍，請嘗試其他搜尋條件。"
                }
        
        final_results = scored_items[:limit]

        # --- 5. Explainability ---
        top_n_explain = 3 
        for i, res in enumerate(final_results):
            if i < top_n_explain:
                item = res['item']
                breakdown = res['breakdown']
                payload = res.get('payload', {})
                
                # 準備 Context: 優先使用 Qdrant 命中的內容片段 (如果有)
                chunks_to_analyze = []
                if payload.get('content'): # 假設 ingest 時欄位叫 content
                    chunks_to_analyze.append(f"【檢索命中的內文片段】\n{payload['content'][:500]}...")
                elif payload.get('intro'):
                     chunks_to_analyze.append(f"【檢索命中的片段】\n{payload['intro'][:500]}...")
                
                # 補充書籍簡介
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
            "is_relaxed": is_relaxed  # 讓前端 UI 可以顯示「找不到精確結果，為您推薦相關書籍」
        }
