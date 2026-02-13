import asyncio
from typing import List, Dict, Any, Tuple, Optional
from src.core.llm import parse_query
from src.models.schemas import QueryParseResult
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.logic.registry import ScoringRegistry
from qdrant_client.http import models as rest  # For Qdrant Filter
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
            # Get parameters as dict
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            elif hasattr(criteria.parameters, 'dict'):
                params = criteria.parameters.dict()
            else:
                params = criteria.parameters if isinstance(criteria.parameters, dict) else {}

            name = criteria.name

            # 1. Numeric Range (e.g., word count)
            if name == "numeric_range":
                field = params.get("field")
                min_val = params.get("min_val")
                max_val = params.get("max_val")
                
                # Map specific fields to payload keys
                if field == "words_total":
                    range_params = {}
                    if min_val is not None:
                        range_params["gte"] = float(min_val)
                    if max_val is not None:
                        range_params["lte"] = float(max_val)
                    
                    if range_params:
                        conditions.append(
                            rest.FieldCondition(
                                key="words_total", 
                                range=rest.Range(**range_params)
                            )
                        )

            # 2. Status Check
            elif name == "status_check":
                target_status = params.get("target_status")
                if target_status:
                    # Map English status to Chinese DB status
                    status_value = target_status
                    if target_status.lower() in ["finished", "completed"]:
                        status_value = "已完結"
                    elif target_status.lower() in ["ongoing", "serializing"]:
                        status_value = "連載中"

                    conditions.append(
                        rest.FieldCondition(
                            key="publish_status",
                            match=rest.MatchValue(value=status_value)
                        )
                    )

            # 3. Keyword Match (e.g. Classification)
            elif name == "keyword_match":
                # Only apply hard filter if confidence/weight is very high (>= 0.9)
                # This prevents "soft" keywords (adjectives) from becoming strict filters
                if criteria.weight < 0.9:
                    continue

                field = params.get("field")
                keyword = params.get("keyword")
                
                if field == "classification" and keyword:
                    # Use nested path for the classification name
                    conditions.append(
                        rest.FieldCondition(
                            key="classification.name",  # Fixed: use nested path
                            match=rest.MatchValue(value=keyword)
                        )
                    )
                elif field == "tags" and keyword:
                    # Handle tags (list of strings)
                    # Qdrant 'MatchValue' on a list checks for containment
                    conditions.append(
                        rest.FieldCondition(
                            key="tags",
                            match=rest.MatchValue(value=keyword)
                        )
                    )

        if not conditions:
            return None

        # Combine with AND logic (must meet all conditions)
        return rest.Filter(must=conditions)

    def calculate_score(self, item: Dict[str, Any], criteria_list: List[Any], vector_score: float = 0.0) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculates the total score and returns a breakdown of scores per criteria.
        """
        total_score = 0.0
        breakdown = []
        
        for criteria in criteria_list:
            weight = criteria.weight
            func_name = criteria.name
            
            # Convert Pydantic model to dict
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            else:
                params = criteria.parameters.dict()
            
            score_contrib = 0.0
            raw_score = 0.0
            reason_msg = ""
            
            # 1. Handle Vector Score (Special Case)
            if func_name == "semantic_similarity":
                raw_score = vector_score
                score_contrib = vector_score * weight
                total_score += score_contrib
                breakdown.append({
                    "criteria": func_name,
                    "weight": weight,
                    "raw_score": raw_score,
                    "weighted_score": score_contrib,
                    "reason": f"語意相似度 (Score: {raw_score:.3f})"
                })
                continue
                
            # 2. Handle Regular Scoring Functions
            func = ScoringRegistry.get(func_name)
            if not func:
                breakdown.append({
                    "criteria": func_name,
                    "weight": weight,
                    "error": "Function not found"
                })
                continue
                
            # --- 關鍵修改：處理 Tuple 回傳值 ---
            result = func(item, params)
            
            if isinstance(result, tuple):
                raw_score, reason_msg = result
            else:
                raw_score = float(result)
                if raw_score >= 1.0: reason_msg = "符合條件"
                elif raw_score <= 0.0: reason_msg = "未符合條件"
                else: reason_msg = f"評分: {raw_score:.2f}"
            # --------------------------------

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
        Executes the full search pipeline: Parse -> Filter -> Vector Search -> Score -> Rank.
        
        Logic push-down: Hard constraints are pushed to Qdrant for DB-level filtering.
        """
        # 1. Parse Query
        parse_result = parse_query(user_query)
        
        # 2. Build Qdrant Filter (Logic Push-down)
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        # 3. Retrieval (Hybrid Strategy with Filter)
        search_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        
        # Get candidates from vector store with filter applied at DB level
        vector_results, query_vector = self.vs.search(
            search_terms, 
            limit=50,
            query_filter=qdrant_filter  # Logic push-down: filter at DB level
        )
        
        candidates_map = {} # Map ID -> Item
        vector_score_map = {} # Map ID -> Score

        # 2a. Process Vector Results
        for hit in vector_results:
            item = self.db.get_item(hit["id"])
            if item:
                candidates_map[str(item["id"])] = item
                vector_score_map[str(item["id"])] = hit["score"]
        
        # 2b. Structural Retrieval (Author Match)
        for criterion in parse_result.criteria:
            if criterion.name == "author_match":
                # Ensure we handle both dict (if manually constructed) and Pydantic model
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
                            # Assign a base vector score substitute for purely structural matches
                            # so they aren't penalized too heavily in 'semantic_similarity' checks if any
                            vector_score_map[b_id] = 0.5 

        candidates = list(candidates_map.values())

        # 3. Scoring
        scored_items = []
        for item in candidates:
            v_score = vector_score_map.get(str(item["id"]), 0.0)
            score, breakdown = self.calculate_score(item, parse_result.criteria, vector_score=v_score)

            # 不再過濾 score <= 0 的結果，確保純語意搜尋也能回傳候選結果
            scored_type = {
                "item": item,
                "score": score,
                "vector_score": v_score,
                "breakdown": breakdown
            }
            scored_items.append(scored_type)
            
        # 4. Rank
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top N results with metadata
        final_results = scored_items[:limit]

        # --- NEW: Generate Explainability (只針對前 3 名) ---
        # 使用 Gemini 的長 Context Window 特性，可以傳入更多資訊
        top_n_explain = 3 
        
        for i, res in enumerate(final_results):
            if i < top_n_explain:
                item = res['item']
                breakdown = res['breakdown']  # 取得該書籍的評分細節
                
                # 準備 Context Chunks:
                # 由於 Gemini 1.5 Flash 有百萬級 Token Window，
                # 我們可以傳入完整的 intro，甚至未來可加入評論 (reviews) 或章節內容
                chunks_to_analyze = [item.get('intro', '')]
                
                # 如果未來有 'reviews' 或 'chapter_1' 欄位，直接 append 進去
                # if 'reviews' in item: chunks_to_analyze.extend(item['reviews'])
                
                # 呼叫 Explainer (傳入 score_breakdown 作為評分證據)
                explanation = generate_explanation(
                    query=user_query,
                    book_item=item,
                    context_chunks=chunks_to_analyze,
                    score_breakdown=breakdown  # 傳入評分細節給 LLM
                )
                
                # 將解釋寫入結果物件
                res['explanation'] = explanation
            else:
                # 第 4 名以後不生成，給個預設值或留空
                res['explanation'] = None
        # ------------------------------------------------
        
        return {
            "query": user_query,
            "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
            "query_vector": query_vector,
            "results": final_results
        }
