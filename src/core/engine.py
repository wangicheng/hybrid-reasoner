import asyncio
from typing import List, Dict, Any, Tuple, Optional
from src.core.llm import parse_query
# [FIX 1] 補上 Criterion
from src.models.schemas import QueryParseResult, Criterion
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.logic.registry import ScoringRegistry
from qdrant_client.http import models as rest  # For Qdrant Filter
import src.logic.scoring_functions 
from src.core.explainer import generate_explanation 
import os

class HybridEngine:
    def __init__(self):
        # [DEBUG] 檢查 API KEY
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
             print(f"✅ API Key recognized: {api_key[:5]}...")
             
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
                    # [FIX 2] 中英文狀態映射 (Mapping)
                    # AI 常常輸出英文 "finished"，但資料庫通常存 "已完結"
                    # 這裡做一個簡單的轉換，確保資料庫 filter 能抓到東西
                    db_status_value = target_status
                    if target_status.lower() in ["finished", "completed"]:
                        db_status_value = "已完結"
                    elif target_status.lower() in ["ongoing", "serializing"]:
                        db_status_value = "連載中"
                    
                    # print(f"DEBUG: Converting status '{target_status}' -> '{db_status_value}'")

                    conditions.append(
                        rest.FieldCondition(
                            key="publish_status",
                            match=rest.MatchValue(value=db_status_value)
                        )
                    )

            # 3. Keyword Match (e.g. Classification)
            elif name == "keyword_match":
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

        if not conditions:
            return None

        # Combine with AND logic (must meet all conditions)
        return rest.Filter(must=conditions)

# 請找到 calculate_score 這個函式，然後用下面這段覆蓋它：

    def calculate_score(self, item: Dict[str, Any], criteria_list: List[Any], vector_score: float = 0.0) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculates the total score and returns a breakdown of scores per criteria.
        """
        total_score = 0.0
        breakdown = []
        
        for criteria in criteria_list:
            # --- FIX: 終極防禦式寫法，同時支援 Pydantic 物件和 Dict ---
            if isinstance(criteria, dict):
                # 萬一傳進來已經是字典了
                name = criteria.get("name")
                weight = criteria.get("weight", 1.0)
                params = criteria.get("parameters", {})
            else:
                # 正常 Pydantic 物件
                name = criteria.name
                weight = criteria.weight
                # 這裡最關鍵：parameters 本身已經是 Dict，不需要 .dict()，也不需要 model_dump
                params = criteria.parameters 
            # --------------------------------------------------------

            func_name = name # 相容變數名稱
            
            score_contrib = 0.0
            raw_score = 0.0
            
            # Handle special semantic similarity logic
            if func_name == "semantic_similarity":
                # Direct use of retrieval score
                raw_score = vector_score
                score_contrib = vector_score * weight
                total_score += score_contrib
                breakdown.append({
                    "criteria": func_name,
                    "weight": weight,
                    "raw_score": raw_score,
                    "weighted_score": score_contrib,
                    "reason": "Vector similarity score"
                })
                continue
                
            func = ScoringRegistry.get(func_name)
            if not func:
                # print(f"Warning: Function '{func_name}' not found.")
                breakdown.append({
                    "criteria": func_name,
                    "weight": weight,
                    "error": "Function not found"
                })
                continue
                
            raw_score = func(item, params)
            score_contrib = raw_score * weight
            total_score += score_contrib
            
            breakdown.append({
                "criteria": func_name,
                "weight": weight,
                "raw_score": raw_score,
                "weighted_score": score_contrib,
                "params": params
            })
            
        return total_score, breakdown

    def search(self, user_query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Executes the full search pipeline: Parse -> Filter -> Vector Search -> Score -> Rank.
        """
        # 1. Parse Query
        parse_result = parse_query(user_query)
        
        # [FIX 3] 保底機制 (Default Semantic Score)
        # 如果解析結果中沒有包含 "semantic_similarity" (語意相似度) 的規則，
        # 我們手動加回去，確保至少會有向量搜尋的基礎分數，不會因為總分 0 而被過濾掉。
        has_semantic = any(c.name == "semantic_similarity" for c in parse_result.criteria)
        if not has_semantic:
            # print("DEBUG: Adding default semantic_similarity rule.")
            default_semantic = Criterion(
                name="semantic_similarity",
                weight=1.0,
                parameters={}
            )
            parse_result.criteria.append(default_semantic)

        # 2. Build Qdrant Filter (Logic Push-down)
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        # 3. Retrieval (Hybrid Strategy with Filter)
        search_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        
        # Get candidates from vector store with filter applied at DB level
        vector_results = self.vs.search(
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
                            vector_score_map[b_id] = 0.5 

        candidates = list(candidates_map.values())

        # 3. Scoring
        scored_items = []
        for item in candidates:
            v_score = vector_score_map.get(str(item["id"]), 0.0)
            score, breakdown = self.calculate_score(item, parse_result.criteria, vector_score=v_score)
            
            if score > 0:
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
        top_n_explain = 3 
        
        for i, res in enumerate(final_results):
            if i < top_n_explain:
                item = res['item']
                chunks_to_analyze = [item.get('intro', '')]
                
                explanation = generate_explanation(
                    query=user_query,
                    book_item=item,
                    context_chunks=chunks_to_analyze
                )
                
                res['explanation'] = explanation
            else:
                res['explanation'] = None
        # ------------------------------------------------
        
        return {
            "query": user_query,
            "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
            "results": final_results
        }