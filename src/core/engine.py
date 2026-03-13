import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.core.llm import parse_query
from src.models.schemas import QueryParseResult
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.core.book_matcher import BookMatcher
from src.logic.registry import ScoringRegistry
import src.logic.scoring_functions 
from src.core.explainer import generate_explanation 
from src.config import settings
from qdrant_client.http import models as rest

class HybridEngine:
    """
    简化的混合搜索引擎 (Simplified Hybrid Search Engine)
    
    核心策略：
    - 主要依赖：多向量语义搜索 (text_semantic 0.7 + tag_semantic 0.3)
    - 可选过滤：状态、作者、字数（Qdrant 硬过滤）
    - 负向语义：二次向量查询实现排除功能
    """
    def __init__(self, db=None, vs=None, retrieval_mode: str = "multi_multiplicative"):
        self.db = db if db is not None else Database()
        self.retrieval_mode = retrieval_mode
        # [USET-REQUEST] Ensure ALL fusion strategies use Multiplicative
        self.fusion_mode = "multiplicative"
        
        # Exp 5: multi_multiplicative (Joined tag matching) - The ONLY mode using multi-vector now
        if (retrieval_mode.startswith("multi_multiplicative") or retrieval_mode.startswith("multi_additive")) and "embedded_tags" not in retrieval_mode:
            collection_name = "novels"
            self.use_multi_vector = True
            print(f"[HybridEngine] Exp 5: Using multi-vector (Joined Tag Matching) for: {retrieval_mode}")
        else:
            collection_name = "novels_fused" if retrieval_mode.startswith("fused") else "novels"
            self.use_multi_vector = False
            if "embedded_tags" in retrieval_mode:
                print(f"[HybridEngine] Exp 3: Using single-vector + embedded_tags for: {retrieval_mode}")
            elif retrieval_mode.startswith("fused"):
                print(f"[HybridEngine] Using single-vector fused embeddings for: {retrieval_mode}")
            else:
                print(f"[HybridEngine] Using baseline embeddings for: {retrieval_mode}")
    
        self.vs = vs if vs is not None else VectorStore(collection_name=collection_name)
        self.book_matcher = BookMatcher(self.db)
        
        # Method 2 Cache: Pre-load tags if using baseline_prompt mode
        self.all_tags_cache = None
        if self.retrieval_mode.startswith("baseline_prompt"):
            self._load_tags_cache()

    def _load_tags_cache(self):
        """Pre-load all tags from JSON for Method 2 to avoid frequent I/O."""
        import json
        import os
        tags_path = "data/all_tags.json"
        if os.path.exists(tags_path):
            try:
                with open(tags_path, "r", encoding="utf-8") as f:
                    self.all_tags_cache = json.load(f)
                print(f"[HybridEngine] Method 2: Pre-loaded {len(self.all_tags_cache)} tags for cache")
            except Exception as e:
                print(f"[HybridEngine] Warning: Failed to load {tags_path}: {e}")
        else:
             print(f"[HybridEngine] Warning: {tags_path} not found for Method 2")


    def _build_qdrant_filter(self, criteria_list: List[Any]) -> Optional[rest.Filter]:
        """
        根据 criteria 构建 Qdrant 硬过滤器（Logic Push-down）
        
        硬過濾器的意義：
        在 Qdrant 向量搜索「之前」先排除不可能符合的候選項，
        這樣向量搜索只在符合條件的子集上進行，大幅提升效率和精確度。
        
        支持的过滤条件：
        - status_check: publish_status = "完結" or "連載"
        - author_match: author 包含指定作者名
        - numeric_range: words_total 的范围查询
        
        Returns:
            rest.Filter 对象，如果没有过滤条件则返回 None
        """
        conditions = []
        
        for criteria in criteria_list:
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            else:
                params = criteria.parameters.dict()
            
            # 1. 状态检查
            if criteria.name == "status_check":
                target_status = params.get("target_status", "")
                target_lower = target_status.lower()
                
                # 映射多种表达方式到数据库实际值（涵蓋繁/簡體中文與英文）
                completed_keywords = [
                    "complet", "finish", "ended", "done",
                    "完結", "完结", "已完結", "已完结"
                ]
                ongoing_keywords = [
                    "ongoing", "serializ", "running", "active",
                    "連載", "连载", "連載中", "连载中"
                ]
                
                if any(x in target_lower or x in target_status for x in completed_keywords):
                    status_value = "完結"
                elif any(x in target_lower or x in target_status for x in ongoing_keywords):
                    status_value = "連載"
                else:
                    print(f"[Filter] 無法辨識狀態值: '{target_status}'，跳過")
                    continue
                
                conditions.append(
                    rest.FieldCondition(
                        key="publish_status",
                        match=rest.MatchValue(value=status_value)
                    )
                )
                print(f"[Filter] 狀態過濾: {status_value}")
            
            # 2. 作者匹配
            elif criteria.name == "author_match":
                author_name = params.get("author_name", "").strip()
                if author_name:
                    # 優先使用 MatchText（支持部分匹配，需要 full-text index）
                    # 若 Qdrant 版本不支持或欄位無 index 則回退到 MatchValue
                    try:
                        conditions.append(
                            rest.FieldCondition(
                                key="author",
                                match=rest.MatchText(text=author_name)
                            )
                        )
                    except Exception:
                        conditions.append(
                            rest.FieldCondition(
                                key="author",
                                match=rest.MatchValue(value=author_name)
                            )
                        )
                    print(f"[Filter] 作者過濾: {author_name}")
            
            # 3. 字数范围（仅支持 words_total 字段）
            elif criteria.name == "numeric_range":
                field = params.get("field")
                if field == "words_total":
                    min_val = params.get("min_val")
                    max_val = params.get("max_val")
                    
                    def _fmt_words(val):
                        """安全格式化字數為萬字單位"""
                        if val is None or val == 0:
                            return "0"
                        return f"{int(val / 10000)}"
                    
                    if min_val is not None and max_val is not None:
                        conditions.append(
                            rest.FieldCondition(
                                key="words_total",
                                range=rest.Range(gte=min_val, lte=max_val)
                            )
                        )
                        print(f"[Filter] 字數範圍: {_fmt_words(min_val)}-{_fmt_words(max_val)}萬字")
                    elif min_val is not None:
                        conditions.append(
                            rest.FieldCondition(
                                key="words_total",
                                range=rest.Range(gte=min_val)
                            )
                        )
                        print(f"[Filter] 字數 >= {_fmt_words(min_val)}萬字")
                    elif max_val is not None:
                        conditions.append(
                            rest.FieldCondition(
                                key="words_total",
                                range=rest.Range(lte=max_val)
                            )
                        )
                        print(f"[Filter] 字數 <= {_fmt_words(max_val)}萬字")
        
        if conditions:
            return rest.Filter(must=conditions)
        return None

    def calculate_score(
        self,
        item: Dict[str, Any],
        criteria_list: List[Any],
        vector_score: float = 0.0,
        tag_terms_list: Optional[List[str]] = None,
        tag_weights: Optional[Dict[str, float]] = None,
        negative_tag_terms: Optional[List[str]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        简化评分逻辑：只计算语义分数（纯分数，不归一化）
        
        Args:
            item: 候选书籍项
            criteria_list: 评分条件列表
            vector_score: 原始向量分数
        
        Returns:
            (总分, 评分明细)
        """
        breakdown = []
        
        # --- 1. 正向语义分数（纯分数）---
        total_score = vector_score
        
        breakdown.append({
            "criteria": "semantic_similarity",
            "label": "語意相似度 (文本×標籤)",
            "raw_score": vector_score,
            "weighted_score": vector_score,
            "is_filter": False,
            "reason": f"多向量融合分數: {vector_score:.4f}"
        })
        

        # --- 2.5 標籤硬性匹配分數 (Limit to Baseline & Method 3) ---
        is_baseline = self.retrieval_mode.startswith("baseline")
        is_method3 = "embedded_tags" in self.retrieval_mode
        
        if (is_baseline or is_method3) and tag_terms_list:
            match_count = 0
            bonus_sum = 0.0
            matched_tags = []
            book_tags = item.get("tags", [])
            if isinstance(book_tags, str):
                import json
                try: book_tags = json.loads(book_tags)
                except: book_tags = []
                
            for target in tag_terms_list:
                for b_tag in book_tags:
                    if target in b_tag or b_tag in target:
                        match_count += 1
                        weight = tag_weights.get(target, 1.0) if tag_weights else 1.0
                        bonus_sum += 0.1 * weight
                        matched_tags.append(f"{b_tag}({weight:.2f})")
                        break
            
            if match_count > 0:
                bonus = bonus_sum
                method_label = "Embedded Tags" if is_method3 else "Baseline"
                if self.fusion_mode == "multiplicative":
                    total_score *= (1.0 + bonus)
                    reason_str = f"關聯 {match_count} 個標籤 (依相似度乘法倍率: {1.0+bonus:.2f}x): {', '.join(matched_tags)}"
                else:
                    total_score += bonus
                    reason_str = f"關聯 {match_count} 個標籤 (依相似度線性加分: +{bonus:.2f}): {', '.join(matched_tags)}"
                
                breakdown.append({
                    "criteria": "keyword_match",
                    "label": f"標籤匹配加成 ({method_label} - {self.fusion_mode})",
                    "raw_score": match_count,
                    "weighted_score": bonus,
                    "is_filter": False,
                    "reason": reason_str
                })

        # --- 2.6 負面標籤硬性排除 (Threshold 0.85) ---
        if (is_baseline or is_method3 or "multi_" in self.retrieval_mode) and negative_tag_terms:
            book_tags = item.get("tags", [])
            if isinstance(book_tags, str):
                try: book_tags = json.loads(book_tags)
                except: book_tags = []
            
            hit_neg_tags = []
            for neg_t in negative_tag_terms:
                for b_t in book_tags:
                    if neg_t in b_t or b_t in neg_t:
                        hit_neg_tags.append(b_t)
            
            if hit_neg_tags:
                # [USER-SET] Hard exclusion with 0.85 threshold mapping result
                return 0.0, [{
                    "criteria": "negative_tag_exclusion",
                    "label": "[排除] 負面標籤命中",
                    "raw_score": 0.0,
                    "weighted_score": 0.0,
                    "is_filter": True,
                    "reason": f"書中包含排除標籤: {', '.join(hit_neg_tags)}"
                }]

        # --- 3. 过滤条件（仅记录，不计分）---
        for criteria in criteria_list:
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            else:
                params = criteria.parameters.dict()
            
            # 状态检查
            if criteria.name == "status_check":
                target_status = params.get("target_status", "").lower()
                status_label = "完結" if any(x in target_status for x in ["complet", "finish", "完結"]) else "連載"
                breakdown.append({
                    "criteria": "status_check",
                    "label": f"[過濾] 狀態: {status_label}",
                    "matched": True,
                    "is_filter": True,
                    "reason": "已在檢索層過濾（Qdrant Filter）"
                })
            
            # 作者匹配
            elif criteria.name == "author_match":
                author_name = params.get("author_name", "")
                breakdown.append({
                    "criteria": "author_match",
                    "label": f"[過濾] 作者: {author_name}",
                    "matched": True,
                    "is_filter": True,
                    "reason": "已在檢索層過濾（Qdrant Filter）"
                })
            
            # 字数范围
            elif criteria.name == "numeric_range" and params.get("field") == "words_total":
                min_v = params.get("min_val")
                max_v = params.get("max_val")
                if min_v and max_v:
                    label = f"[過濾] 字數: {int(min_v/10000)}-{int(max_v/10000)}萬字"
                elif min_v:
                    label = f"[過濾] 字數 >= {int(min_v/10000)}萬字"
                elif max_v:
                    label = f"[過濾] 字數 <= {int(max_v/10000)}萬字"
                else:
                    label = "[過濾] 字數範圍"
                
                breakdown.append({
                    "criteria": "numeric_range",
                    "label": label,
                    "matched": True,
                    "is_filter": True,
                    "reason": "已在檢索層過濾（Qdrant Filter）"
                })
            
        return total_score, breakdown

    def _extract_reference_novel_tags(
        self,
        user_query: str,
        search_terms: str = "",
        reference_books: Optional[List[str]] = None,
    ) -> List[str]:
        """委派給 BookMatcher 進行三層書名比對。"""
        return self.book_matcher.extract_reference_tags(
            user_query,
            search_terms=search_terms,
            reference_books=reference_books,
        )

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        简化的搜索流程：语义搜索 + 硬过滤 + 负向语义
        """
        # 1. Parse Query
        tag_list = self.all_tags_cache if self.retrieval_mode.startswith("baseline_prompt") else None
        if tag_list:
            print(f"[Engine] Method 2: Using cached {len(tag_list)} tags for LLM reference")
        
        parse_result = parse_query(user_query, model_id=model_id, tag_list=tag_list)
        
        # 1.5 提取参考小说标签（用 search_terms 做模糊查詢）
        reference_tags = self._extract_reference_novel_tags(
            user_query, 
            search_terms=parse_result.search_terms,
            reference_books=parse_result.reference_books,
        )
        
        # 2. 构建 Qdrant 硬过滤器
        qdrant_filter = self._build_qdrant_filter(parse_result.criteria)
        
        # 3. 準備檢索字詞（擴展查詢 + 正向語義條件 + 參考標籤）
        base_terms = parse_result.search_terms or parse_result.original_query
        
        expanded_terms = base_terms
        
        # 3.1 將正向 semantic_similarity 的 query_text 加入搜索查詢
        #     這是關鍵：LLM 解析出的標籤/概念語義必須參與向量搜索
        positive_semantic = [c for c in parse_result.criteria 
                            if c.name == "semantic_similarity" and not getattr(c, 'is_negative', False)]
        if positive_semantic:
            semantic_texts = []
            for sc in positive_semantic:
                params = sc.parameters.model_dump() if hasattr(sc.parameters, 'model_dump') else sc.parameters.dict()
                qt = params.get("query_text", "").strip()
                if qt:
                    semantic_texts.append(qt)
            if semantic_texts:
                semantic_expansion = " ".join(semantic_texts)
                print(f"[Engine] 正向語義條件加入搜索: {semantic_expansion}")
                expanded_terms += f" {semantic_expansion}"
        
        # 3.2 LLM 生成的擴展關鍵字
        if parse_result.generated_keywords:
            cleaned_keywords = [kw.replace(" ", "") for kw in parse_result.generated_keywords]
            expansion_str = " ".join(cleaned_keywords)
            print(f"[Engine] LLM-expanded keywords: {expansion_str}")
            expanded_terms += f" {expansion_str}"
        
        # 3.3 添加参考小说的标签到查询中
        if reference_tags:
            tags_str = " ".join(reference_tags[:8])  # 最多使用8个标签
            print(f"[Engine] 添加參考標籤到查詢: {tags_str}")
            expanded_terms += f" {tags_str}"
        
        # 3.4 HyDE 假設文檔嵌入 (Disabled manually)
        # if parse_result.hypothetical_intro:
        #     print(f"[Engine] HyDE hypothetical intro: {parse_result.hypothetical_intro[:80]}...")
        #     expanded_terms += f" {parse_result.hypothetical_intro}"
        
        tag_terms_list = []
        tag_weights = {}
        if parse_result.generated_keywords:
            tag_terms_list.extend([kw.replace(" ", "") for kw in parse_result.generated_keywords])
        if reference_tags:
            tag_terms_list.extend([t.replace(" ", "") for t in reference_tags[:8]])
        tag_query_text = " ".join(tag_terms_list)
        print(f"[Engine] Pure tags query for tag_semantic: '{tag_query_text}'")

        # Method 3: Maps LLM tags to system tags individually via embedding similarity
        if "embedded_tags" in self.retrieval_mode and tag_terms_list:
            print(f"[Engine] Method 3: Mapping {len(tag_terms_list)} tags individually with Thres 0.85")
            original_tags = list(tag_terms_list)
            tag_terms_list = []
            
            for t in original_tags:
                try:
                    # [USER-SET] Consistent 0.85 threshold for tag mapping
                    # Increase limit to capture all relevant tags
                    similar = self.vs.search_tags(t, limit=20, similarity_threshold=0.85)
                    for res in similar:
                        s_tag = res["tag"]
                        s_score = res["score"]
                        # Store best score for calculating bonus
                        if s_tag not in tag_weights or s_score > tag_weights[s_tag]:
                            tag_weights[s_tag] = s_score
                        if s_tag not in tag_terms_list:
                            tag_terms_list.append(s_tag)
                except Exception as e:
                    print(f"[Engine] Method 3 mapping failed for '{t}': {e}")
            
            print(f"[Engine] Method 3: Final mapped tags -> {tag_terms_list}")
        
        # --- Negative Tag Terms Handling ---
        negative_tag_terms = []
        negative_criteria_list = [c for c in parse_result.criteria if c.name == "semantic_similarity" and getattr(c, 'is_negative', False)]
        
        for nc in negative_criteria_list:
            p = nc.parameters.model_dump() if hasattr(nc.parameters, 'model_dump') else nc.parameters.dict()
            qt = p.get("query_text", "").strip()
            if not qt: continue
            
            # For Method 3 / Multi-Vector: Try to map negative keyword to system tags
            if "embedded_tags" in self.retrieval_mode or "multi_" in self.retrieval_mode:
                try:
                    neg_mapped = self.vs.search_tags(qt, limit=5, similarity_threshold=0.85)
                    negative_tag_terms.extend([res["tag"] for res in neg_mapped])
                    if neg_mapped:
                        print(f"[Engine] Negative mapped tags (Thres 0.85) for '{qt}': {[res['tag'] for res in neg_mapped]}")
                except Exception as e:
                    print(f"[Engine] Warning: Negative tag mapping failed: {e}")
            else:
                # Baseline: exact keyword matching
                negative_tag_terms.append(qt)
                
        # Method 4 (Feature Fusion): Format query into structured components
        if self.retrieval_mode.startswith("fused"):
            # Structure: [TITLE] ... [/TITLE] [TAGS] ... [/TAGS] [ABSTRACT] ... [/ABSTRACT]
            pseudo_title = " ".join(parse_result.reference_books) if parse_result.reference_books else ""
            pseudo_tags = tag_query_text
            pseudo_abstract = base_terms
            if "semantic_expansion" in locals() and "semantic_expansion" in dir():
               pseudo_abstract += f" {semantic_expansion}"
            
            expanded_terms = f"[TITLE] {pseudo_title} [/TITLE] [TAGS] {pseudo_tags} [/TAGS] [ABSTRACT] {pseudo_abstract} [/ABSTRACT]"
            print(f"[Engine] Method 4: Fused query format applied -> {expanded_terms}")
        
        
        # 4. 执行正向语义搜索（带硬过滤）
        retrieval_limit = 100  # 减少检索数量，因为有硬过滤
        
        # Use multi-vector search if available
        query_vector = None
        if self.use_multi_vector:
            # Pass tag_terms_list for Exp 3 (Individual matching) 
            # or tag_query_text for Exp 5 (Joined matching)
            tag_query_list = tag_terms_list if "embedded_tags" in self.retrieval_mode else None
            
            vector_results, query_vector = self.vs.search_multi_vector(
                expanded_terms,
                limit=retrieval_limit,
                query_filter=qdrant_filter,  # 应用硬过滤
                with_payload=True,
                text_weight=0.7,  # Text semantic priority
                tag_weight=0.3,   # Tag semantic secondary
                fusion_mode=self.fusion_mode,
                tag_query_text=tag_query_text,
                tag_query_list=tag_query_list
            )
            print(f"[Engine] Multi-vector search: text_weight=0.7, tag_weight=0.3, fusion_mode={self.fusion_mode}")
        else:
            vector_results, query_vector = self.vs.search(
                expanded_terms,
                limit=retrieval_limit,
                query_filter=qdrant_filter,  # 应用硬过滤
                with_payload=True
            )
        
        # 5. 收集候选项（並補充完整資料）
        candidates_map = {} 
        vector_score_map = {}
        payload_map = {} 

        for hit in vector_results:
            # Use payload data directly from Qdrant
            if hit.get('payload') and hit['payload'].get("name"):
                str_id = hit['payload'].get("id")
                if not str_id:
                    continue
                    
                bid = str(str_id)
                payload = hit['payload']
                
                # 檢查 payload 是否缺少關鍵字段（classification, words_total, author 等）
                # novels_multi_vector collection 只有 id, name, intro, tags 四個字段
                missing_fields = not payload.get('classification') or not payload.get('words_total')
                
                if missing_fields:
                    # 從資料庫補充完整數據
                    db_item = self.db.get_item(bid)
                    if db_item:
                        # 合併：保留 payload 的 intro/tags（可能更新），補充資料庫的其他字段
                        full_item = {**db_item, **payload}  # payload 覆蓋 db_item
                        candidates_map[bid] = full_item
                    else:
                        # 資料庫也沒有？只能用 payload 的部分數據
                        candidates_map[bid] = payload
                else:
                    # Payload 已完整
                    candidates_map[bid] = payload
                
                vector_score_map[bid] = hit["score"]
                payload_map[bid] = payload
            else:
                # Fallback: try to get from database
                item = self.db.get_item(hit.get("id"))
                if item and item.get("name"):
                    bid = str(item["id"])
                    candidates_map[bid] = item
                    vector_score_map[bid] = hit["score"]
                    if hit.get('payload'):
                        payload_map[bid] = hit['payload']
        
        candidates = list(candidates_map.values())
        print(f"[Engine] Retrieved {len(candidates)} candidates after filtering")


        # 7. 评分和排序（纯分数，不归一化）
        scored_items = []
        for item in candidates:
            bid = str(item["id"])
            v_score = vector_score_map.get(bid, 0.0)
            
            # Calculate final score
            score_val, breakdown = self.calculate_score(
                item,
                parse_result.criteria,
                vector_score=v_score,
                tag_terms_list=tag_terms_list,
                tag_weights=tag_weights,
                negative_tag_terms=negative_tag_terms
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
        
        if not scored_items:
            print("[Engine] ℹ️ 無足夠相關結果")
            return {
                "query": user_query,
                "parsed_criteria": [c.dict() if hasattr(c, 'dict') else c.model_dump() for c in parse_result.criteria],
                "query_vector": query_vector,
                "results": [],
                "message": "資料庫中無相關書籍，請嘗試其他搜尋條件。",
                "engine": "HybridEngine (Simplified)"
            }
        
        final_results = scored_items[:limit]

        # 9. 生成解释（可选）
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
            "search_terms": parse_result.search_terms,
            "generated_keywords": parse_result.generated_keywords,
            "hypothetical_intro": parse_result.hypothetical_intro,
            "reference_tags": reference_tags,
            "query_vector": query_vector,
            "results": final_results,
            "engine": "HybridEngine (Simplified: Semantic + Filters)",
        }
