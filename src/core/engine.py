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
        self.fusion_mode = "multiplicative" if "multiplicative" in retrieval_mode else "additive"
        
        # Determine collection name and use_multi_vector flag
        if retrieval_mode.startswith("multi_"):
            collection_name = "novels_multi_vector"
            self.use_multi_vector = True
            print(f"[HybridEngine] Using multi-vector embeddings for semantic search ({retrieval_mode})")
            print(f"[HybridEngine] Vectors: text_semantic (Title+Introduction) + tag_semantic (Tags)")
        else:
            collection_name = "novels_fused" if retrieval_mode == "fused" else "novels"
            self.use_multi_vector = False
            if retrieval_mode == "fused":
                print("[HybridEngine] Using fused embeddings for semantic search")
                print("[HybridEngine] Fused content: Title + Tags + Introduction")
            else:
                print("[HybridEngine] Using baseline embeddings for semantic search")
        
        self.vs = vs if vs is not None else VectorStore(collection_name=collection_name)
        self.book_matcher = BookMatcher(self.db)

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
        negative_semantic_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        简化评分逻辑：只计算语义分数（纯分数，不归一化）
        
        Args:
            item: 候选书籍项
            criteria_list: 评分条件列表
            vector_score: 原始向量分数
            negative_semantic_scores: 负向语义分数字典 {item_id: negative_score}
        
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
        
        # --- 2. 负向语义分数（纯分数）---
        item_id = str(item.get("id"))
        if negative_semantic_scores and item_id in negative_semantic_scores:
            neg_score = negative_semantic_scores[item_id]
            total_score -= neg_score
            
            breakdown.append({
                "criteria": "semantic_similarity",
                "label": "[排除] 負向語意",
                "raw_score": neg_score,
                "weighted_score": -neg_score,
                "is_negative": True,
                "is_filter": False,
                "reason": f"排除內容相似度: {neg_score:.4f}"
            })

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
        parse_result = parse_query(user_query, model_id=model_id)
        
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
        
        # 构建 pure tags string for tag_semantic
        tag_terms_list = []
        if parse_result.generated_keywords:
            tag_terms_list.extend([kw.replace(" ", "") for kw in parse_result.generated_keywords])
        if reference_tags:
            tag_terms_list.extend([t.replace(" ", "") for t in reference_tags[:8]])
        tag_query_text = " ".join(tag_terms_list)
        print(f"[Engine] Pure tags query for tag_semantic: '{tag_query_text}'")
        
        # 4. 执行正向语义搜索（带硬过滤）
        retrieval_limit = 100  # 减少检索数量，因为有硬过滤
        
        # Use multi-vector search if available
        query_vector = None
        if self.use_multi_vector:
            vector_results, query_vector = self.vs.search_multi_vector(
                expanded_terms,
                limit=retrieval_limit,
                query_filter=qdrant_filter,  # 应用硬过滤
                with_payload=True,
                text_weight=0.7,  # Text semantic priority
                tag_weight=0.3,   # Tag semantic secondary
                fusion_mode=self.fusion_mode,
                tag_query_text=tag_query_text
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

        # 6. 计算负向语义分数（如果有排除条件）
        negative_semantic_scores = {}
        negative_criteria = [c for c in parse_result.criteria 
                             if c.name == "semantic_similarity" and getattr(c, 'is_negative', False)]
        
        if negative_criteria and candidates:
            print(f"[Engine] Computing negative semantic scores for {len(negative_criteria)} exclusion(s)...")
            
            # 对每个负向条件进行向量嵌入
            for neg_crit in negative_criteria:
                params = neg_crit.parameters.model_dump() if hasattr(neg_crit.parameters, 'model_dump') else neg_crit.parameters.dict()
                neg_query_text = params.get("query_text", "")
                
                if not neg_query_text:
                    continue
                
                print(f"[Engine] Negative semantic: '{neg_query_text}'")
                
                # 嵌入负向查询
                if self.use_multi_vector:
                    neg_results, neg_vector = self.vs.search_multi_vector(
                        neg_query_text,
                        limit=len(candidates),  # 只需要对候选项计算
                        query_filter=None,  # 不应用过滤
                        with_payload=False,  # 只需要分数
                        text_weight=0.7,
                        tag_weight=0.3,
                        fusion_mode=self.fusion_mode
                    )
                else:
                    neg_results, neg_vector = self.vs.search(
                        neg_query_text,
                        limit=len(candidates),
                        query_filter=None,
                        with_payload=False
                    )
                
                # 构建负向分数映射
                neg_score_map = {str(hit.get('payload', {}).get('id', hit.get('id'))): hit['score'] 
                                 for hit in neg_results if hit.get('score')}
                
                # 累加每个候选项的负向分数（纯分数）
                for bid in candidates_map.keys():
                    if bid in neg_score_map:
                        current_neg = negative_semantic_scores.get(bid, 0.0)
                        raw_neg = neg_score_map[bid]
                        negative_semantic_scores[bid] = current_neg + raw_neg

        # 7. 评分和排序（纯分数，不归一化）
        scored_items = []
        for item in candidates:
            bid = str(item["id"])
            v_score = vector_score_map.get(bid, 0.0)
            
            # Calculate final score with negative semantics (raw scores only)
            score_val, breakdown = self.calculate_score(
                item,
                parse_result.criteria,
                vector_score=v_score,
                negative_semantic_scores=negative_semantic_scores,
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
