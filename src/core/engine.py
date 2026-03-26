import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.core.llm import parse_query
from src.models.schemas import QueryParseResult
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.core.book_matcher import BookMatcher
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
    def __init__(
        self, 
        db=None, 
        vs=None, 
        retrieval_mode: str = "multi_vector",
        semantic_weight: Optional[float] = None,
        attribute_weight: Optional[float] = None
    ):
        self.db = db if db is not None else Database()
        self.retrieval_mode = retrieval_mode
        
        # Load Fusion weights and mode from settings, with optional overrides
        self.semantic_weight = semantic_weight if semantic_weight is not None else settings.SEMANTIC_WEIGHT
        self.attribute_weight = attribute_weight if attribute_weight is not None else settings.ATTRIBUTE_WEIGHT
        
        # [USER-SET] Architecture Flags to avoid logic overlap
        self.is_feature_fusion = "fused" in retrieval_mode  # Exp 4
        self.is_exp3_mapping = "embedded_tags" in retrieval_mode  # Exp 3
        self.is_exp5_multi = retrieval_mode == "multi_vector"  # Exp 5
        self.is_baseline_hybrid = not (self.is_feature_fusion or self.is_exp3_mapping or self.is_exp5_multi) # Exp 1, 2

        if self.is_feature_fusion:
            # Exp 4: Feature Fusion (Single-Vector, All-in-one pre-fused)
            collection_name = "novels_fused"
            self.use_multi_vector = False
            print(f"[HybridEngine] Exp 4: Feature Fusion")
        elif self.is_exp5_multi:
            # Exp 5: Joined Matching (Multi-Vector Fusion)
            collection_name = "novels"
            self.use_multi_vector = True
            print(f"[HybridEngine] Exp 5: Multi-Vector Fusion")
        elif self.is_exp3_mapping:
            # Exp 3: Individual Mapping (Hard Matching)
            collection_name = "novels"
            self.use_multi_vector = False
            print(f"[HybridEngine] Exp 3: Tag Mapping")
        else:
            # Exp 1 & 2: Baseline Hybrid (Vector Text + SQL Hard Tag)
            collection_name = "novels"
            self.use_multi_vector = False
            print(f"[HybridEngine] Exp 1/2: Baseline Hybrid")
    
        self.vs = vs if vs is not None else VectorStore(collection_name=collection_name)
        self.book_matcher = BookMatcher(self.db)
        
        # Method 2 Cache: Pre-load tags if using baseline_prompt mode
        self.all_tags_cache = None
        if "prompt" in self.retrieval_mode:
            self._load_tags_cache()

    def _load_tags_cache(self):
        """Pre-load all tags from JSON for Method 2 to avoid frequent I/O."""
        import json
        import os
        tags_path = "data/all_tags.json"
        if os.path.exists(tags_path):
            try:
                # Try UTF-8 first
                with open(tags_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                # Fallback directly to UTF-16
                try:
                    with open(tags_path, "r", encoding="utf-16") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[HybridEngine] Warning: Failed to load {tags_path} with UTF-16: {e}")
                    data = []
            except Exception as e:
                print(f"[HybridEngine] Warning: Failed to load {tags_path}: {e}")
                data = []

            if data:
                # Convert to tuple so it's hashable for lru_cache
                self.all_tags_cache = tuple(data)
                print(f"[HybridEngine] Method 2: Pre-loaded {len(self.all_tags_cache)} tags for cache")
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
        text_vector_score: Optional[float] = None,
        tag_vector_score: Optional[float] = None,
        tag_mapping_weights: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        純內容相關性評分 (Pure Content Relevance Scoring)
        
        Track 1 - 語意音軌 (Semantic Track)
        Track 2 - 屬性音軌 (Attribute Track)
        """
        breakdown = []
        
        # ================================================================
        # Track 1: 語意音軌 (Semantic Track)
        # ================================================================
        if self.use_multi_vector and text_vector_score is not None:
            raw_semantic = text_vector_score
        else:
            raw_semantic = vector_score
        semantic_score = 0.1 + 0.9 * raw_semantic
        
        breakdown.append({
            "criteria": "semantic_track",
            "label": "語意音軌 (Semantic Track)",
            "raw_score": raw_semantic,
            "weighted_score": semantic_score,
            "is_filter": False,
            "reason": f"語意相似度: {semantic_score:.4f} (raw text: {raw_semantic:.4f})"
        })
        
        # ================================================================
        # Track 2: 屬性音軌 (Attribute Track) — 標籤評分
        # ================================================================
        attribute_score = 1.0
        has_tag_scoring = False
        
        # [Priority 1] Exp 5: Joined Tag Vector Similarity (Path B Result)
        if self.is_exp5_multi and tag_vector_score is not None:
            attribute_score = 0.1 + 0.9 * tag_vector_score
            has_tag_scoring = True
            breakdown.append({
                "criteria": "attr_tag_vector_joined",
                "label": "[屬性] 標籤全域向量相似度 (Exp5)",
                "raw_score": tag_vector_score,
                "weighted_score": attribute_score,
                "is_filter": False,
                "reason": f"標籤串聯向量分: {tag_vector_score:.4f} → 屬性分: {attribute_score:.4f}"
            })
            
        # [Priority 2] Exp 3: Individual Mapping (Facet-based MaxSim Scoring)
        elif self.is_exp3_mapping and tag_mapping_weights and tag_terms_list:
            book_tags = item.get("tags", [])
            if isinstance(book_tags, str):
                import json
                try: book_tags = json.loads(book_tags)
                except: book_tags = []
            
            # tag_mapping_weights is now List[Dict[SystemTag, Score]] (one per target_tag)
            n_facets = len(tag_mapping_weights)
            total_facet_score = 0.0
            matched_details = []
            
            for i, facet_map in enumerate(tag_mapping_weights):
                target_term = tag_terms_list[i] if i < len(tag_terms_list) else f"Facet_{i}"
                
                # Find the best match in this book for THIS specific query facet
                best_sim_for_facet = 0.0
                best_tag_for_facet = None
                
                for bt in book_tags:
                    sim = facet_map.get(bt, 0.0)
                    if sim > best_sim_for_facet:
                        best_sim_for_facet = sim
                        best_tag_for_facet = bt
                
                total_facet_score += best_sim_for_facet
                if best_sim_for_facet > 0:
                    matched_details.append(f"{target_term}→{best_tag_for_facet}({best_sim_for_facet:.2f})")
            
            if n_facets > 0:
                avg_similarity = total_facet_score / n_facets
                attribute_score = 0.1 + 0.9 * avg_similarity
                has_tag_scoring = True
                breakdown.append({
                    "criteria": "attr_tag_mapped",
                    "label": f"[屬性] 標籤語意映射評分 (Exp3)",
                    "raw_score": avg_similarity,
                    "weighted_score": attribute_score,
                    "is_filter": False,
                    "reason": f"面向命中分: {total_facet_score:.4f} / {n_facets} 面向 = {avg_similarity:.4f}, 詳情: [{', '.join(matched_details) if matched_details else '無'}] → 屬性分: {attribute_score:.4f}"
                })

        # [Priority 3] Exp 1/2: Hard Matching (Metadata vs req terms)
        elif self.is_baseline_hybrid and tag_terms_list:
            match_count = 0
            n_total = len(tag_terms_list)
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
                        matched_tags.append(b_tag)
                        break
            
            if n_total > 0:
                attribute_score = 0.1 + 0.9 * (match_count / n_total)
                has_tag_scoring = True
                breakdown.append({
                    "criteria": "attr_tag_match",
                    "label": f"[屬性] 標籤字面匹配 ({match_count}/{n_total})",
                    "raw_score": match_count,
                    "weighted_score": attribute_score,
                    "is_filter": False,
                    "reason": f"命中標籤: {', '.join(matched_tags) if matched_tags else '無'}"
                })
        
        # ================================================================
        # 全域融合 (Global Fusion)
        # ================================================================
        if has_tag_scoring:
            total_score = (semantic_score * self.semantic_weight) + (attribute_score * self.attribute_weight)
            fusion_label = f"線性融合: ({semantic_score:.4f} * {self.semantic_weight}) + ({attribute_score:.4f} * {self.attribute_weight})"
        else:
            total_score = semantic_score
            fusion_label = f"純語意分: {semantic_score:.4f}"
        
        breakdown.append({
            "criteria": "global_fusion",
            "label": "全域融合 (Global Fusion)",
            "raw_score": total_score,
            "weighted_score": total_score,
            "is_filter": False,
            "reason": fusion_label
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

    def _post_filter(
        self,
        scored_items: List[Dict[str, Any]],
        criteria_list: List[Any],
        negative_tag_terms: List[str],
    ) -> List[Dict[str, Any]]:
        """
        後置篩選層 (Post-Filter Layer)
        
        在引擎完成純內容相關性評分並排序後，以 Boolean 方式篩除
        不符合硬性約束的候選項。
        
        篩選條件：
        1. 負向標籤：命中任一排除標籤 → 移除
        2. 發布狀態：不符合指定狀態 → 移除
        3. 作者匹配：不符合指定作者 → 移除
        4. 字數範圍：超出指定範圍 → 移除
        
        Args:
            scored_items: 已評分並排序的候選列表
            criteria_list: LLM 解析出的條件列表
            negative_tag_terms: 已映射的負向標籤列表
        
        Returns:
            篩選後的候選列表（保持原排序）
        """
        filtered = []
        
        # 預解析硬性約束條件
        status_filter = None
        author_filter = None
        words_min = None
        words_max = None
        
        for criteria in criteria_list:
            if hasattr(criteria.parameters, 'model_dump'):
                params = criteria.parameters.model_dump()
            else:
                params = criteria.parameters.dict()
            
            if criteria.name == "status_check":
                target_status = params.get("target_status", "").lower()
                completed_kw = ["complet", "finish", "ended", "done", "完結", "完结", "已完結", "已完结"]
                ongoing_kw = ["ongoing", "serializ", "running", "active", "連載", "连载", "連載中", "连载中"]
                if any(x in target_status for x in completed_kw):
                    status_filter = "完結"
                elif any(x in target_status for x in ongoing_kw):
                    status_filter = "連載"
            
            elif criteria.name == "author_match":
                author_filter = params.get("author_name", "").strip()
            
            elif criteria.name == "numeric_range" and params.get("field") == "words_total":
                words_min = params.get("min_val")
                words_max = params.get("max_val")
        
        for res in scored_items:
            item = res["item"]
            excluded = False
            
            # 1. 負向標籤篩除
            if negative_tag_terms:
                book_tags = item.get("tags", [])
                if isinstance(book_tags, str):
                    import json
                    try: book_tags = json.loads(book_tags)
                    except: book_tags = []
                
                for neg_t in negative_tag_terms:
                    for b_t in book_tags:
                        if neg_t in b_t or b_t in neg_t:
                            excluded = True
                            break
                    if excluded:
                        break
            
            # 2. 狀態篩除
            if not excluded and status_filter:
                actual_status = item.get("publish_status", "")
                if actual_status != status_filter:
                    excluded = True
            
            # 3. 作者篩除
            if not excluded and author_filter:
                actual_author = item.get("author", "")
                if not (author_filter in actual_author or actual_author in author_filter):
                    excluded = True
            
            # 4. 字數範圍篩除
            if not excluded and (words_min is not None or words_max is not None):
                actual_words = item.get("words_total", 0) or 0
                if words_min is not None and actual_words < words_min:
                    excluded = True
                if not excluded and words_max is not None and actual_words > words_max:
                    excluded = True
            
            if not excluded:
                filtered.append(res)
        
        print(f"[PostFilter] 篩選結果: {len(scored_items)} → {len(filtered)} (移除 {len(scored_items) - len(filtered)} 筆)")
        return filtered

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        搜索流程：語意檢索 → 內容評分 → 後置篩選
        
        架構分離原則：
        - 引擎負責「內容符合需求」的語意評分
        - 硬性約束（負向標籤、狀態、作者、字數）由後置篩選層處理
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
        
        # 1.6 [Exp 3 核心修復] 預計算標籤語意映射表 (Target Tag -> System Tag Score Map)
        tag_mapping_weights = {}
        tag_terms_list = []
        if parse_result.generated_keywords:
            tag_terms_list.extend([kw.replace(" ", "") for kw in parse_result.generated_keywords])
        if reference_tags:
            tag_terms_list.extend([t.replace(" ", "") for t in reference_tags[:8]])

        if self.is_exp3_mapping and tag_terms_list:
            print(f"[Engine] Exp 3: Pre-mapping {len(tag_terms_list)} target tags to system tags...")
            tag_mapping_weights = self.vs.batch_map_tags(tag_terms_list)
            print(f"[Engine] Exp 3: Mapped into {len(tag_mapping_weights)} facets with weight maps.")

        # 2. 不使用 Qdrant 硬過濾器，硬性約束由後置篩選層 _post_filter 處理
        qdrant_filter = None
        
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
            # [FIX] Only Exp 4 (Feature Fusion) should add tags to the SINGLE vector query string
            if self.is_feature_fusion:
                print(f"[Engine] Exp 4: adding LLM-expanded keywords to text query: {expansion_str}")
                expanded_terms += f" {expansion_str}"
            elif self.use_multi_vector:
                print(f"[Engine] Multi-Vector: Keywords reserved for tag_semantic: {expansion_str}")
            else:
                print(f"[Engine] Baseline/Exp3: Keywords handled separately (SQL or Score Fusion)")
        
        # 3.3 添加参考小说的标签到查询中
        if reference_tags:
            tags_str = " ".join(reference_tags[:8])
            # [FIX] Only Exp 4 (Feature Fusion) should add tags to the SINGLE vector query string
            if self.is_feature_fusion:
                print(f"[Engine] Exp 4: adding reference tags to text query: {tags_str}")
                expanded_terms += f" {tags_str}"
            elif self.use_multi_vector:
                print(f"[Engine] Multi-Vector: Reference tags reserved for tag_semantic: {tags_str}")
            else:
                print(f"[Engine] Baseline/Exp3: Reference tags handled separately (SQL or Score Fusion)")
        
        # 3.4 HyDE 假設文檔嵌入 (Disabled manually)
        # if parse_result.hypothetical_intro:
        #     print(f"[Engine] HyDE hypothetical intro: {parse_result.hypothetical_intro[:80]}...")
        #     expanded_terms += f" {parse_result.hypothetical_intro}"
        
        tag_query_text = " ".join(tag_terms_list)
        print(f"[Engine] Pure tags query for tag_semantic: '{tag_query_text}'")

        # --- 負向標籤處理 (Negative Tag Filter Analysis) ---
        negative_tag_terms = []
        negative_criteria_list = [c for c in parse_result.criteria if c.name == "semantic_similarity" and getattr(c, 'is_negative', False)]
        
        for nc in negative_criteria_list:
            p = nc.parameters.model_dump() if hasattr(nc.parameters, 'model_dump') else nc.parameters.dict()
            qt = p.get("query_text", "").strip()
            if not qt: continue
            
            # [USER-SET] 語意映射僅限於 Exp 3 (Individual Mapping)
            # 確保其餘實驗（含 Exp 4）不會混入映射邏輯，維持實驗對稱性
            if self.is_exp3_mapping:
                try:
                    neg_mapped = self.vs.search_tags(f"標籤： {qt}", limit=1, similarity_threshold=0.7)
                    if neg_mapped:
                        mapped_tags = [res["tag"] for res in neg_mapped]
                        negative_tag_terms.extend(mapped_tags)
                        print(f"[Engine] Exp 3: Negative mapped tags for '{qt}': {mapped_tags}")
                    else:
                        negative_tag_terms.append(qt)
                except Exception as e:
                    print(f"[Engine] Warning: Negative tag mapping failed: {e}")
                    negative_tag_terms.append(qt)
            else:
                # Exp 1, 2, 4, 5: 直接使用原始解析結果进行後置篩選
                negative_tag_terms.append(qt)

                
        # [USER-SET] Exp 4 (Feature Fusion): Must format query into structure before embedding
        if "fused" in self.retrieval_mode:
            # Structure: [TITLE] ... [/TITLE] [TAGS] ... [/TAGS] [ABSTRACT] ... [/ABSTRACT]
            pseudo_title = " ".join(parse_result.reference_books) if parse_result.reference_books else ""
            pseudo_tags = tag_query_text
            pseudo_abstract = base_terms
            if "semantic_expansion" in locals() and "semantic_expansion" in dir():
               pseudo_abstract += f" {semantic_expansion}"
            
            expanded_terms = f"[TITLE] {pseudo_title} [/TITLE] [TAGS] {pseudo_tags} [/TAGS] [ABSTRACT] {pseudo_abstract} [/ABSTRACT]"
            print(f"[Engine] Exp 4 (Single-Vector) Fused Query Format: {expanded_terms}")
        
        
        # 4. 执行召回 (Hybrid Retrieval Logic)
        retrieval_limit = 10000  # 過採樣緩衝：後置篩選可能移除大量候選
        candidates_map = {}
        vector_score_map = {}
        text_score_map = {}
        tag_score_map = {}
        payload_map = {}
        
        # Path A: Core Vector Search (Text/Semantic) - SAME for ALL Experiments
        # This ensures the "Semantic Track" entry point is consistent.
        vector_results, query_vector = self.vs.search(
            expanded_terms,
            limit=retrieval_limit,
            query_filter=None,  # 不再使用硬過濾
            with_payload=True
        )
            
        # Collect results from Track 1 (Semantic)
        for hit in vector_results:
            if hit.get('payload') and hit['payload'].get("id"):
                bid = str(hit['payload']["id"])
                candidates_map[bid] = hit['payload']
                vector_score_map[bid] = hit["score"]
                text_score_map[bid] = hit["score"]
                payload_map[bid] = hit['payload']

        # Path B: Specific Attribute Retrieval (Tag Paths)
        # Exp 3: Individual Tag Vector Search (MaxSim)
        if self.is_exp3_mapping and tag_terms_list:
            print(f"[Engine] Exp 3: Triggering Path B (Individual Tag Search) for: {tag_terms_list}")
            tag_results = self.vs.search_individual(
                [f"標籤： {t}" for t in tag_terms_list],
                limit=retrieval_limit,
                collection_name="novels_tags"
            )
            for hit in tag_results:
                bid = str(hit.get('payload', {}).get("id", hit.get("id")))
                if bid not in candidates_map:
                    candidates_map[bid] = hit.get('payload', {})
                    payload_map[bid] = hit.get('payload', {})
                    text_score_map[bid] = 0.0
                    vector_score_map[bid] = 0.0
                tag_score_map[bid] = hit["score"]

        # Exp 5: Joined Tag Vector Search
        if self.is_exp5_multi and tag_query_text:
            print(f"[Engine] Exp 5: Triggering Path B (Joined Tag Search) for: '{tag_query_text}'")
            tag_results, _ = self.vs.search(
                f"標籤： {tag_query_text}",
                limit=retrieval_limit,
                collection_name="novels_tags"
            )
            for hit in tag_results:
                bid = str(hit['payload']["id"]) if hit.get('payload') else str(hit['id'])
                if bid not in candidates_map:
                    # Found via tags but not text
                    candidates_map[bid] = hit.get('payload', {})
                    payload_map[bid] = hit.get('payload', {})
                    text_score_map[bid] = 0.0 # No text match score
                    vector_score_map[bid] = 0.0
                tag_score_map[bid] = hit["score"]

        # Exp 1 & 2: SQL Fuzzy Tag Search
        if self.is_baseline_hybrid and tag_terms_list:
            print(f"[Engine] Exp 1/2: Triggering Hybrid Retrieval Path B (SQL Tag Search) for: {tag_terms_list}")
            sql_results = self.db.search_by_tags_fuzzy(tag_terms_list, limit=50)
            
            for item in sql_results:
                bid = str(item["id"])
                if bid not in candidates_map:
                    # [Hybrid] This book matched tags but NOT the vector search initially
                    candidates_map[bid] = item
                    vector_score_map[bid] = 0.0  # It has no vector score yet
                    payload_map[bid] = item
                    print(f"  + Added via SQL Tag Match: 《{item.get('name')}》")

        # 5. 补充资料 (Ensure candidates have full fields)
        candidates = []
        for bid, item in candidates_map.items():
            # If item is missing critical fields (common in slim vector payloads)
            if not item.get('classification') or not item.get('words_total'):
                db_item = self.db.get_item(bid)
                if db_item:
                    item = {**db_item, **item}
                    candidates_map[bid] = item
            candidates.append(item)
            
        print(f"[Engine] Final Hybrid Candidate Pool size: {len(candidates)}")


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
                text_vector_score=text_score_map.get(bid),
                tag_vector_score=tag_score_map.get(bid),
                tag_mapping_weights=tag_mapping_weights,
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
        
        # 後置篩選：移除不符合硬性約束的候選項
        scored_items = self._post_filter(
            scored_items,
            parse_result.criteria,
            negative_tag_terms,
        )
        
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
