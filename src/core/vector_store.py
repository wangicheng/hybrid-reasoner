from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from google import genai
from google.genai import types
from src.config import settings
from src.core.api_utils import get_api_key_rotator
from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

class VectorStore:
    def __init__(self, collection_name: str = "items"):
        if settings.QDRANT_PATH == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            path = Path(settings.QDRANT_PATH)
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path.resolve()))
        self.collection_name = collection_name
        
        # Initialize Google GenAI client with API key rotation
        from src.core.api_utils import get_current_api_key
        api_key = get_current_api_key()
        self.genai_client = genai.Client(api_key=api_key)
        self.embedding_model = "gemini-embedding-001"
        
        self._ensure_collection()

    @staticmethod
    def build_fused_text(item: Dict[str, Any]) -> str:
        """
        融合書籍信息為結構化文本用於向量化。
        格式: [TITLE] 書名 [/TITLE] [TAGS] tag1 tag2 tag3 [/TAGS] [ABSTRACT] 簡介 [/ABSTRACT]
        
        研究表明：結構化標記能幫助模型更好區分各個內容部分，提升語意理解效果。
        
        Args:
            item: 書籍資訊字典，包含 name, tags, intro
        
        Returns:
            融合後的結構化文本
        """
        # 書名
        title = item.get('name', '').strip()
        
        # 標籤 - 支援多種格式 (列表或 JSON string)
        tags_raw = item.get('tags', [])
        if isinstance(tags_raw, str):
            try:
                import json
                tags_raw = json.loads(tags_raw)
            except:
                tags_raw = []
        
        if isinstance(tags_raw, list):
            tags_text = ' '.join([str(t).strip() for t in tags_raw if t])
        else:
            tags_text = ''
        
        # 簡介
        abstract = item.get('intro', '').strip()
        
        # 組合成結構化格式
        fused_text = f"[TITLE] {title} [/TITLE] [TAGS] {tags_text} [/TAGS] [ABSTRACT] {abstract} [/ABSTRACT]"
        
        return fused_text
    
    def _update_api_key_on_rate_limit(self):
        """Update the genai client with a new API key when rate limit is hit."""
        from src.core.api_utils import get_api_key_rotator
        try:
            rotator = get_api_key_rotator()
            new_key = rotator.on_rate_limit_error()
            self.genai_client = genai.Client(api_key=new_key)
            print(f"[VectorStore] API key rotated. Now using key index: {rotator.current_index}")
        except Exception as e:
            print(f"[VectorStore] Failed to rotate API key: {e}")
            raise

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=3072, # gemini-embedding-001 dimension
                    distance=rest.Distance.COSINE
                )
            )

    def _embed_with_retry(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        """Embeds text with automatic retry and API key rotation."""
        from src.core.api_utils import _is_retryable, get_rate_limiter
        import time
        
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            try:
                # Enforce shared rate limit
                get_rate_limiter().wait()
                
                response = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                return list(response.embeddings[0].values)
            except Exception as e:
                attempt += 1
                if _is_retryable(e) and attempt < max_attempts:
                    print(f"[VectorStore] Embedding retryable error: {e}. Rotating key...")
                    self._update_api_key_on_rate_limit()
                    time.sleep(1) # Small buffer
                else:
                    raise
        raise Exception("Max embed retries exceeded")

    def search(
        self, 
        query_text: str, 
        limit: int = 10, 
        query_filter: Optional[rest.Filter] = None,
        with_payload: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Performs semantic search with optional filtering.
        
        Args:
            query_text: The search query text to embed and search.
            limit: Maximum number of results to return.
            query_filter: Optional Qdrant Filter object for logic push-down.
                          Filters are applied at the database level for efficiency.
            with_payload: Whether to return the payload with the results.
        
        Returns:
            Tuple of (search results, query vector).
        """
        # Embed query text with retry/rotation
        vector = self._embed_with_retry(query_text, task_type="RETRIEVAL_QUERY")
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,  # Logic push-down: filter at DB level
            limit=limit,
            with_payload=with_payload
        )
        search_result = response.points
        
        formatted_results = [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in search_result
        ]
        
        return formatted_results, vector

    def search_multi_vector(
        self,
        query_text: str,
        limit: int = 10,
        query_filter: Optional[rest.Filter] = None,
        with_payload: bool = True,
        text_weight: float = 0.7,
        tag_weight: float = 0.3,
        batch_size: int = 20,  # Limit pre-fetch to avoid excessive merging
        fusion_mode: str = "multiplicative",
        tag_query_text: str = "",
        tag_query_list: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Performs Late Interaction multi-vector semantic search (text + tag).

        This implementation queries both vector spaces independently, collects
        per-space scores, then computes the fused score.

        Exp 3 vs Exp 5 Difference:
        - Exp 3 (Individual): Embeds each query tag separately and uses MaxSim aggregation.
        - Exp 5 (Joined): Embeds tags as a single joined string.
        """
        # Step 1: Embed query text with retry
        query_vector = self._embed_with_retry(query_text, task_type="RETRIEVAL_QUERY")
        
        # Step 2: Prepare Tag Queries
        fetch_limit = max(batch_size, limit * 5)
        text_scores: Dict[Any, float] = {}
        tag_scores: Dict[Any, float] = {}
        payload_cache: Dict[Any, Any] = {}

        # 1. Query Text collection (Baseline collection)
        text_response = self.client.query_points(
            collection_name="novels",
            query=query_vector,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=with_payload
        )
        for hit in text_response.points:
            text_scores[hit.id] = float(hit.score)
            if hit.payload:
                payload_cache[hit.id] = hit.payload

        # 2. Query Tag-only collection
        # Note: novels_tags uses the same UUID IDs as novels
        if tag_query_list and len(tag_query_list) > 0:
            # [Exp 3] Individual Matching
            print(f"[VectorStore] Exp 3: Individual matching for {len(tag_query_list)} tags")
            from google.genai import types
            
            # Batch embed query tags
            embed_resp = self.genai_client.models.embed_content(
                model=self.embedding_model,
                contents=tag_query_list,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            tag_vectors = [list(e.values) for e in embed_resp.embeddings]
            
            # Search across tags individually
            for v in tag_vectors:
                indiv_tag_response = self.client.query_points(
                    collection_name="novels_tags",
                    query=v,
                    query_filter=query_filter,
                    limit=fetch_limit,
                    score_threshold=0.85, # [Exp 3] Must be > 0.85 as per user request
                    with_payload=with_payload
                )
                
                # Aggregate tag scores using MaxSim (best match for any tag)
                for hit in indiv_tag_response.points:
                    tag_scores[hit.id] = max(tag_scores.get(hit.id, 0.0), float(hit.score))
                    if hit.payload and hit.id not in payload_cache:
                        payload_cache[hit.id] = hit.payload
        else:
            # [Exp 5] Joined Matching
            if tag_query_text and tag_query_text.strip():
                print(f"[VectorStore] Exp 5: Joined matching for tags: '{tag_query_text}'")
                tag_query_vector = self._embed_with_retry(tag_query_text.strip(), task_type="RETRIEVAL_QUERY")
            else:
                tag_query_vector = query_vector

            tag_response = self.client.query_points(
                collection_name="novels_tags",
                query=tag_query_vector,
                query_filter=query_filter, 
                limit=fetch_limit,
                with_payload=with_payload
            )
            for hit in tag_response.points:
                tag_scores[hit.id] = float(hit.score)
                if hit.payload and hit.id not in payload_cache:
                    payload_cache[hit.id] = hit.payload

        # Step 3: Fusion
        all_ids = set(text_scores.keys()) | set(tag_scores.keys())
        fused_map: Dict[Any, Dict[str, float]] = {}
        
        for pid in all_ids:
            t_raw = text_scores.get(pid, 0.0)
            g_raw = tag_scores.get(pid, 0.0)

            if fusion_mode == "additive":
                fused = (t_raw * text_weight) + (g_raw * tag_weight)
            else:  # "multiplicative"
                # Scale found scores by 10 (0.X → X.X) then multiply.
                # Missing component defaults to 1.0 (neutral in multiplication).
                t_comp = (t_raw * 10) if pid in text_scores else 1.0
                g_comp = (g_raw * 10) if pid in tag_scores else 1.0
                fused = t_comp * g_comp

            fused_map[pid] = {"fused": fused, "text_score": t_raw, "tag_score": g_raw}

        # Sort and return top-k
        sorted_ids = sorted(fused_map.items(), key=lambda x: x[1]["fused"], reverse=True)[:limit]
        formatted_results: List[Dict[str, Any]] = []
        for pid, metrics in sorted_ids:
            formatted_results.append({
                "id": pid,
                "score": metrics["fused"],
                "payload": payload_cache.get(pid),
                "text_score": metrics.get("text_score", 0.0),
                "tag_score": metrics.get("tag_score", 0.0),
            })

        return formatted_results, query_vector

    def search_tags(
        self,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for actual tags semantically similar to the provided query text.
        """
        embed_response = self.genai_client.models.embed_content(
            model=self.embedding_model,
            contents=query_text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = list(embed_response.embeddings[0].values)
        
        response = self.client.query_points(
            collection_name="novel_tags",
            query=query_vector,
            limit=limit,
            score_threshold=similarity_threshold,
            with_payload=True
        )
        
        results = []
        for hit in response.points:
            if hit.payload and "tag" in hit.payload:
                results.append({
                    "tag": hit.payload["tag"],
                    "score": hit.score
                })
        return results

    def add_items(self, items: List[Dict[str, Any]]):
        """
        Embeds and adds items to Qdrant in batches.
        """
        valid_items = []
        texts_to_embed = []
        
        for item in items:
            author_name = item.get('author', '')

            # 2. Tags
            tag_names = []
            tags_raw = item.get('tags')
            if isinstance(tags_raw, list):
                tag_names = [str(t) for t in tags_raw]

            # Construct rich text representation
            parts = [
                f"書名: {item.get('name', '')}",
                f"簡介: {item.get('intro', '')}",
            ]
            text = "\n".join([p for p in parts if p.strip()])
            
            if not text.strip():
                text = item.get("name", "")
            if not text:
                continue
                
            texts_to_embed.append(text)
            valid_items.append(item)
            
        # --- RESUME LOGIC ---
        # Fetch existing IDs in the collection to skip them
        existing_ids = set()
        try:
            scroll_point = None
            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=scroll_point,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False
                )
                points, scroll_point = response
                existing_ids.update(p.id for p in points)
                if scroll_point is None:
                    break
        except Exception:
            pass # Collection might be empty or missing
            
        print(f"  [Resume] Found {len(existing_ids)} existing items in Qdrant. Skipping...")

        batch_size = 50
        import time
        from google.genai.errors import ClientError
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = []
            batch_items = []
            
            # Filter batch items that are not already in DB
            for text, item in zip(texts_to_embed[i:i+batch_size], valid_items[i:i+batch_size]):
                if item["id"] not in existing_ids:
                    batch_texts.append(text)
                    batch_items.append(item)
                    
            if not batch_texts:
                continue # Skip batch if all items are already embedded
            
            from src.core.api_utils import retry_on_rate_limit, _is_retryable
            # Retry with API key rotation on rate limit
            attempt = 0
            max_attempts = 3
            while attempt < max_attempts:
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                    
                    if is_rate_limit and attempt < max_attempts - 1:
                        print(f"[VectorStore] Rate limit detected. Rotating API key...")
                        self._update_api_key_on_rate_limit()
                        attempt += 1
                        time.sleep(5)
                    else:
                        print(f"Failed to embed batch after attempts: {e}")
                        raise
                
            if not response:
                continue
            
            points = []
            for item, embedding in zip(batch_items, response.embeddings):
                points.append(rest.PointStruct(
                    id=item["id"],
                    vector=embedding.values,
                    payload=item
                ))
                
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            # Avoid hitting rate limits rapidly
            if i + batch_size < len(texts_to_embed) and batch_texts:
                time.sleep(2)

    def add_fused_items(self, items: List[Dict[str, Any]]):
        """
        為書籍添加融合向量（書名 + 標籤 + 簡介融合）。
        融合後的向量存儲到單獨的 collection。
        
        Args:
            items: 書籍信息字典列表
        """
        from src.core.api_utils import get_api_key_rotator
        import hashlib
        
        valid_items = []
        texts_to_embed = []
        id_mappings = {}  # Map string IDs to numeric IDs
        
        # 為每個項目生成融合文本
        for item in items:
            fused_text = self.build_fused_text(item)
            
            if not fused_text.strip():
                continue
            
            texts_to_embed.append(fused_text)
            valid_items.append(item)
            
            # Create numeric ID from string ID using hash
            str_id = str(item.get("id", ""))
            numeric_id = int(hashlib.md5(str_id.encode()).hexdigest(), 16) % (2**63)
            id_mappings[len(valid_items) - 1] = (numeric_id, str_id)
        
        if not texts_to_embed:
            print("[add_fused_items] No valid items to process")
            return
        
        # 檢查已存在的項目以支援恢復邏輯
        existing_ids = set()
        try:
            scroll_point = None
            while True:
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=scroll_point,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False
                )
                points, scroll_point = response
                existing_ids.update(p.id for p in points)
                if scroll_point is None:
                    break
        except Exception:
            pass
        
        print(f"[add_fused_items] Found {len(existing_ids)} existing items, skipping...")
        
        batch_size = 50
        import time
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = []
            batch_items = []
            batch_ids = []
            
            # 過濾已存在的項目
            for idx in range(i, min(i + batch_size, len(texts_to_embed))):
                numeric_id, str_id = id_mappings[idx]
                if numeric_id not in existing_ids:
                    batch_texts.append(texts_to_embed[idx])
                    batch_items.append(valid_items[idx])
                    batch_ids.append(numeric_id)
            
            if not batch_texts:
                continue
            
            # 使用 API key 輪換重試融合向量計算
            attempt = 0
            max_attempts = 5
            response = None
            while attempt < max_attempts:
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                    is_invalid_key = "API key not valid" in error_str or "INVALID_ARGUMENT" in error_str
                    
                    if (is_rate_limit or is_invalid_key) and attempt < max_attempts - 1:
                        if is_invalid_key:
                            rotator = get_api_key_rotator()
                            old_idx = rotator.current_index
                            rotator.rotate()
                            new_idx = rotator.current_index
                            print(f"[add_fused_items] Invalid key {old_idx}, rotating to key {new_idx}...")
                            self._update_api_key_on_rate_limit()
                        else:
                            print(f"[add_fused_items] Rate limit, rotating API key...")
                            self._update_api_key_on_rate_limit()
                        attempt += 1
                        time.sleep(5)
                    else:
                        print(f"[add_fused_items] Embedding failed: {e}")
                        raise
            
            if not response:
                continue
            
            # 構造向量點
            points = []
            for numeric_id, item, embedding in zip(batch_ids, batch_items, response.embeddings):
                points.append(rest.PointStruct(
                    id=numeric_id,
                    vector=embedding.values,
                    payload=item
                ))
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"[add_fused_items] Uploaded {len(points)} fused vectors")
            
            # 避免過快觸發速率限制
            if i + batch_size < len(texts_to_embed) and batch_texts:
                time.sleep(2)
        
        print(f"[add_fused_items] Done processing {len(valid_items)} items")
