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

    def _embed_with_retry(self, text: Any, task_type: str = "RETRIEVAL_QUERY") -> Any:
        """Embeds text or list of texts with automatic retry and API key rotation."""
        from src.core.api_utils import _is_retryable, get_rate_limiter
        import time
        
        attempt = 0
        max_attempts = 5
        is_list = isinstance(text, list)
        
        while attempt < max_attempts:
            try:
                # Enforce shared rate limit
                get_rate_limiter().wait()
                
                response = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                
                if is_list:
                    return [list(e.values) for e in response.embeddings]
                else:
                    return list(response.embeddings[0].values)
                    
            except Exception as e:
                attempt += 1
                error_str = str(e)
                # Check for 429 or other retryable errors
                is_quota_error = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                
                if (is_quota_error or _is_retryable(e)) and attempt < max_attempts:
                    print(f"[VectorStore] Embedding error ({error_str[:50]}...). Rotating key and retrying (attempt {attempt})...")
                    self._update_api_key_on_rate_limit()
                    time.sleep(2 * attempt) # Incremental backoff
                else:
                    raise
        raise Exception("Max embed retries exceeded")

    def search(
        self, 
        query_text: str, 
        limit: int = 10000, 
        query_filter: Optional[rest.Filter] = None,
        with_payload: bool = True,
        collection_name: Optional[str] = None
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
        
        target_collection = collection_name if collection_name else self.collection_name
        
        response = self.client.query_points(
            collection_name=target_collection,
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


    def search_individual(
        self,
        query_list: List[str],
        limit: int = 10000,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        [Exp 3 Strategy] 個別向量化搜尋 + MaxSim 聚合。
        為每個查詢詞分別搜尋，並對每本書保留最高的相似度得分。
        """
        if not query_list:
            return []

        target_collection = collection_name if collection_name else self.collection_name
        
        # 1. 批次嵌入所有查詢標籤 (使用重試機制)
        query_vectors = self._embed_with_retry(query_list, task_type="RETRIEVAL_QUERY")
        
        # 2. 執行多次檢索並進行 MaxSim 聚合
        aggregated_scores: Dict[Any, float] = {}
        payload_cache: Dict[Any, Any] = {}
        
        for v in query_vectors:
            response = self.client.query_points(
                collection_name=target_collection,
                query=v,
                limit=limit * 2, # 稍微擴大範圍以確保覆蓋率
                with_payload=True
            )
            for hit in response.points:
                bid = hit.id
                score = float(hit.score)
                if bid not in aggregated_scores or score > aggregated_scores[bid]:
                    aggregated_scores[bid] = score
                    if hit.payload:
                        payload_cache[bid] = hit.payload
        
        # 3. 排序並過濾
        sorted_results = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [
            {"id": bid, "score": score, "payload": payload_cache.get(bid, {})}
            for bid, score in sorted_results
        ]

    def search_tags(
        self,
        query_text: str,
        limit: int = 10000,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for actual tags semantically similar to the provided query text.
        """
        query_vector = self._embed_with_retry(query_text, task_type="RETRIEVAL_QUERY")
        
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

    def batch_map_tags(
        self,
        target_tags: List[str],
        similarity_threshold: float = 0.6,
        limit_per_tag: int = 10000
    ) -> List[Dict[str, float]]:
        """
        [Exp 3 核心修復] 將用戶要求的標籤詞映射到系統實際標籤。
        返回 List[Dict[系統標籤名, 相似度分數]]，列表順序與 target_tags 一致。
        """
        if not target_tags:
            return []
            
        try:
            # 1. 批次嵌入 (使用重試機制)
            target_texts = [f"標籤： {t}" for t in target_tags]
            query_vectors = self._embed_with_retry(target_texts, task_type="RETRIEVAL_QUERY")
            
            results_list = []
            # 2. 為每個要求標籤尋找最接近的系統標籤
            for vector in query_vectors:
                mapping = {}
                response = self.client.query_points(
                    collection_name="novel_tags",
                    query=vector,
                    limit=limit_per_tag,
                    score_threshold=similarity_threshold,
                    with_payload=True
                )
                for hit in response.points:
                    if hit.payload and "tag" in hit.payload:
                        tag_name = hit.payload["tag"]
                        score = float(hit.score)
                        if tag_name not in mapping or score > mapping[tag_name]:
                            mapping[tag_name] = score
                results_list.append(mapping)
                            
            return results_list
        except Exception as e:
            print(f"[VectorStore] Batch tag mapping failed: {e}")
            return []

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
