import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from google import genai
from google.genai import types
from src.config import settings
from src.core.api_utils import retry_on_rate_limit
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
        
        # Initialize Google GenAI client
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.genai_client = genai.Client(api_key=api_key)
        self.embedding_model = "gemini-embedding-001"
        
        self._ensure_collection()

    def _ensure_collection(self):
        """
        Creates a collection with Named Vectors:
          - "content": for book title + intro (used in semantic retrieval)
          - "tags":    for classification + tags (used in tag-level retrieval)
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "content": rest.VectorParams(
                        size=3072,  # gemini-embedding-001 dimension
                        distance=rest.Distance.COSINE
                    ),
                    "tags": rest.VectorParams(
                        size=3072,
                        distance=rest.Distance.COSINE
                    ),
                }
            )

    def search(
        self, 
        query_text: str, 
        limit: int = 10, 
        query_filter: Optional[rest.Filter] = None,
        with_payload: bool = True,
        vector_name: str = "content",
        task_type: str = "RETRIEVAL_QUERY",
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Performs semantic search on a specific named vector.
        
        Args:
            query_text: The search query text to embed and search.
            limit: Maximum number of results to return.
            query_filter: Optional Qdrant Filter object for logic push-down.
            with_payload: Whether to return the payload with the results.
            vector_name: Which named vector to search against ("content" or "tags").
            task_type: Embedding task type ("RETRIEVAL_QUERY" for content,
                       "SEMANTIC_SIMILARITY" for tags).
        
        Returns:
            Tuple of (search results, query vector).
        """
        embed_response = self.genai_client.models.embed_content(
            model=self.embedding_model,
            contents=query_text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        vector = embed_response.embeddings[0].values
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            using=vector_name,  # Named vector selection
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload
        )
        search_result = response.points
        
        formatted_results = [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in search_result
        ]
        
        return formatted_results, vector

    @staticmethod
    def _build_content_text(item: Dict[str, Any]) -> str:
        """Builds text for the content vector (title + intro)."""
        parts = [
            f"書名: {item.get('name', '')}",
            f"簡介: {item.get('intro', '')}",
        ]
        text = "\n".join([p for p in parts if p.strip()])
        if not text.strip():
            text = item.get("name", "")
        return text

    @staticmethod
    def _build_tags_text(item: Dict[str, Any]) -> str:
        """Builds text for the tags vector (classification + tags)."""
        tag_names = []
        tags_raw = item.get('tags')
        if isinstance(tags_raw, list):
            tag_names = [str(t) for t in tags_raw]
        
        classification = item.get('classification', '')
        
        parts = []
        if classification:
            parts.append(classification)
        if tag_names:
            parts.append(', '.join(tag_names))
        
        text = ' '.join(parts)
        if not text.strip():
            # Fallback: use book name if no tags/classification
            text = item.get("name", "")
        return text

    def add_items(self, items: List[Dict[str, Any]]):
        """
        Embeds and adds items to Qdrant with dual named vectors (content + tags).
        """
        valid_items = []
        content_texts = []
        tags_texts = []
        
        for item in items:
            content_text = self._build_content_text(item)
            tags_text = self._build_tags_text(item)
            
            if not content_text:
                continue
                
            content_texts.append(content_text)
            tags_texts.append(tags_text)
            valid_items.append(item)
            
        # --- RESUME LOGIC ---
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
            
        print(f"  [Resume] Found {len(existing_ids)} existing items in Qdrant. Skipping...")

        batch_size = 50
        from google.genai.errors import ClientError
        
        for i in range(0, len(valid_items), batch_size):
            batch_content_texts = []
            batch_tags_texts = []
            batch_items = []
            
            for content_t, tags_t, item in zip(
                content_texts[i:i+batch_size],
                tags_texts[i:i+batch_size],
                valid_items[i:i+batch_size]
            ):
                if item["id"] not in existing_ids:
                    batch_content_texts.append(content_t)
                    batch_tags_texts.append(tags_t)
                    batch_items.append(item)
                    
            if not batch_items:
                continue
            
            # --- Embed content texts ---
            @retry_on_rate_limit(max_retries=3, base_delay=10.0)
            def _embed_content():
                return self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch_content_texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
            
            # --- Embed tags texts ---
            @retry_on_rate_limit(max_retries=3, base_delay=10.0)
            def _embed_tags():
                return self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch_tags_texts,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                
            try:
                content_response = _embed_content()
                time.sleep(1)  # Brief pause between API calls
                tags_response = _embed_tags()
            except Exception as e:
                print(f"Failed to embed batch after retries: {e}")
                continue
                
            if not content_response or not tags_response:
                continue
            
            points = []
            for item, content_emb, tags_emb in zip(
                batch_items,
                content_response.embeddings,
                tags_response.embeddings
            ):
                points.append(rest.PointStruct(
                    id=item["id"],
                    vector={
                        "content": content_emb.values,
                        "tags": tags_emb.values,
                    },
                    payload=item
                ))
                
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            print(f"  [Batch] Embedded {len(points)} items (content + tags)")
            
            if i + batch_size < len(valid_items) and batch_items:
                time.sleep(2)
