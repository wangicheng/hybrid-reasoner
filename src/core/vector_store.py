from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from google import genai
from google.genai import types
from src.config import settings
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
        # Embed query text using text-embedding-004
        embed_response = self.genai_client.models.embed_content(
            model=self.embedding_model,
            contents=query_text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        vector = embed_response.embeddings[0].values
        
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

    def add_items(self, items: List[Dict[str, Any]]):
        """
        Embeds and adds items to Qdrant in batches.
        """
        valid_items = []
        texts_to_embed = []
        
        for item in items:
            # --- Handle different schema structures (MirrorFiction vs Linovelib) ---
            # 1. Author
            author_name = ""
            if isinstance(item.get('user'), dict):
                # Old Schema
                author_name = item.get('user', {}).get('name', '')
            else:
                # New Schema (Simple string)
                author_name = item.get('author', '')

            # 2. Tags
            tag_names = []
            tags_raw = item.get('tags')
            if isinstance(tags_raw, dict) and 'data' in tags_raw:
                # Old Schema: {'data': [{'name': 'tag1'}, ...]}
                tag_names = [t.get('name') for t in tags_raw.get('data', [])]
            elif isinstance(tags_raw, list):
                # New Schema: ['tag1', 'tag2']
                tag_names = [str(t) for t in tags_raw]

            # Construct rich text representation
            content_snippet = item.get('content', '')[:500] # 擷取代表性內文
            parts = [
                f"書名: {item.get('name', '')}",
                f"作者: {author_name}",
                f"標語: {item.get('slogan', '')}",
                f"標籤: {', '.join(tag_names)}",
                f"簡介: {item.get('intro', '')}",
                f"內文片段: {content_snippet}"
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
            
            from src.core.api_utils import retry_on_rate_limit
            
            @retry_on_rate_limit(max_retries=3, base_delay=10.0)
            def _embed():
                return self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
            try:
                response = _embed()
            except Exception as e:
                print(f"Failed to embed batch after retries: {e}")
                continue
                
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
