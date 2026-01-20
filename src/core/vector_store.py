from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from src.config import settings
from typing import List, Dict, Any, Optional
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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=384, # all-MiniLM-L6-v2 dimension
                    distance=rest.Distance.COSINE
                )
            )

    def search(
        self, 
        query_text: str, 
        limit: int = 10, 
        query_filter: Optional[rest.Filter] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search with optional filtering.
        
        Args:
            query_text: The search query text to embed and search.
            limit: Maximum number of results to return.
            query_filter: Optional Qdrant Filter object for logic push-down.
                          Filters are applied at the database level for efficiency.
        
        Returns:
            List of search results with id, score, and payload.
        """
        # Token limit handling (truncation)
        self.model.max_seq_length = 256
        
        vector = self.model.encode(query_text)
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,  # Logic push-down: filter at DB level
            limit=limit
        )
        search_result = response.points
        
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in search_result
        ]

    def add_items(self, items: List[Dict[str, Any]]):
        """
        Embeds and adds items to Qdrant.
        Expects items to have 'id' and 'text_content' (or similar) to embed.
        """
        # Batching could be added for performance
        points = []
        for item in items:
            # Construct rich text representation
            parts = [
                f"Title: {item.get('name', '')}",
                f"Author: {item.get('user', {}).get('name', '') if isinstance(item.get('user'), dict) else ''}",
                f"Slogan: {item.get('slogan', '')}",
                f"Tags: {', '.join([t.get('name') for t in item.get('tags', {}).get('data', [])] if isinstance(item.get('tags'), dict) else [])}",
                f"Intro: {item.get('intro', '')}"
            ]
            text = "\n".join([p for p in parts if p.strip()])
            
            if not text.strip():
                text = item.get("name", "")
            if not text:
                continue
                
            vector = self.model.encode(text)
            points.append(rest.PointStruct(
                id=item["id"],
                vector=vector,
                payload=item
            ))
            
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
