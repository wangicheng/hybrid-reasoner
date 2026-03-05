from sentence_transformers import CrossEncoder
import torch
from typing import List, Dict, Any, Tuple

class Reranker:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            # Load a lightweight but effective Cross-Encoder model
            # ms-marco-MiniLM-L-6-v2 is fast and good for passage ranking
            print("[Reranker] Loading Cross-Encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2...")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
            except Exception as e:
                print(f"[Reranker] Failed to load model: {e}")
                cls._model = None
                
        return cls._instance

    @staticmethod
    def _build_doc_text(item: Dict[str, Any]) -> str:
        """Builds a unified text representation of an item for Cross-Encoder input."""
        title = item.get("title") or item.get("name") or ""

        tags = item.get("tags") or []
        if isinstance(tags, list):
            tags_text = ", ".join([str(tag) for tag in tags if tag is not None])
        else:
            tags_text = str(tags)

        intro = item.get("description") or item.get("summary") or item.get("intro") or ""

        chapters = item.get("chapters")
        if chapters is None and isinstance(item.get("attributes"), dict):
            chapters = item.get("attributes", {}).get("chapters")

        chapter_titles = []
        if isinstance(chapters, list):
            for chapter in chapters:
                if isinstance(chapter, dict):
                    chapter_titles.append(str(chapter.get("title") or chapter.get("name") or ""))
                else:
                    chapter_titles.append(str(chapter))
        chapters_text = " | ".join([c for c in chapter_titles if c])

        return (
            f"Title: {title}\n"
            f"Tags: {tags_text}\n"
            f"Intro: {intro}\n"
            f"Chapters: {chapters_text}"
        )

    async def score_feature(self, feature_text: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        Uses the Cross-Encoder to independently score a specific semantic feature
        (e.g. "主角聰明") against each candidate item asynchronously.

        Args:
            feature_text: The semantic feature string extracted by LLM.
            candidates: List of item dictionaries.

        Returns:
            A list of floats (0.0~1.0) corresponding to each candidate's relevance
            to the given feature. Returns list of 0.5 if model is unavailable.
        """
        if not self._model or not candidates:
            return [0.5] * len(candidates)

        pairs = [(feature_text, self._build_doc_text(item)) for item in candidates]

        try:
            import asyncio
            # Run the heavy model prediction in a separate thread so it doesn't block the async event loop
            logits = await asyncio.to_thread(self._model.predict, pairs, batch_size=32, show_progress_bar=False)
        except Exception as e:
            print(f"[Reranker] score_feature prediction failed: {e}")
            return [0.5] * len(candidates)

        # Normalize logits to 0~1 via sigmoid
        import math
        def _sigmoid(x: float) -> float:
            if x > 20:
                return 1.0
            if x < -20:
                return 0.0
            return 1.0 / (1.0 + math.exp(-x))

        return [_sigmoid(float(s)) for s in logits]

    async def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate items based on the query using Cross-Encoder asynchronously.
        
        Args:
            query: The user's search query.
            candidates: List of item dictionaries. Must contain 'title' and 'description' or 'summary'.
            top_k: Number of items to return after reranking.
            
        Returns:
            A list of reranked items with updated 'rerank_score'.
        """
        if not self._model or not candidates:
            return candidates[:top_k]

        pairs = [(query, self._build_doc_text(item)) for item in candidates]

        try:
            import asyncio
            scores = await asyncio.to_thread(self._model.predict, pairs, batch_size=32, show_progress_bar=False)
        except Exception as e:
            print(f"[Reranker] Prediction failed: {e}")
            return candidates[:top_k]

        for i, item in enumerate(candidates):
            item["rerank_score"] = float(scores[i])
            item["original_rank"] = i
            
        ranked_candidates = sorted(candidates, key=lambda x: x.get("rerank_score", -999), reverse=True)
        
        return ranked_candidates[:top_k]

