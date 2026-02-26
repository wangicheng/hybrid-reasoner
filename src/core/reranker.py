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

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate items based on the query using Cross-Encoder.
        
        Args:
            query: The user's search query.
            candidates: List of item dictionaries. Must contain 'title' and 'description' or 'summary'.
            top_k: Number of items to return after reranking.
            
        Returns:
            A list of reranked items with updated 'rerank_score'.
        """
        if not self._model or not candidates:
            return candidates[:top_k]

        # 1. Prepare pairs for the model: [ (Query, Doc1), (Query, Doc2), ... ]
        # Combine title and description for better context
        pairs = []
        for item in candidates:
            # Handle potential missing fields gracefully
            title = item.get("title", "")
            desc = item.get("description") or item.get("summary") or item.get("intro") or ""
            # A good format for novels is usually: "Title: {title}. Summary: {desc}"
            doc_text = f"Title: {title}. Summary: {desc[:300]}" # Truncate long descriptions for speed
            pairs.append((query, doc_text))

        # 2. Predict scores
        # scores will be an array of floats (logits or probabilities depending on model)
        # ms-marco models output logits (unbounded), so higher is better.
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            print(f"[Reranker] Prediction failed: {e}")
            return candidates[:top_k]

        # 3. Attach scores and sort
        for i, item in enumerate(candidates):
            item["rerank_score"] = float(scores[i])
            item["original_rank"] = i
            
        # 4. Sort descending by rerank_score
        # Fix: Ensure we are sorting a list of dicts, and handling potential errors
        ranked_candidates = sorted(candidates, key=lambda x: x.get("rerank_score", -999), reverse=True)
        
        return ranked_candidates[:top_k]

