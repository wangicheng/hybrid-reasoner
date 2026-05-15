import json
import jieba
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from src.core.database import Database
from src.config import settings

class LexicalStore:
    def __init__(self, db: Optional[Database] = None):
        self.db = db if db is not None else Database()
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_items: List[Dict[str, Any]] = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Naive tokenization using jieba
        if not text:
            return []
        return list(jieba.cut_for_search(text))

    def _build_index(self) -> None:
        items = self.db.get_all_items()
        tokenized_corpus = []
        self.corpus_items = []
        
        for item in items:
            if not item.get("id"):
                continue
            
            # Use title, author, and tags for lexical indexing (intro removed to reduce noise)
            parts = []
            if item.get("name"):
                parts.append(str(item["name"]))
            if item.get("author"):
                parts.append(str(item["author"]))
            tags = item.get("tags")
            if isinstance(tags, list):
                parts.extend(str(t) for t in tags)
                
            text = " ".join(parts)
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)
            self.corpus_items.append(item)
            
        if tokenized_corpus:
            # We can pass k1, b parameters if using a custom BM25 class 
            # or just rely on defaults of BM25Okapi for standard Okapi.
            # rank_bm25's BM25Okapi takes k1 and b.
            k1 = getattr(settings, "BM25_K1", 1.5)
            b = getattr(settings, "BM25_B", 0.75)
            self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        else:
            self.bm25 = None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns list of dicts with {"item": item_dict, "score": bm25_score}
        """
        if not self.bm25 or not query:
            return []
            
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        # Get top K
        # argsort scores descending
        import numpy as np
        top_n = np.argsort(scores)[::-1][:limit]
        
        results = []
        for idx in top_n:
            score = float(scores[idx])
            if score > 0:
                results.append({
                    "item": self.corpus_items[idx],
                    "score": score
                })
                
        return results
