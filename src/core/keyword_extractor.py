import jieba
import jieba.analyse
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class KeywordExtractor:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeywordExtractor, cls).__new__(cls)
            # Initialize jieba
            jieba.initialize()
            # Explicitly load a multilingual or Chinese model better suited for keywords
            # If the project uses a specific model, we should match it, but 'paraphrase-multilingual' is safer for mixed
            # However, to avoid downloading large new models, we may want to reuse the one from VectorStore if possible.
            # But here we load a small one or the same one.
            # Let's use 'all-MiniLM-L6-v2' to match vector_store.py to save memory/download time
            cls._model = SentenceTransformer('all-MiniLM-L6-v2') 
        return cls._instance

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extracts keywords using a KeyBERT-inspired approach with jieba segmentation.
        """
        if not text:
            return []
            
        # 1. Candidate Selection (Jieba)
        # Combine jieba.cut (segmentation) with POS filtering could be better, but keep it simple first.
        # Adding simple stopword filtering
        stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '我們', '你們', '他們', '找', '一本', '想', '要', '有', '甚麼', '什麼', '推薦', '小說'}
        candidates = [w for w in jieba.cut(text) if len(w) > 1 and w not in stopwords]
        
        # Remove duplicates while preserving order
        candidates = list(dict.fromkeys(candidates))
        
        if not candidates:
            return []

        # 2. Embedding Generation
        # Encode doc and candidates
        try:
            # Check if model is loaded (it should be in __new__)
            if self._model is None:
                 self._model = SentenceTransformer('all-MiniLM-L6-v2') 
                 
            doc_embedding = self._model.encode([text])
            candidate_embeddings = self._model.encode(candidates)

            # 3. Similarity Calculation
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            
            # 4. Ranking
            keywords_idx = np.argsort(distances[0])[-top_k:]
            keywords = [candidates[index] for index in keywords_idx]
            keywords.reverse()
            
            return keywords
        except Exception as e:
            print(f"[KeywordExtractor] Error in KeyBERT extraction: {e}")
            return candidates[:top_k]

    def extract_tfidf(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extracts keywords using Jieba's TF-IDF algorithm. Good for catching specific nouns.
        """
        # Enhance with custom stopwords if needed
        return jieba.analyse.extract_tags(text, topK=top_k)

    def extract_textrank(self, text: str, top_k: int = 5) -> List[str]:
        return jieba.analyse.textrank(text, topK=top_k)

    def hybrid_extract(self, text: str, top_k: int = 8) -> List[str]:
        """
        Combines Semantic (BERT), TF-IDF, and TextRank.
        Returns a broad set of keywords to help the LLM.
        """
        bert_kw = self.extract_keywords(text, top_k=top_k)
        tfidf_kw = self.extract_tfidf(text, top_k=top_k)
        textrank_kw = self.extract_textrank(text, top_k=top_k)
        
        # Merge: BERT > TextRank > TF-IDF
        merged = []
        seen = set()
        
        for kw in bert_kw + textrank_kw + tfidf_kw:
            if kw not in seen:
                merged.append(kw)
                seen.add(kw)
        
        return merged[:top_k]
