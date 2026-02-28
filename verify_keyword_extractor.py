from src.core.keyword_extractor import KeywordExtractor

def test_extraction():
    extractor = KeywordExtractor()
    query = "找一本字數超過十萬字的奇幻小說，主角要是廢柴逆襲，最好有系統。"
    
    print(f"Original Query: {query}")
    
    print("\n--- KeyBERT Extraction ---")
    bert_kw = extractor.extract_keywords(query, top_k=5)
    print(bert_kw)
    
    print("\n--- TF-IDF Extraction ---")
    tfidf_kw = extractor.extract_tfidf(query, top_k=5)
    print(tfidf_kw)
    
    print("\n--- Hybrid Extraction ---")
    hybrid = extractor.hybrid_extract(query, top_k=5)
    print(hybrid)

if __name__ == "__main__":
    test_extraction()
