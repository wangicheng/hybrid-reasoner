import asyncio
import sys
import os
from dotenv import load_dotenv

# Load env before anything else
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.llm import parse_query
from src.core.keyword_extractor import KeywordExtractor

async def compare_extraction():
    query = "找一本字數超過十萬字的奇幻小說，主角要是廢柴逆襲，最好有系統。"
    print(f"Target Query: {query}\n")

    print(f"Loading LLM and Keyword Extractor models...")
    extractor = KeywordExtractor()

    # 1. LLM Extraction (The "Brain")
    print("\nXXX [1. LLM Extraction (With Hints)] XXX")
    print("Sending query to Google Gemini (with KeyBERT hints)...")
    try:
        # This calls the current implementation which HAS the hints
        llm_result_with_hints = parse_query(query)
        
        print(f"→ Search Terms: {llm_result_with_hints.search_terms}")
        print(f"→ Generated Keywords: {llm_result_with_hints.generated_keywords}")
            
    except Exception as e:
        print(f"LLM Error: {e}")

    # 3. Simulate "Old" LLM (No Hints) - We can't easily undo the code change, 
    # but we can observe that the 'Reflexes' below are what's being fed INTO the LLM now.
    
    # 2. Keyword Extractor (The "Reflexes" / NER)
    print("\nXXX [2. Keyword Extractor Only (The Hints)] XXX")
    print("Running KeyBERT + TF-IDF...")
    
    bert_kw = extractor.extract_keywords(query, top_k=5)
    tfidf_kw = extractor.extract_tfidf(query, top_k=5)
    hybrid = extractor.hybrid_extract(query, top_k=5)
    
    print(f"→ KeyBERT (Semantic Focus): {bert_kw}")
    print(f"→ TF-IDF (Frequency Focus): {tfidf_kw}")
    print(f"→ Hybrid (Combined): {hybrid}")

    print("\n=== Analysis ===")
    print("LLM understands 'intent' (e.g., 'system' -> 'Genre: System').")
    print("Keyword Extractor captures 'important nouns' (e.g., 'Novel', 'Fantasy').")

if __name__ == "__main__":
    asyncio.run(compare_extraction())
