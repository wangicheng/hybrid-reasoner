import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from src.core.llm import parse_query
from src.core.explainer import generate_explanation

def test_gemini_api():
    print("=== Testing Gemini API Integration ===")
    
    # Check API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return

    print(f"✅ API Key found: {api_key[:5]}...")

    # 1. Test Query Parsing
    print("\n--- 1. Testing Query Parsing (LLM) ---")
    query = "Completed romance novels with magic"
    print(f"Query: {query}")
    try:
        result = parse_query(query)
        print("✅ Parse Successful!")
        print(f"Original Query: {result.original_query}")
        print(f"Search Terms: {result.search_terms}")
        print(f"Criteria: {result.criteria}")
    except Exception as e:
        print(f"❌ Parse Failed: {e}")
        return

    # 2. Test Explanation Generation
    print("\n--- 2. Testing Explanation Generation (LLM) ---")
    
    # Mock book item
    book_item = {
        "name": "The Magical Romance",
        "author": "Test Author",
        "tags": ["magic", "romance"],
        "intro": "In a world where magic defines your soulmate, a young wizard falls in love with a non-magical girl. Their journey tests the boundaries of their society."
    }
    
    # Mock chunks
    context_chunks = [
        book_item["intro"],
        "Review: This book was absolutely enchanting! The magic system is unique.",
        "Chapter 1: The sparks flew when they first touched, literal sparks of blue fire."
    ]

    try:
        explanation = generate_explanation(
            query=query,
            book_item=book_item,
            context_chunks=context_chunks
        )
        print("✅ Explanation Generation Successful!")
        print(f"Generated Explanation:\n{explanation}")
    except Exception as e:
        print(f"❌ Explanation Generation Failed: {e}")

if __name__ == "__main__":
    test_gemini_api()
