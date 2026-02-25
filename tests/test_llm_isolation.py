
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from src.core.llm import parse_query
import os

# Mock environment if needed (assuming user has .env loaded or env vars set in terminal)
# We won't set API key here to avoid leaking, assuming it's in env.

def test_parse_query():
    print("🚀 Testing parse_query in isolation...")
    
    # Test 1: Simple query
    query = "網遊小說"
    print(f"\nScanning query: {query}")
    try:
        result = parse_query(query)
        print("✅ Parse Successful!")
        print(f"   Original Query: {result.original_query}")
        print(f"   Criteria Count: {len(result.criteria)}")
        for c in result.criteria:
            print(f"   - {c.name}: {c.parameters}")
    except Exception as e:
        print(f"❌ Parse Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Complex query with potential validation issues (Gemma often fails here)
    query_complex = "字數超過200萬字的已完結小說，主角要聰明，不要小白文"
    print(f"\nScanning complex query: {query_complex}")
    try:
        result = parse_query(query_complex)
        print("✅ Parse Successful!")
        print(f"   Criteria Count: {len(result.criteria)}")
    except Exception as e:
        print(f"❌ Parse Failed: {e}")

if __name__ == "__main__":
    test_parse_query()
