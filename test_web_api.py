"""
Test the web API multi-vector search
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_web_api():
    """Test web API search endpoint"""
    
    test_queries = [
        "adventure knight",
        "mystery romance",
        "fantasy magic"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing Query: {query}")
        print(f"{'='*60}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/search",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                print(f"Status: OK (200)")
                print(f"Results found: {len(results.get('results', []))}")
                
                for i, item in enumerate(results.get('results', [])[:3], 1):
                    print(f"\n{i}. {item.get('name', 'Unknown')}")
                    print(f"   Score: {item.get('final_score', 0.0):.4f}")
                    if item.get('vector_score'):
                        print(f"   Vector Score: {item.get('vector_score'):.4f}")
                    print(f"   Author: {item.get('author', 'Unknown')}")
                        
            else:
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\n{'='*60}")
    print("Web API test completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_web_api()
