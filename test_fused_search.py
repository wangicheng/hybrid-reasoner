#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test fused embedding search functionality."""

import sys
import asyncio
import os
from pathlib import Path

# Set UTF-8 encoding for stdout
os.environ['PYTHONIOENCODING'] = 'utf-8'

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.engine import HybridEngine
from src.config import settings


async def test_fused_search():
    """Test search with fused embeddings."""
    print("=" * 70)
    print("Testing Fused Embedding Search")
    print("=" * 70)
    
    # Initialize engine with fused vectors
    print("\n[1/2] Initializing HybridEngine with fused vectors...")
    engine = HybridEngine(use_fused_vectors=True)
    print("      [OK] Engine initialized")
    
    # Test search
    print("\n[2/2] Testing search queries...")
    test_queries = [
        "magic girl",
        "martial arts adventure",
        "sci-fi future",
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        try:
            results = await engine.search(query, limit=3)
            
            if results and results.get('candidates'):
                candidates = results['candidates']
                print(f"  Found {len(candidates)} candidates:")
                for i, result in enumerate(candidates[:3], 1):
                    score = result.get('final_score', 'N/A')
                    title = result.get('name', 'N/A')
                    author = result.get('author', 'N/A')
                    if isinstance(score, (int, float)):
                        print(f"    {i}. [{score:.4f}] {title} - {author}")
                    else:
                        print(f"    {i}. [score] {title} - {author}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Fused embedding search test completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(test_fused_search())
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        sys.exit(1)
