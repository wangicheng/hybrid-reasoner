#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detailed debug trace of search flow."""

import sys
from pathlib import Path
import asyncio

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.engine import HybridEngine
from src.core.vector_store import VectorStore
from src.core.database import Database
from qdrant_client import QdrantClient
from src.config import settings


async def debug_search():
    """Trace the search pipeline step by step."""
    
    print("=" * 70)
    print("DETAILED SEARCH DEBUGGING")
    print("=" * 70)
    
    # Step 1: Check Qdrant collections
    print("\n[STEP 1] Checking Qdrant Collections...")
    client = QdrantClient(path=settings.QDRANT_PATH)
    
    try:
        collections_list = client.get_collections()
        print(f"  Available collections: {len(collections_list.collections)}")
        for col in collections_list.collections:
            col_info = client.get_collection(col.name)
            print(f"    - {col.name}: {col_info.points_count} points")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Initialize components
    print("\n[STEP 2] Initializing Components...")
    db = Database()
    print(f"  Database: OK")
    
    # Use the same client we already have to avoid locking issues
    vs_fused = VectorStore(collection_name="novels_fused")
    vs_fused.client = client  # Reuse the client
    print(f"  VectorStore (novels_fused): OK - using collection '{vs_fused.collection_name}'")
    
    # Step 3: Initialize HybridEngine
    print("\n[STEP 3] Initializing HybridEngine...")
    engine = HybridEngine(use_fused_vectors=True)
    engine.vs.client = client  # Reuse the client
    print(f"  HybridEngine: OK")
    print(f"  Using fused vectors: {engine.use_fused_vectors}")
    print(f"  Engine VectorStore collection: '{engine.vs.collection_name}'")
    
    # Step 4: Test vector search directly
    print("\n[STEP 4] Direct Vector Search (bypassing HybridEngine)...")
    test_query = "adventure knight"
    print(f"  Query: '{test_query}'")
    
    try:
        results, vector = vs_fused.search(test_query, limit=5, with_payload=True)
        print(f"  Results from novels_fused: {len(results)} items")
        if results:
            for i, result in enumerate(results[:3], 1):
                print(f"    {i}. Score: {result['score']:.4f}")
                if result.get('payload'):
                    print(f"       Name: {result['payload'].get('name', 'N/A')}")
        print(f"  Query vector dimension: {len(vector)}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Test parse_query
    print("\n[STEP 5] Testing parse_query (LLM)...")
    from src.core.llm import parse_query
    try:
        parse_result = parse_query(test_query)
        print(f"  Original query: {parse_result.original_query}")
        print(f"  Search terms: {parse_result.search_terms}")
        print(f"  Generated keywords: {parse_result.generated_keywords[:3] if parse_result.generated_keywords else 'None'}")
        print(f"  Hypothetical intro: {parse_result.hypothetical_intro[:60] if parse_result.hypothetical_intro else 'None'}...")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Test HybridEngine.search
    print("\n[STEP 6] Testing HybridEngine.search()...")
    try:
        result = await engine.search(test_query, limit=3, explain=False)
        print(f"  Query: {result.get('query')}")
        print(f"  Candidates found: {len(result.get('candidates', []))}")
        if result.get('candidates'):
            for i, candidate in enumerate(result['candidates'][:3], 1):
                print(f"    {i}. {candidate.get('name', 'N/A')} - Score: {candidate.get('final_score', 'N/A')}")
        else:
            print(f"  Message: {result.get('message', 'N/A')}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Check collection data directly
    print("\n[STEP 7] Checking novels_fused Collection Data...")
    try:
        info = client.get_collection("novels_fused")
        print(f"  Points in novels_fused: {info.points_count}")
        
        # Get a sample point
        scroll_result = client.scroll(collection_name="novels_fused", limit=1)
        if scroll_result[0]:
            point = scroll_result[0][0]
            print(f"  Sample point ID: {point.id}")
            if point.payload:
                print(f"  Sample payload keys: {list(point.payload.keys())}")
                print(f"  Sample name: {point.payload.get('name', 'N/A')}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 70)
    return


if __name__ == "__main__":
    asyncio.run(debug_search())
