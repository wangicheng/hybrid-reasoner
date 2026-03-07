#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified debug trace - check what's in Qdrant."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from src.config import settings
import google.genai as genai
from google.genai import types

def debug_search():
    """Check Qdrant data and test search."""
    
    print("=" * 70)
    print("QDRANT DATA AND SEARCH DEBUG")
    print("=" * 70)
    
    # Initialize Qdrant
    print("\n[1] Connecting to Qdrant...")
    client = QdrantClient(path=settings.QDRANT_PATH)
    print("  OK - Connected")
    
    # Check collections
    print("\n[2] Checking Collections...")
    collections_list = client.get_collections()
    print(f"  Found {len(collections_list.collections)} collections:")
    for col in collections_list.collections:
        col_info = client.get_collection(col.name)
        print(f"    - {col.name}: {col_info.points_count} points")
    
    # Check novels_fused data
    print("\n[3] Sampling novels_fused Data...")
    try:
        scroll_result = client.scroll(collection_name="novels_fused", limit=3)
        points = scroll_result[0]
        print(f"  Retrieved {len(points)} sample points:")
        for point in points:
            payload = point.payload
            print(f"    ID: {point.id}")
            print(f"    Name: {payload.get('name', 'N/A')}")
            print(f"    Has vector: {hasattr(point, 'vector') and point.vector is not None}")
            print()
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # Test vector search directly
    print("\n[4] Testing Vector Search...")
    
    # Initialize Gemini for embedding
    genai_client = genai.Client(api_key=settings.GOOGLE_API_KEYS[0])
    
    test_query = "adventure knight"
    print(f"  Query: '{test_query}'")
    
    try:
        # Generate embedding
        response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=test_query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        
        query_vector = response.embeddings[0].values
        print(f"  Query vector dimension: {len(query_vector)}")
        
        # Search in Qdrant
        search_results = client.query_points(
            collection_name="novels_fused",
            query=query_vector,
            limit=5,
            with_payload=True
        )
        
        print(f"\n  Search Results ({len(search_results.points)} found):")
        for i, point in enumerate(search_results.points, 1):
            payload = point.payload
            print(f"\n    {i}. Score: {point.score:.4f}")
            print(f"       Name: {payload.get('name', 'N/A')}")
            print(f"       Author: {payload.get('author', 'N/A')}")
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Debug completed")
    print("=" * 70)


if __name__ == "__main__":
    try:
        debug_search()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
