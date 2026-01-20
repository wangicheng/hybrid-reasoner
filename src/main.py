import argparse
import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from src.core.llm import parse_query
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.logic.registry import ScoringRegistry
import src.logic.scoring_functions 


from src.core.engine import HybridEngine
from src.core.database import Database
from src.core.vector_store import VectorStore

def seed_data(db: Database, vs: VectorStore):
    """Populates DB and VectorStore with crawled data."""
    data_path = Path("data/books_crawled.json")
    if not data_path.exists():
        print("No crawled data found at data/books_crawled.json. Please run src/crawler.py first.")
        return

    print("Loading crawled data...")
    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    # Transform keys if necessary, but our DB expects dicts that roughly match
    # Database.add_item handles extraction.
    
    for item in items:
        # DB add
        db.add_item(item)
    
    # Vector Seed
    # Vector store expects items with 'id' and 'intro'/'name'
    vs.add_items(items)
    print(f"Seeding complete. Added {len(items)} items.")

def main():
    parser = argparse.ArgumentParser(description="Hybrid Reasoner CLI (Novels)")
    parser.add_argument("--query", type=str, help="Natural language query")
    parser.add_argument("--seed", action="store_true", help="Seed database with crawled data")
    args = parser.parse_args()
    
    # Initialize separate instances for seeding if needed, or just let Engine handle it (Engine has its own DB/VS init)
    # But for seeding we need direct access.
    
    if args.seed:
        db = Database()
        vs = VectorStore(collection_name="novels")
        seed_data(db, vs)
        if not args.query:
            return

    if not args.query:
        print("Please provide a query with --query")
        return

    print(f"Processing query: {args.query}")
    
    engine = HybridEngine()
    search_result = engine.search(args.query, limit=5)
    
    print(f"Parsed Criteria: {search_result['parsed_criteria']}")
    print("\nTop Recommendations:")
    
    for res in search_result['results']:
        item = res['item']
        score = res['score']
        status = "完結" if item["publish_status"] == "completed" else "連載"
        print(f"[{score:.4f}] {item['name']} ({status}) - {item['classification']}")
        print(f"         Tags: {item['tags']}")
        print(f"         Intro: {item['intro'][:50]}...")
        print("-" * 40)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
