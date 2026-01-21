"""
Data Ingestion Script for Hybrid Reasoner

This script reads novel data from `data/books_crawled.json` and populates:
1. SQLite Database (Metadata via src.core.database.Database)
2. Qdrant Vector Store (Embeddings via src.core.vector_store.VectorStore)

Usage:
    python -m src.scripts.ingest_data
"""

import sys
import json
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database
from src.core.vector_store import VectorStore

DATA_FILE = project_root / "data" / "books_crawled.json"
BATCH_SIZE = 100  # For vector store batching


def load_data(filepath: Path):
    """Load JSON data from file."""
    print(f"Loading data from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")
    return data


def ingest_to_sqlite(db: Database, items: list):
    """Insert items into SQLite database."""
    print(f"Ingesting {len(items)} items into SQLite...")
    for i, item in enumerate(items):
        db.add_item(item)
        if (i + 1) % 500 == 0:
            print(f"  SQLite: {i + 1}/{len(items)} items ingested.")
    print(f"SQLite ingestion complete. Total: {len(items)} items.")


def ingest_to_qdrant(vs: VectorStore, items: list):
    """Embed and insert items into Qdrant vector store."""
    print(f"Ingesting {len(items)} items into Qdrant (batch size: {BATCH_SIZE})...")
    
    for start_idx in range(0, len(items), BATCH_SIZE):
        batch = items[start_idx : start_idx + BATCH_SIZE]
        vs.add_items(batch)
        print(f"  Qdrant: {min(start_idx + BATCH_SIZE, len(items))}/{len(items)} items ingested.")
    
    print(f"Qdrant ingestion complete. Total: {len(items)} items.")


def main():
    print("=== Data Ingestion Script ===")
    
    # Load Data
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found at {DATA_FILE}")
        return
    
    items = load_data(DATA_FILE)
    
    if not items:
        print("No data to ingest.")
        return
    
    # Initialize Stores
    print("\nInitializing Database and VectorStore...")
    db = Database()
    vs = VectorStore(collection_name="novels")
    
    # Ingest to SQLite
    print("\n--- Step 1: SQLite Ingestion ---")
    ingest_to_sqlite(db, items)
    
    # Ingest to Qdrant
    print("\n--- Step 2: Qdrant Ingestion ---")
    ingest_to_qdrant(vs, items)
    
    print("\n=== Ingestion Complete! ===")


if __name__ == "__main__":
    main()
