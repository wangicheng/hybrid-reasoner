"""
Data Ingestion Script for Linovelib Data

This script reads novel data from `data/books_crawled.json` and populates:
1. SQLite Database (Metadata via src.core.database.Database)
2. Qdrant Vector Store (Embeddings via src.core.vector_store.VectorStore)

Usage:
    python -m src.scripts.ingest_linovelib
"""

import sys
import json
import uuid
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database
from src.core.vector_store import VectorStore

DATA_FILE = project_root / "data" / "books_crawled.json"
BATCH_SIZE = 500  # For vector store batching


def generate_uuid(unique_string: str) -> str:
    """Generate a consistent UUID from a string ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))


def load_data(filepath: Path):
    """Load JSON data from file."""
    print(f"Loading data from {filepath}...")
    if not filepath.exists():
        print(f"Error: File {filepath} not found. Please run src/scripts/crawler_linovelib.py first.")
        return []
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items.")
    
    # Pre-process IDs to be valid UUIDs for Qdrant
    for item in data:
        original_id = str(item['id'])
        # Store original ID in payload just in case we need it later
        item['original_id'] = original_id
        # Convert ID to UUID
        item['id'] = generate_uuid(original_id)
        
    return data


def ingest_to_sqlite(db: Database, items: list):
    """Insert items into SQLite database."""
    print(f"Ingesting {len(items)} items into SQLite...")
    for i, item in enumerate(items):
        # Ensure ID is string (just in case)
        item['id'] = str(item['id'])
        db.add_item(item)
        if (i + 1) % 500 == 0:
            print(f"  SQLite: {i + 1}/{len(items)} items ingested.")
    print(f"SQLite ingestion complete. Total: {len(items)} items.")


def ingest_to_qdrant(vs: VectorStore, items: list):
    """Embed and insert items into Qdrant vector store."""
    print(f"Ingesting {len(items)} items into Qdrant (batch size: {BATCH_SIZE})...")
    
    # Ensure ID is string
    for item in items:
        item['id'] = str(item['id'])

    for start_idx in range(0, len(items), BATCH_SIZE):
        batch = items[start_idx : start_idx + BATCH_SIZE]
        try:
            vs.add_items(batch)
            print(f"  Qdrant: Ingested batch {start_idx} to {start_idx + len(batch)}")
        except Exception as e:
            print(f"  Qdrant Error in batch {start_idx}: {e}")

    print("Qdrant ingestion complete.")


def main():
    # 1. Load Data
    data = load_data(DATA_FILE)
    if not data:
        return

    # 2. Init Components
    print("Initializing Database...")
    db = Database()
    
    print("Initializing Vector Store (Qdrant)...")
    vs = VectorStore(collection_name="novels")

    # 3. Ingest
    ingest_to_sqlite(db, data)
    ingest_to_qdrant(vs, data)

    print("✅ Ingestion Pipeline Finished Successfully!")


if __name__ == "__main__":
    main()
