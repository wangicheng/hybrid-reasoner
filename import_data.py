#!/usr/bin/env python3
"""
Data Import Script - Load books_crawled.json into database
"""

import json
import sys
from pathlib import Path

# Ensure src can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.database import Database


def main():
    print("=" * 70)
    print("Data Import - Load books_crawled.json")
    print("=" * 70)
    
    # Load crawled data
    data_file = Path("data") / "books_crawled.json"
    
    if not data_file.exists():
        print(f"\n[ERROR] Data file not found: {data_file}")
        print(f"  Current directory: {Path.cwd()}")
        return 1
    
    print(f"\n[1/3] Reading data from {data_file.name}...")
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {e}")
        return 1
    
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
    elif isinstance(data, list):
        items = data
    else:
        print("[ERROR] Unexpected data format")
        return 1
    
    print(f"      Found {len(items)} books to import")
    
    # Insert into database
    print(f"\n[2/3] Inserting into database...")
    db = Database()
    
    success_count = 0
    error_count = 0
    
    for i, item in enumerate(items):
        try:
            db.add_item(item)
            success_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"      Processed {i + 1}/{len(items)}...")
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Show first 5 errors
                print(f"      [WARN] Failed to insert item {i}: {e}")
    
    print(f"      Completed: {success_count} inserted, {error_count} failed")
    
    # Verify
    print(f"\n[3/3] Verifying import...")
    verify_count = len(db.get_all_items())
    print(f"      Database now contains {verify_count} items")
    
    print("\n" + "=" * 70)
    if verify_count > 0:
        print("SUCCESS! Data import completed")
        print(f"Ready to generate search vectors for {verify_count} books")
        return 0
    else:
        print("ERROR! No data was imported")
        return 1


if __name__ == "__main__":
    exit(main())
