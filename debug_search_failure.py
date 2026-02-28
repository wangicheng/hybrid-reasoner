import sys
import os
import asyncio
import json
import sqlite3
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.database import Database
from src.core.llm import parse_query

async def diagnose_search_issue(query_text):
    print(f"🔍 Diagnosing Query: '{query_text}'")
    
    # 1. Check Database directly
    db = Database()
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # List some books to verify DB content
    print("\n[Database Check]")
    cursor.execute("SELECT name, author FROM novels LIMIT 5")
    rows = cursor.fetchall()
    print("Sample books in DB:")
    for r in rows:
        print(f" - {r}")
        
    # Try exact match query in DB
    print(f"\nChecking if '{query_text}' exists in DB (LIKE %...%):")
    cursor.execute("SELECT id, name FROM novels WHERE name LIKE ?", (f"%{query_text}%",))
    matches = cursor.fetchall()
    if matches:
        print(f"✅ FOUND in DB: {matches}")
    else:
        print(f"❌ NOT FOUND in DB. Trying broader search...")
        # Try finding ANY book just to be sure
        cursor.execute("SELECT count(*) FROM novels")
        count = cursor.fetchone()[0]
        print(f"Total books in DB: {count}")

    conn.close()

    # 2. Check LLM Parsing
    print("\n[LLM Parsing Check]")
    try:
        # Force a specific model if needed, or rely on default
        result = parse_query(query_text)
        print("LLM Parse Result:")
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        
        # Check search terms
        print(f"\nSearch Terms extracted: {result.search_terms}")
        
        # Simulate Engine Logic
        print("\n[Engine Logic Simulation]")
        found_by_term = False
        for term in result.search_terms:
            print(f"Checking term: '{term}'")
            db_matches = db.search_by_title_fuzzy(term)
            if db_matches:
                print(f"  -> Match found in DB for term '{term}': {[b['name'] for b in db_matches]}")
                found_by_term = True
            else:
                print(f"  -> No match in DB for term '{term}'")
        
        if not found_by_term:
            print("⚠️ Engine logic would FAIL to find this book using search_terms.")
            
            # Check if original_query can find it
            print(f"Checking original query: '{result.original_query}'")
            db_matches = db.search_by_title_fuzzy(result.original_query)
            if db_matches:
                 print(f"  -> Match found using original_query! We should probably use that too.")
            else:
                 print(f"  -> Even original query didn't find it directly.")

    except Exception as e:
        print(f"LLM Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = sys.argv[1]
    else:
        q = "詭秘之主"
    asyncio.run(diagnose_search_issue(q))
