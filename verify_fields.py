import sqlite3
import json
import sys

def verify_db():
    conn = sqlite3.connect("c:/dev/hybrid-reasoner/hybrid_reasoner.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check schema
    cursor.execute("PRAGMA table_info(novels)")
    columns = [row["name"] for row in cursor.fetchall()]
    print(f"Columns: {columns}")
    
    required = ["author", "slogan", "is_free", "tts", "restricted_age"]
    missing = [c for c in required if c not in columns]
    
    if missing:
        print(f"FAIL: Missing columns: {missing}")
        sys.exit(1)
        
    # Check data content
    cursor.execute("SELECT name, author, is_free, tts FROM novels LIMIT 5")
    rows = cursor.fetchall()
    print("\nSample Data:")
    for row in rows:
        print(dict(row))
        
    # Check specific author if possible
    cursor.execute("SELECT count(*) as count FROM novels WHERE author LIKE '%蔡芳紜%'")
    count = cursor.fetchone()["count"]
    print(f"\nBooks by 蔡芳紜: {count}")

    conn.close()

if __name__ == "__main__":
    verify_db()
