import sqlite3
import json
from src.config import settings

def debug_db():
    print(f"Connecting to DB at {settings.DB_PATH}")
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check table info
    cursor.execute("PRAGMA table_info(novels)")
    columns = [info[1] for info in cursor.fetchall()]
    print(f"Columns: {columns}")
    
    # Check for 阿亞梅
    print("Checking for author/nickname '阿亞梅'...")
    cursor.execute("SELECT id, name, author, author_nickname FROM novels WHERE author LIKE '%阿亞梅%' OR author_nickname LIKE '%阿亞梅%'")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} matching rows:")
    for row in rows:
        print(dict(row))
        
    # Check if author_nickname is populated at all
    print("Checking sample for non-null nicknames:")
    cursor.execute("SELECT author_nickname FROM novels WHERE author_nickname IS NOT NULL LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(dict(row))

    conn.close()

if __name__ == "__main__":
    debug_db()
