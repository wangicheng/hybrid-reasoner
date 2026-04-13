import sqlite3
import json
from src.config import settings

def count_overlap():
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, tags FROM novels')
    rows = cursor.fetchall()
    
    count = 0
    matches = []
    
    for name, tag_json in rows:
        try:
            tags = json.loads(tag_json) if tag_json else []
            if '異世界' in tags and '戰鬥' in tags:
                count += 1
                matches.append(name)
        except:
            continue
            
    print(f"同時擁有「異世界」且「戰鬥」標籤的書籍總數: {count}")
    if count > 0:
        print("前 5 本範例:")
        for m in matches[:5]:
            print(f"- {m}")
    conn.close()

if __name__ == "__main__":
    count_overlap()
