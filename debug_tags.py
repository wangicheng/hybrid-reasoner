import requests
from bs4 import BeautifulSoup
import re

URL = "https://tw.linovelib.com/novel/3095.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://tw.linovelib.com/"
}

def test_fetch_tags():
    print(f"Fetching {URL}...")
    resp = requests.get(URL, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Failed to fetch page. Status: {resp.status_code}")
        return

    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Logic from crawler_linovelib.py (slightly simplified for testing core logic)
    tags = []
    
    # 1. From tag links (href="/tagarticle/...")
    # Based on fetch_webpage output, the tags are links like [校園](https://tw.linovelib.com/tagarticle/63/1.html)
    # They seem to appear in a specific section.
    
    # Let's find all links containing "tagarticle"
    tag_links = soup.find_all('a', href=re.compile(r'/tagarticle/\d+/\d+\.html'))
    
    print("\n--- Found Tag Links ---")
    for link in tag_links:
        t_text = link.get_text().strip()
        print(f"Found tag: {t_text}")
        if t_text and t_text not in tags:
            tags.append(t_text)
            
    print("\n--- Final Extracted Tags ---")
    print(tags)
    
    required_tags = ["校園", "歡樂向", "青春"]
    missing = [t for t in required_tags if t not in tags]
    
    if not missing:
        print("\n✅ SUCCESS: All required tags found!")
    else:
        print(f"\n❌ FAILURE: Missing tags: {missing}")

if __name__ == "__main__":
    test_fetch_tags()
