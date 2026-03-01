import requests
import time
import json
from typing import List, Dict, Any
from pathlib import Path

from src.core.text_utils import clean_html

class MirrorFictionCrawler:
    BASE_URL = "https://www.mirrorfiction.com/api/books/index"
    DETAILS_URL = "https://www.mirrorfiction.com/api/books"

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def fetch_book_list(self, page: int = 1) -> List[Dict[str, Any]]:
        """Fetches a page of books from the index API."""
        params = {
            "lang": "zh-Hant",
            "type": 1, # Potentially "Novel" type
            "orderBy": "click",
            "sortedBy": "desc",
            "r": 1,
            "page": page
        }
        print(f"Fetching page {page}...")
        try:
            resp = requests.get(self.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("result") == "success":
                return data.get("data", [])
            else:
                print(f"API Error on page {page}: {data}")
                return []
        except Exception as e:
            print(f"Request failed for page {page}: {e}")
            return []

    def fetch_book_details(self, book_id: int) -> Dict[str, Any]:
        """Fetches detailed info for a specific book."""
        url = f"{self.DETAILS_URL}/{book_id}"
        params = {
            "lang": "zh-Hant",
            "include": "classification,attribute.type,statistic,tags,user.userStatistic,chapters:limit(1|1):order(order_column|asc),news:limit(2|1):order(updated_at|desc),personalized"
        }
        # Be nice to the server
        time.sleep(0.5) 
        
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("result") == "success":
                book_data = data.get("data", {})
                # Clean HTML from text fields
                if "intro" in book_data:
                    book_data["intro"] = clean_html(book_data["intro"])
                if "slogan" in book_data:
                    book_data["slogan"] = clean_html(book_data["slogan"])
                return book_data
            else:
                print(f"API Error for book {book_id}: {data}")
                return {}
        except Exception as e:
            print(f"Request failed for book {book_id}: {e}")
            return {}

    def crawl(self, pages: int = 2):
        """Crawls top N pages of books and saves them."""
        all_books = []
        
        for p in range(1, pages + 1):
            books_summary = self.fetch_book_list(page=p)
            for summary in books_summary:
                book_id = summary.get("id")
                if book_id:
                    details = self.fetch_book_details(book_id)
                    if details:
                        all_books.append(details)
                        print(f"fetched: {details.get('name')}")
            
        # Save to file
        output_file = self.output_dir / "books_crawled.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_books, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_books)} books to {output_file}")

if __name__ == "__main__":
    crawler = MirrorFictionCrawler()
    crawler.crawl(pages=100)
