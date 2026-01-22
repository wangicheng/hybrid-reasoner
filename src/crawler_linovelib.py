import requests
from bs4 import BeautifulSoup
import time
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

class LinovelibCrawler:
    """
    Crawler for tw.linovelib.com
    Note: This is a standalone script to fetch data and save it to data/books_linovelib.json
    """
    BASE_URL = "https://tw.linovelib.com"
    
    def __init__(self, output_file: str = "books_linovelib.json"):
        # Mimic a real browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://tw.linovelib.com/"
        }
        self.output_dir = Path(__file__).parent.parent / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / output_file

    def crawl(self, pages: int = 1):
        """
        Crawls the ranking pages to find books.
        """
        books = []
        seen_urls = set()
        
        # Crawl Ranking (Month Visit)
        # https://tw.linovelib.com/top/monthvisit/1.html
        print(f"🚀 Starting crawl from {self.BASE_URL}...")
        
        for page in range(1, pages + 1):
            rank_url = f"{self.BASE_URL}/top/monthvisit/{page}.html"
            print(f"📄 Fetching ranking page {page}: {rank_url}")
            
            try:
                resp = requests.get(rank_url, headers=self.headers)
                if resp.status_code != 200:
                    print(f"⚠️ Failed to fetch page {page}: Status {resp.status_code}")
                    continue
                
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Extract links to novel details
                # Pattern: <a href="/novel/2570.html" class="book-layout"> ... </a>
                # Or just scanning all links
                links = soup.find_all('a', href=True)
                
                page_books_found = 0
                for a in links:
                    href = a['href']
                    # Match /novel/1234.html
                    if re.match(r'^/novel/\d+\.html$', href):
                        full_url = f"{self.BASE_URL}{href}"
                        if full_url not in seen_urls:
                            seen_urls.add(full_url)
                            
                            # Parse Book Details
                            book_info = self.parse_book_details(full_url)
                            if book_info:
                                books.append(book_info)
                                page_books_found += 1
                                print(f"  ✅ Parsed: {book_info['name']} ({book_info['publish_status']})")
                            
                            # Be polite
                            time.sleep(random.uniform(0.5, 1.5))
                            
                            # Limit for demo purposes (don't crawl thousands in one go unless asked)
                            # if len(books) >= 20: 
                            #    break
                
                # if len(books) >= 20:
                #    print("🛑 Reached demo limit of 20 books.")
                #    break
                    
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")

        # Save to JSON
        print(f"💾 Saving {len(books)} books to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(books, f, ensure_ascii=False, indent=2)
        print("🎉 Done!")

    def parse_book_details(self, url: str) -> Dict[str, Any]:
        """
        Parses a single book page.
        """
        try:
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 1. Title
            # <meta property="og:title" content="惡魔高校DxD" />
            og_title = soup.find('meta', property='og:title')
            name = og_title['content'] if og_title else "Unknown"
            
            # 2. Author
            # <meta property="og:novel:author" content="石踏一榮" />
            og_author = soup.find('meta', property='og:novel:author')
            author = og_author['content'] if og_author else "Unknown"
            
            # 3. Description (Intro)
            # <meta property="og:description" content="..." />
            og_desc = soup.find('meta', property='og:description')
            intro = og_desc['content'] if og_desc else ""
            # Clean up introductory text (sometimes it has "share.linovelib.net...")
            intro = re.sub(r'share\.linovelib\.net.*', '', intro).strip()
            
            # 4. Cover Image
            og_image = soup.find('meta', property='og:image')
            cover_url = og_image['content'] if og_image else None
            
            # 5. Status & Metadata
            # <meta property="og:novel:status" content="完結" />
            og_status = soup.find('meta', property='og:novel:status')
            raw_status = og_status['content'] if og_status else "連載中"
            
            # Map to our standard
            publish_status = "已完結" if "完結" in raw_status else "連載中"
            
            # 6. Classification / Tags
            # <meta property="og:novel:category" content="校園" />
            og_category = soup.find('meta', property='og:novel:category')
            classification = og_category['content'] if og_category else "其他"
            
            # --- Enhanced Tag Parsing ---
            tags = [classification]
            # Add site-wide implicit tags for better searchability
            tags.append("二次元")
            tags.append("輕小說")
            
            # Parsing "作品標籤" section
            # Looking for links commonly found in the tag area
            # Based on structure, they might be in a div with specific class or just after generic text.
            # Let's look for known tag patterns in the soup
            # Example: <a href="/tagarticle/...">TagName</a>
            tag_links = soup.find_all('a', href=re.compile(r'/tagarticle/\d+/\d+\.html'))
            for link in tag_links:
                t_text = link.get_text().strip()
                if t_text and t_text not in tags:
                    tags.append(t_text)
            
            # Also check if "異世界" is in the title or intro, and explicitly add it if missing
            # This helps vector search and potential hard filters
            if "異世界" in name or "異世界" in intro:
                if "異世界" not in tags:
                    tags.append("異世界")
            
            # 7. Word Count & Click Count (Harder, needs regex on body text)
            # Text Example: "398.1 萬字"
            body_text = soup.get_text()
            
            words_total = 0
            word_match = re.search(r'([\d\.]+)\s*萬字', body_text)
            if word_match:
                try:
                    val = float(word_match.group(1))
                    words_total = int(val * 10000)
                except:
                    pass
            
            click_count = 0
            # Look for something like "28598 人收藏" or heat
            # Regex: (\d+)\s*人收藏
            collect_match = re.search(r'(\d+)\s*人收藏', body_text)
            if collect_match:
                 click_count = int(collect_match.group(1)) # Use collection count as proxy for clicks

            # Generate ID
            book_id = re.search(r'/novel/(\d+)\.html', url).group(1)
            
            return {
                "id": f"linovelib_{book_id}",
                "name": name,
                "source": "linovelib",
                "author": author,
                "classification": classification,
                "tags": tags,
                "intro": intro,
                "click_count": click_count, # Using Collection count as proxy
                "words_total": words_total,
                "publish_status": publish_status,
                "url": url,
                "cover_url": cover_url
            }
            
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return None

if __name__ == "__main__":
    crawler = LinovelibCrawler()
    # Crawl 5 pages (approx. 100 books) for a larger dataset
    crawler.crawl(pages=5)
