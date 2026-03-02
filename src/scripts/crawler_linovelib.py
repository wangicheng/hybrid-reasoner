"""
Bilinovel (tw.linovelib.com) 爬蟲 v2
從收藏榜爬取全部小說資料
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


class LinovelibCrawler:
    """Crawler for tw.linovelib.com"""
    
    BASE_URL = "https://tw.linovelib.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
    }
    
    def __init__(self, delay: float = 1.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def _random_delay(self):
        """隨機延遲 (delay ± 50%)"""
        jitter = self.delay * random.uniform(0.5, 1.5)
        time.sleep(jitter)
        
    def get_page(self, url: str, retries: int = 5) -> Optional[BeautifulSoup]:
        """獲取頁面並解析，包含 429 指數退避"""
        backoff_times = [5, 10, 20, 40, 60]  # 指數退避等待時間
        
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    return BeautifulSoup(resp.text, 'html.parser')
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 429:
                    # Rate limited - 指數退避
                    wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                    print(f"  Rate limited (429), 等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                    continue
                elif resp.status_code == 403:
                    # Forbidden - 可能是地區限制或暫時封鎖
                    wait_time = backoff_times[min(attempt, len(backoff_times) - 1)]
                    print(f"  Forbidden (403), 等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Warning: {url} returned {resp.status_code}")
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(backoff_times[min(attempt, len(backoff_times) - 1)])
        return None
    
    def get_novel_ids_from_wenku(self, max_pages: int = 200) -> List[str]:
        """從收藏榜獲取所有小說 ID"""
        novel_ids = []
        seen_ids = set()
        page = 1
        
        print(f"正在從收藏榜爬取小說列表...")
        
        while page <= max_pages:
            url = f"{self.BASE_URL}/wenku/goodnum_0_0_0_0_0_0_0_{page}_0.html"
            soup = self.get_page(url)
            
            if not soup:
                print(f"頁面 {page} 無法訪問，停止爬取")
                break
            
            # 找所有小說連結
            found_new = False
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                match = re.search(r'/novel/(\d+)\.html', href)
                if match:
                    novel_id = match.group(1)
                    if novel_id not in seen_ids:
                        seen_ids.add(novel_id)
                        novel_ids.append(novel_id)
                        found_new = True
            
            if not found_new:
                print(f"頁面 {page} 沒有新小說，停止爬取")
                break
                
            if page % 10 == 0:
                print(f"  頁面 {page}: 已收集 {len(novel_ids)} 本小說")
            page += 1
            self._random_delay()
        
        print(f"  完成: 共收集 {len(novel_ids)} 本小說")
        return novel_ids
    
    def parse_novel_detail(self, novel_id: str) -> Optional[Dict[str, Any]]:
        """解析單本小說詳細資訊"""
        url = f"{self.BASE_URL}/novel/{novel_id}.html"
        soup = self.get_page(url)
        
        if not soup:
            return None
        
        try:
            novel = {
                "id": f"linovelib_{novel_id}",
                "source": "linovelib",
                "url": url,
            }
            
            # === 標題 ===
            title_meta = soup.find('meta', {'property': 'og:novel:book_name'})
            if title_meta:
                novel["name"] = title_meta.get('content', '').strip()
            else:
                title_elem = soup.select_one('h1.book-title')
                novel["name"] = title_elem.get_text(strip=True) if title_elem else ""
            
            # === 作者 ===
            author_meta = soup.find('meta', {'property': 'og:novel:author'})
            if author_meta:
                novel["author"] = author_meta.get('content', '').strip()
            else:
                author_elem = soup.select_one('.authorname a')
                novel["author"] = author_elem.get_text(strip=True) if author_elem else ""
            
            # === 插畫家 (illname) ===
            illname_elem = soup.select_one('.illname a')
            if illname_elem:
                # 移除 ruby 標籤的 rt 部分
                rt = illname_elem.find('rt')
                if rt:
                    rt.decompose()
                novel["illname"] = illname_elem.get_text(strip=True)
            else:
                novel["illname"] = None
            
            # === 分類/文庫 ===
            cat_meta = soup.find('meta', {'property': 'og:novel:category'})
            novel["classification"] = cat_meta.get('content', '').strip() if cat_meta else ""
            
            # === 簡介 ===
            intro_elem = soup.select_one('#bookSummary content')
            novel["intro"] = intro_elem.get_text(strip=True) if intro_elem else ""
            
            # === 別名 (backupname) ===
            backup_elem = soup.select_one('.backupname .bkname-body')
            novel["backupname"] = backup_elem.get_text(strip=True) if backup_elem else None
            
            # === 標籤 (tags) ===
            tags = []
            for tag_elem in soup.select('.tag-small-group .tag-small a'):
                tag_text = tag_elem.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)
            novel["tags"] = tags
            
            # === 統計資料 (從 .book-meta 解析) ===
            meta_text = ""
            for meta_elem in soup.select('.book-meta.book-layout-inline'):
                meta_text += meta_elem.get_text()
            
            # 字數
            words_match = re.search(r'([\d.]+)\s*萬?\s*字', meta_text)
            if words_match:
                words = float(words_match.group(1))
                if '萬' in meta_text[:meta_text.find('字') + 5]:
                    words *= 10000
                novel["words_total"] = int(words)
            else:
                novel["words_total"] = 0
            
            # 收藏數
            bookmark_match = re.search(r'(\d+)\s*人收藏', meta_text)
            novel["bookmark_count"] = int(bookmark_match.group(1)) if bookmark_match else 0
            
            # 推薦數
            rec_match = re.search(r'(\d+)\s*次推薦', meta_text)
            novel["total_recommendations"] = int(rec_match.group(1)) if rec_match else 0
            
            # 連載狀態
            status_meta = soup.find('meta', {'property': 'og:novel:status'})
            novel["publish_status"] = status_meta.get('content', '').strip() if status_meta else "未知"
            
            # 已動畫化狀態
            novel["is_animated"] = "已動畫化" in meta_text
            
            # === 評分 ===
            score_elem = soup.select_one('.score-num')
            if score_elem:
                try:
                    novel["rating_score"] = float(score_elem.get_text(strip=True))
                except ValueError:
                    novel["rating_score"] = None
            else:
                novel["rating_score"] = None
            
            # 評價人數
            rating_elem = soup.select_one('.sub-rating')
            if rating_elem:
                text = rating_elem.get_text()
                count_match = re.search(r'(\d+)\s*人', text)
                novel["rating_count"] = int(count_match.group(1)) if count_match else 0
            else:
                novel["rating_count"] = 0
            
            # === 封面圖片 ===
            cover_meta = soup.find('meta', {'property': 'og:image'})
            novel["cover_url"] = cover_meta.get('content', '') if cover_meta else None
            
            return novel
            
        except Exception as e:
            print(f"Error parsing novel {novel_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def crawl(self, max_pages: int = 200,
              max_books: int = 0,
              output_file: str = "data/books_linovelib.json") -> List[Dict[str, Any]]:
        """
        主爬取函數
        
        Args:
            max_pages: 最大頁數
            max_books: 最大書籍數 (0 = 無限制)
            output_file: 輸出文件路徑
        """
        # 獲取小說 ID 列表
        novel_ids = self.get_novel_ids_from_wenku(max_pages)
        
        # 限制書籍數量
        if max_books > 0 and len(novel_ids) > max_books:
            novel_ids = novel_ids[:max_books]
            
        print(f"\n共獲取 {len(novel_ids)} 個小說 ID")
        
        if not novel_ids:
            print("沒有獲取到任何小說 ID")
            return []
        
        # 爬取詳細資料
        novels = []
        failed = []
        
        print(f"\n開始爬取小說詳細資料...")
        
        for i, novel_id in enumerate(novel_ids, 1):
            novel = self.parse_novel_detail(novel_id)
            
            if novel:
                novels.append(novel)
                if i % 50 == 0:
                    print(f"  已爬取: {i}/{len(novel_ids)} ({len(novels)} 成功)")
                    # 中間保存
                    self._save_json(novels, output_file)
            else:
                failed.append(novel_id)
            
            self._random_delay()
        
        # 最終保存
        self._save_json(novels, output_file)
        
        print(f"\n爬取完成!")
        print(f"  成功: {len(novels)}")
        print(f"  失敗: {len(failed)}")
        if failed:
            print(f"  失敗 ID: {failed[:20]}{'...' if len(failed) > 20 else ''}")
        
        return novels
    
    def _save_json(self, data: List[Dict], filepath: str):
        """保存 JSON 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Linovelib Crawler')
    parser.add_argument('--max-pages', type=int, default=200,
                        help='Maximum number of pages to crawl')
    parser.add_argument('--max-books', type=int, default=0,
                        help='Maximum number of books to crawl (0 = unlimited)')
    parser.add_argument('--output', type=str, default='data/books_linovelib.json',
                        help='Output file path')
    parser.add_argument('--delay', type=float, default=1.5,
                        help='Base delay between requests in seconds (actual delay = delay ± 50%%)')
    
    args = parser.parse_args()
    
    crawler = LinovelibCrawler(delay=args.delay)
    crawler.crawl(
        max_pages=args.max_pages,
        max_books=args.max_books,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
