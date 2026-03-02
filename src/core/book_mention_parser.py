"""
Book Mention Parser - 書籍提及解析器

分析用戶輸入中的書籍提及，匹配資料庫中的書籍，並提取特徵標籤。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from src.core.database import Database


@dataclass
class BookMention:
    """書籍提及資料結構"""
    title: str
    sentiment: str  # 'liked', 'disliked', 'neutral'
    matched_book: Optional[Dict[str, Any]] = None
    match_confidence: float = 0.0


@dataclass
class ParsedMentions:
    """解析後的書籍提及集合"""
    liked_books: List[BookMention] = field(default_factory=list)
    disliked_books: List[BookMention] = field(default_factory=list)
    neutral_books: List[BookMention] = field(default_factory=list)


class BookMatcher:
    """書籍匹配器 - 使用模糊搜尋匹配資料庫中的書籍"""
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
    
    def match_single(self, title: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        匹配單個書籍名稱
        
        Args:
            title: 用戶提到的書籍名稱
            
        Returns:
            Tuple of (matched_book, confidence)
        """
        if not title or len(title) < 2:
            return None, 0.0
        
        # 使用模糊搜尋
        results = self.db.search_by_title_fuzzy(title)[:5]  # 取前5個結果
        
        if not results:
            return None, 0.0
        
        # 計算匹配信心度
        best_match = results[0]
        best_name = best_match.get("name", "")
        
        # 簡單的相似度計算
        title_lower = title.lower().replace(" ", "")
        name_lower = best_name.lower().replace(" ", "")
        
        # 完全包含關係
        if title_lower in name_lower or name_lower in title_lower:
            # 根據長度比例計算信心度
            ratio = min(len(title_lower), len(name_lower)) / max(len(title_lower), len(name_lower))
            confidence = 0.3 + 0.7 * ratio
        else:
            # 部分匹配，較低信心度
            confidence = 0.2
        
        return best_match, min(1.0, confidence)
    
    def match_all(self, parsed: ParsedMentions) -> ParsedMentions:
        """
        匹配所有提及的書籍
        
        Args:
            parsed: 包含用戶提及書籍的 ParsedMentions
            
        Returns:
            更新了 matched_book 和 match_confidence 的 ParsedMentions
        """
        for mention in parsed.liked_books:
            if not mention.matched_book:
                mention.matched_book, mention.match_confidence = self.match_single(mention.title)
        
        for mention in parsed.disliked_books:
            if not mention.matched_book:
                mention.matched_book, mention.match_confidence = self.match_single(mention.title)
        
        for mention in parsed.neutral_books:
            if not mention.matched_book:
                mention.matched_book, mention.match_confidence = self.match_single(mention.title)
        
        return parsed


class QueryEnhancer:
    """查詢增強器 - 從匹配的書籍中提取特徵標籤"""
    
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold
    
    def enhance(self, parsed: ParsedMentions) -> Dict[str, Any]:
        """
        從匹配的書籍中提取增強資訊
        
        Args:
            parsed: 已匹配的 ParsedMentions
            
        Returns:
            {
                "positive_tags": [...],
                "negative_tags": [...],
                "positive_classifications": [...],
                "reference_intros": [...]
            }
        """
        positive_tags = []
        negative_tags = []
        positive_classifications = []
        reference_intros = []
        
        # 從喜歡的書籍提取正向特徵
        for mention in parsed.liked_books:
            if mention.matched_book and mention.match_confidence >= self.confidence_threshold:
                book = mention.matched_book
                
                # 提取標籤
                tags_str = book.get("tags", "")
                if tags_str:
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    positive_tags.extend(tags[:5])  # 限制每本書最多5個標籤
                
                # 提取分類
                classification = book.get("classification", "")
                if classification:
                    positive_classifications.append(classification)
                
                # 提取簡介
                intro = book.get("intro", "")
                if intro:
                    reference_intros.append(intro[:200])  # 限制長度
        
        # 從不喜歡的書籍提取負向特徵
        for mention in parsed.disliked_books:
            if mention.matched_book and mention.match_confidence >= self.confidence_threshold:
                book = mention.matched_book
                
                # 提取標籤
                tags_str = book.get("tags", "")
                if tags_str:
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    negative_tags.extend(tags[:5])
        
        # 去重
        positive_tags = list(dict.fromkeys(positive_tags))
        negative_tags = list(dict.fromkeys(negative_tags))
        positive_classifications = list(dict.fromkeys(positive_classifications))
        
        return {
            "positive_tags": positive_tags,
            "negative_tags": negative_tags,
            "positive_classifications": positive_classifications,
            "reference_intros": reference_intros
        }


class BookMentionParser:
    """
    書籍提及解析器 - 整合 LLM 提取和資料庫匹配
    
    這是向後相容的整合入口點。
    新的 tool calling 架構中，LLM 提取由 tool_registry.py 處理。
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.matcher = BookMatcher(self.db)
        self.enhancer = QueryEnhancer(confidence_threshold=0.35)
    
    def parse(self, user_query: str) -> Dict[str, Any]:
        """
        解析用戶查詢中的書籍提及（舊接口，向後相容）
        
        注意：新架構中，書籍提取由 LLM 在 Stage 1 完成，
        此方法主要用於直接調用場景。
        
        Args:
            user_query: 用戶的原始查詢
            
        Returns:
            {
                "parsed_mentions": ParsedMentions,
                "enhancement": {...}
            }
        """
        # 簡單的啟發式提取（不使用 LLM）
        parsed = ParsedMentions()
        
        # 使用簡單的模式匹配提取書籍名稱
        # 這是一個簡化的實現，完整的 LLM 提取在 tool calling 流程中
        import re
        
        # 匹配書名號或引號中的內容
        patterns = [
            r'《([^》]+)》',
            r'「([^」]+)」',
            r'"([^"]+)"',
            r"'([^']+)'"
        ]
        
        found_titles = []
        for pattern in patterns:
            matches = re.findall(pattern, user_query)
            found_titles.extend(matches)
        
        # 將找到的書籍都作為中性提及處理
        for title in found_titles:
            if len(title) >= 2:
                matched_book, confidence = self.matcher.match_single(title)
                parsed.neutral_books.append(BookMention(
                    title=title,
                    sentiment="neutral",
                    matched_book=matched_book,
                    match_confidence=confidence
                ))
        
        # 生成增強資訊
        enhancement = self.enhancer.enhance(parsed)
        
        return {
            "parsed_mentions": parsed,
            "enhancement": enhancement
        }
