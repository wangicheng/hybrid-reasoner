"""
Book Mention Parser - 用戶輸入書籍提及分析模組

功能：
1. 從用戶輸入中提取書籍名稱和情感傾向（喜歡/不喜歡）
2. 將提取的書名匹配到資料庫中的實際書籍
3. 根據匹配到的書籍資訊增強搜尋查詢
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from google import genai
from google.genai import types
from src.core.api_utils import retry_on_rate_limit


@dataclass
class BookMention:
    """書籍提及資料結構"""
    title: str                          # 用戶提到的書名
    sentiment: str                      # 'liked', 'disliked', 'neutral'
    matched_book: Optional[Dict] = None # 匹配到的資料庫書籍記錄
    match_confidence: float = 0.0       # 匹配信心度 (0-1)


@dataclass
class ParsedMentions:
    """解析結果"""
    liked_books: List[BookMention] = field(default_factory=list)
    disliked_books: List[BookMention] = field(default_factory=list)
    neutral_books: List[BookMention] = field(default_factory=list)
    
    @property
    def all_mentions(self) -> List[BookMention]:
        return self.liked_books + self.disliked_books + self.neutral_books
    
    @property
    def has_mentions(self) -> bool:
        return len(self.all_mentions) > 0


class BookMentionExtractor:
    """
    使用 LLM 從用戶輸入中提取書籍提及和情感傾向。
    
    範例：
        輸入: "我喜歡《刀劍神域》，但不太喜歡回復術士那種"
        輸出: {
            "liked_books": ["刀劍神域"],
            "disliked_books": ["回復術士"],
            "neutral_books": []
        }
    """
    
    def __init__(self, model_id: Optional[str] = None):
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "gemini-2.5-flash-lite")
        
    @retry_on_rate_limit(max_retries=3, base_delay=1.0)
    def extract(self, user_query: str) -> Dict[str, List[str]]:
        """
        從用戶查詢中提取書籍名稱和情感。
        
        Returns:
            {
                "liked_books": List[str],      # 喜歡的書名
                "disliked_books": List[str],   # 不喜歡的書名  
                "neutral_books": List[str]     # 中性提及的書名
            }
        """
        schema = {
            "type": "object",
            "properties": {
                "liked_books": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用戶明確表示喜歡、推薦、想找類似的書籍名稱"
                },
                "disliked_books": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用戶明確表示不喜歡、不想要、想避免的書籍名稱"
                },
                "neutral_books": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "用戶提到但沒有明確表達喜好的書籍名稱"
                }
            },
            "required": ["liked_books", "disliked_books", "neutral_books"]
        }
        
        system_prompt = """你是一個專業的書籍名稱提取助手。你的任務是從用戶的查詢中識別出提到的書籍名稱，並判斷用戶對這些書籍的態度。

規則：
1. 提取所有書籍名稱，包括輕小說、網路小說、漫畫等
2. 書名可能用《》、「」包裹，也可能直接提及
3. 書名可能是簡稱或暱稱（如"刀劍"指"刀劍神域"）
4. 根據上下文判斷情感：
   - liked_books: 喜歡、愛、推薦、想找類似的、想看像...這樣的
   - disliked_books: 不喜歡、討厭、想避免、不要像...那種
   - neutral_books: 單純提到、沒有明確態度
5. 如果用戶沒有提到任何書籍，返回空陣列
6. 只提取書名，不要包含作者名或其他資訊

範例：
- "我喜歡無職轉生，推薦類似的" → liked_books: ["無職轉生"]
- "不要像回復術士那種" → disliked_books: ["回復術士"]
- "刀劍神域很棒，但不太喜歡SAO progressive" → liked_books: ["刀劍神域"], disliked_books: ["SAO progressive"]
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=user_query,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1  # 低溫度確保一致性
                )
            )
            
            result = json.loads(response.text)
            
            # 確保所有欄位存在
            return {
                "liked_books": result.get("liked_books", []),
                "disliked_books": result.get("disliked_books", []),
                "neutral_books": result.get("neutral_books", [])
            }
            
        except Exception as e:
            print(f"[BookMentionExtractor] Error: {e}")
            return {"liked_books": [], "disliked_books": [], "neutral_books": []}


class BookMatcher:
    """
    將提取的書名匹配到資料庫中的實際書籍記錄。
    結合模糊搜尋和向量搜尋，取最佳結果。
    """
    
    def __init__(self, db=None, vs=None):
        from src.core.database import Database
        from src.core.vector_store import VectorStore
        
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        
    def match_single(self, book_title: str, top_k: int = 3) -> Tuple[Optional[Dict], float]:
        """
        匹配單一書名到資料庫記錄。
        
        Returns:
            (matched_book, confidence): 最佳匹配和信心度
        """
        candidates = []
        
        # 策略 1: 模糊搜尋 (精確匹配)
        fuzzy_results = self.db.search_by_title_fuzzy(book_title)
        for book in fuzzy_results:
            # 計算相似度：完全匹配給高分，部分匹配降分
            name = book.get("name", "")
            if name == book_title:
                confidence = 1.0
            elif book_title in name or name in book_title:
                confidence = 0.85
            else:
                confidence = 0.6
            candidates.append((book, confidence, "fuzzy"))
            
        # 策略 2: 向量搜尋 (語意匹配)
        try:
            vector_results, _ = self.vs.search(
                query_text=f"書名：{book_title}",
                limit=top_k
            )
            for result in vector_results:
                payload = result.get("payload", {})
                score = result.get("score", 0)
                # 將向量分數轉換為信心度 (cosine similarity 0-1)
                confidence = min(score, 1.0) * 0.9  # 稍微降低向量結果的權重
                
                # 檢查是否已在候選中
                book_id = payload.get("id")
                if not any(c[0].get("id") == book_id for c in candidates):
                    candidates.append((payload, confidence, "vector"))
                else:
                    # 更新現有候選的分數（取較高者）
                    for i, (book, conf, src) in enumerate(candidates):
                        if book.get("id") == book_id:
                            candidates[i] = (book, max(conf, confidence), "hybrid")
                            break
        except Exception as e:
            print(f"[BookMatcher] Vector search error: {e}")
        
        if not candidates:
            return None, 0.0
            
        # 按信心度排序，取最佳
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_match, best_confidence, source = candidates[0]
        
        # 額外驗證：書名相似度檢查
        matched_name = best_match.get("name", "")
        if self._name_similarity(book_title, matched_name) < 0.3:
            # 名字太不相似，降低信心度
            best_confidence *= 0.5
            
        return best_match, best_confidence
    
    def match_all(self, mentions: Dict[str, List[str]]) -> ParsedMentions:
        """
        批量匹配所有提及的書籍。
        
        Args:
            mentions: BookMentionExtractor.extract() 的輸出
            
        Returns:
            ParsedMentions 結果
        """
        result = ParsedMentions()
        
        for title in mentions.get("liked_books", []):
            matched, confidence = self.match_single(title)
            mention = BookMention(
                title=title,
                sentiment="liked",
                matched_book=matched,
                match_confidence=confidence
            )
            result.liked_books.append(mention)
            
        for title in mentions.get("disliked_books", []):
            matched, confidence = self.match_single(title)
            mention = BookMention(
                title=title,
                sentiment="disliked", 
                matched_book=matched,
                match_confidence=confidence
            )
            result.disliked_books.append(mention)
            
        for title in mentions.get("neutral_books", []):
            matched, confidence = self.match_single(title)
            mention = BookMention(
                title=title,
                sentiment="neutral",
                matched_book=matched,
                match_confidence=confidence
            )
            result.neutral_books.append(mention)
            
        return result
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """簡單的名稱相似度計算（基於字元重疊）"""
        if not name1 or not name2:
            return 0.0
        set1 = set(name1)
        set2 = set(name2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class QueryEnhancer:
    """
    根據匹配到的書籍資訊增強搜尋查詢。
    
    - 喜歡的書: 提取 tags, classification 作為正向關鍵詞
    - 不喜歡的書: 提取 tags 作為負面特徵，用於降權
    """
    
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold
        
    def enhance(self, parsed_mentions: ParsedMentions) -> Dict[str, Any]:
        """
        根據書籍提及增強查詢。
        
        Returns:
            {
                "positive_tags": List[str],        # 正向標籤（來自喜歡的書）
                "positive_classifications": List[str],
                "negative_tags": List[str],        # 負面標籤（來自不喜歡的書）
                "positive_keywords": List[str],    # 從 intro 提取的關鍵詞
                "reference_intros": List[str],     # 參考書籍的簡介
                "enhancement_metadata": Dict       # 增強元資料（用於解釋）
            }
        """
        positive_tags = []
        positive_classifications = []
        negative_tags = []
        positive_keywords = []
        reference_intros = []
        
        metadata = {
            "liked_matches": [],
            "disliked_matches": [],
            "applied_enhancements": []
        }
        
        # 處理喜歡的書
        for mention in parsed_mentions.liked_books:
            if mention.matched_book and mention.match_confidence >= self.confidence_threshold:
                book = mention.matched_book
                
                # 收集標籤
                tags = book.get("tags", [])
                if isinstance(tags, list):
                    positive_tags.extend(tags)
                    
                # 收集分類
                classification = book.get("classification")
                if classification:
                    positive_classifications.append(classification)
                    
                # 收集簡介
                intro = book.get("intro", "")
                if intro:
                    reference_intros.append(intro[:200])  # 只取前200字
                    
                metadata["liked_matches"].append({
                    "user_input": mention.title,
                    "matched_name": book.get("name"),
                    "confidence": mention.match_confidence
                })
                
        # 處理不喜歡的書 - 提取負面特徵
        for mention in parsed_mentions.disliked_books:
            if mention.matched_book and mention.match_confidence >= self.confidence_threshold:
                book = mention.matched_book
                
                # 收集負面標籤
                tags = book.get("tags", [])
                if isinstance(tags, list):
                    negative_tags.extend(tags)
                    
                metadata["disliked_matches"].append({
                    "user_input": mention.title,
                    "matched_name": book.get("name"),
                    "confidence": mention.match_confidence,
                    "negative_tags": tags
                })
        
        # 去重
        positive_tags = list(dict.fromkeys(positive_tags))
        positive_classifications = list(dict.fromkeys(positive_classifications))
        negative_tags = list(dict.fromkeys(negative_tags))
        
        # 移除正負衝突的標籤（正向優先）
        negative_tags = [t for t in negative_tags if t not in positive_tags]
        
        # 記錄應用的增強
        if positive_tags:
            metadata["applied_enhancements"].append(f"Added {len(positive_tags)} positive tags")
        if negative_tags:
            metadata["applied_enhancements"].append(f"Added {len(negative_tags)} negative tags for filtering")
        if reference_intros:
            metadata["applied_enhancements"].append(f"Using {len(reference_intros)} reference intros")
            
        return {
            "positive_tags": positive_tags,
            "positive_classifications": positive_classifications,
            "negative_tags": negative_tags,
            "positive_keywords": positive_keywords,
            "reference_intros": reference_intros,
            "enhancement_metadata": metadata
        }


class BookMentionParser:
    """
    整合入口：完整的書籍提及解析流程。
    
    Usage:
        parser = BookMentionParser()
        result = parser.parse("我喜歡《刀劍神域》，推薦類似的")
        
        # result.parsed_mentions - 解析的書籍提及
        # result.enhancement - 查詢增強資訊
    """
    
    def __init__(self, db=None, vs=None, model_id: Optional[str] = None):
        self.extractor = BookMentionExtractor(model_id=model_id)
        self.matcher = BookMatcher(db=db, vs=vs)
        self.enhancer = QueryEnhancer()
        
    def parse(self, user_query: str) -> Dict[str, Any]:
        """
        完整解析流程：提取 → 匹配 → 增強
        
        Returns:
            {
                "has_book_mentions": bool,
                "parsed_mentions": ParsedMentions,
                "enhancement": Dict,
                "original_query": str
            }
        """
        # Step 1: 提取書籍名稱和情感
        raw_mentions = self.extractor.extract(user_query)
        
        # 快速檢查：如果沒有提到任何書籍，直接返回
        if not any(raw_mentions.values()):
            return {
                "has_book_mentions": False,
                "parsed_mentions": ParsedMentions(),
                "enhancement": {},
                "original_query": user_query
            }
        
        # Step 2: 匹配到資料庫記錄
        parsed_mentions = self.matcher.match_all(raw_mentions)
        
        # Step 3: 生成查詢增強
        enhancement = self.enhancer.enhance(parsed_mentions)
        
        return {
            "has_book_mentions": parsed_mentions.has_mentions,
            "parsed_mentions": parsed_mentions,
            "enhancement": enhancement,
            "original_query": user_query
        }
