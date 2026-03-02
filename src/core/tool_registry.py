"""
Tool Registry - LLM 工具調用註冊表

提供類似 function calling 的架構，讓 LLM 可以決定是否調用外部工具。
目前支援的工具：
- analyze_book_mentions: 分析用戶提到的書籍，提取標籤和偏好資訊
"""

from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class ToolDefinition:
    """工具定義"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]  # 實際執行函數


@dataclass 
class ToolCall:
    """LLM 請求的工具調用"""
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """工具執行結果"""
    name: str
    success: bool
    result: Dict[str, Any]
    error: Optional[str] = None


class ToolRegistry:
    """
    工具註冊表 - 管理可供 LLM 調用的工具
    """
    _tools: Dict[str, ToolDefinition] = {}
    
    @classmethod
    def register(cls, name: str, description: str, parameters: Dict[str, Any]):
        """
        裝飾器：註冊一個工具
        
        Usage:
            @ToolRegistry.register(
                name="analyze_book_mentions",
                description="分析用戶提到的書籍名稱和偏好",
                parameters={...}
            )
            def analyze_book_mentions(args: Dict) -> Dict:
                ...
        """
        def decorator(func: Callable):
            cls._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=func
            )
            return func
        return decorator
    
    @classmethod
    def get_tool(cls, name: str) -> Optional[ToolDefinition]:
        return cls._tools.get(name)
    
    @classmethod
    def get_all_tools(cls) -> List[ToolDefinition]:
        return list(cls._tools.values())
    
    @classmethod
    def get_tools_schema(cls) -> List[Dict[str, Any]]:
        """
        生成供 LLM 使用的工具 schema 列表
        格式類似 OpenAI function calling
        """
        schemas = []
        for tool in cls._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        return schemas
    
    @classmethod
    def execute(cls, tool_call: ToolCall) -> ToolResult:
        """執行工具調用"""
        tool = cls.get_tool(tool_call.name)
        if not tool:
            return ToolResult(
                name=tool_call.name,
                success=False,
                result={},
                error=f"Tool '{tool_call.name}' not found"
            )
        
        try:
            result = tool.handler(tool_call.arguments)
            return ToolResult(
                name=tool_call.name,
                success=True,
                result=result
            )
        except Exception as e:
            return ToolResult(
                name=tool_call.name,
                success=False,
                result={},
                error=str(e)
            )
    
    @classmethod
    def execute_all(cls, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """批量執行工具調用"""
        return [cls.execute(tc) for tc in tool_calls]


# ============================================================
# 註冊工具: analyze_book_mentions
# ============================================================

@ToolRegistry.register(
    name="analyze_book_mentions",
    description="分析用戶提到的書籍名稱，查找資料庫中的匹配書籍，並提取標籤、分類等特徵。用於理解用戶的閱讀偏好。",
    parameters={
        "type": "object",
        "properties": {
            "liked_books": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用戶明確表示喜歡的書籍名稱列表"
            },
            "disliked_books": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用戶明確表示不喜歡的書籍名稱列表"
            },
            "neutral_books": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用戶提到但未表達明確偏好的書籍名稱列表"
            }
        },
        "required": ["liked_books", "disliked_books"]
    }
)
def analyze_book_mentions(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    執行書籍提及分析
    
    Args:
        args: {
            "liked_books": ["刀劍神域", ...],
            "disliked_books": ["回復術士", ...],
            "neutral_books": [...]
        }
    
    Returns:
        {
            "positive_tags": [...],
            "negative_tags": [...],
            "positive_classifications": [...],
            "reference_intros": [...],
            "matched_books": {...}
        }
    """
    from src.core.book_mention_parser import BookMatcher, QueryEnhancer, ParsedMentions, BookMention
    
    # 初始化匹配器和增強器
    matcher = BookMatcher()
    enhancer = QueryEnhancer(confidence_threshold=0.35)
    
    # 構建 ParsedMentions
    parsed = ParsedMentions()
    
    for title in args.get("liked_books", []):
        matched_book, confidence = matcher.match_single(title)
        parsed.liked_books.append(BookMention(
            title=title,
            sentiment="liked",
            matched_book=matched_book,
            match_confidence=confidence
        ))
    
    for title in args.get("disliked_books", []):
        matched_book, confidence = matcher.match_single(title)
        parsed.disliked_books.append(BookMention(
            title=title,
            sentiment="disliked",
            matched_book=matched_book,
            match_confidence=confidence
        ))
    
    for title in args.get("neutral_books", []):
        matched_book, confidence = matcher.match_single(title)
        parsed.neutral_books.append(BookMention(
            title=title,
            sentiment="neutral",
            matched_book=matched_book,
            match_confidence=confidence
        ))
    
    # 生成增強資訊
    enhancement = enhancer.enhance(parsed)
    
    # 構建匹配結果摘要
    matched_books = {
        "liked": [
            {"input": m.title, "matched": m.matched_book.get("name") if m.matched_book else None, "confidence": m.match_confidence}
            for m in parsed.liked_books
        ],
        "disliked": [
            {"input": m.title, "matched": m.matched_book.get("name") if m.matched_book else None, "confidence": m.match_confidence}
            for m in parsed.disliked_books
        ]
    }
    
    return {
        "positive_tags": enhancement.get("positive_tags", []),
        "negative_tags": enhancement.get("negative_tags", []),
        "positive_classifications": enhancement.get("positive_classifications", []),
        "reference_intros": enhancement.get("reference_intros", []),
        "matched_books": matched_books
    }


def format_tool_result_for_query(tool_results: List[ToolResult]) -> str:
    """
    將工具執行結果格式化為可附加到用戶查詢的文字
    
    Returns:
        格式化的增強資訊文字
    """
    lines = []
    
    for result in tool_results:
        if not result.success:
            continue
            
        if result.name == "analyze_book_mentions":
            data = result.result
            
            # 匹配結果
            matched = data.get("matched_books", {})
            liked_matches = matched.get("liked", [])
            disliked_matches = matched.get("disliked", [])
            
            if liked_matches:
                matched_names = [m["matched"] for m in liked_matches if m["matched"]]
                if matched_names:
                    lines.append(f"[用戶喜歡的書籍: {', '.join(matched_names)}]")
            
            if disliked_matches:
                matched_names = [m["matched"] for m in disliked_matches if m["matched"]]
                if matched_names:
                    lines.append(f"[用戶不喜歡的書籍: {', '.join(matched_names)}]")
            
            # 正向標籤
            positive_tags = data.get("positive_tags", [])
            if positive_tags:
                lines.append(f"[喜歡書籍的特徵標籤: {', '.join(positive_tags[:8])}]")
            
            # 負面標籤
            negative_tags = data.get("negative_tags", [])
            if negative_tags:
                lines.append(f"[不喜歡書籍的特徵標籤: {', '.join(negative_tags[:8])}]")
            
            # 分類
            classifications = data.get("positive_classifications", [])
            if classifications:
                lines.append(f"[喜歡書籍的分類: {', '.join(classifications)}]")
    
    return "\n".join(lines)
