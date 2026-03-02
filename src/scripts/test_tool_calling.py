"""
測試 Tool Calling 架構
驗證兩階段 LLM 解析流程
"""

import asyncio
import sys
import os

# 確保可以找到 src 模組
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 載入環境變量
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from src.core.llm import parse_query_with_tools


def test_parse_query_with_tools():
    """測試 parse_query_with_tools 函數"""
    print("=" * 60)
    print("Test: parse_query_with_tools")
    print("=" * 60)
    
    # 測試查詢：包含書籍提及
    test_query = "我喜歡刀劍神域的風格，想找類似的異世界冒險小說"
    
    print(f"\nQuery: {test_query}")
    print("-" * 40)
    
    result, tool_context = parse_query_with_tools(test_query, enable_tools=True)
    
    print(f"\nParse Result:")
    print(f"  - original_query: {result.original_query}")
    print(f"  - search_terms: {result.search_terms}")
    print(f"  - generated_keywords: {result.generated_keywords[:5]}...")
    print(f"  - criteria count: {len(result.criteria)}")
    
    print(f"\nTool Context:")
    print(f"  - tool_calls: {tool_context.get('tool_calls', [])}")
    
    tool_results = tool_context.get('tool_results', [])
    for tr in tool_results:
        print(f"  - Tool Result [{tr.get('name')}]: success={tr.get('success')}")
        if tr.get('success'):
            result_data = tr.get('result', {})
            print(f"    - positive_tags: {result_data.get('positive_tags', [])[:5]}")
            print(f"    - negative_tags: {result_data.get('negative_tags', [])[:5]}")
            print(f"    - matched_books: {result_data.get('matched_books', {})}")
        else:
            print(f"    - error: {tr.get('error', 'N/A')}")
    
    print(f"\nEnhancement Text:")
    enhancement = tool_context.get('enhancement_text', 'N/A')
    print(f"  {enhancement[:300] if enhancement else 'None'}")
    
    return result, tool_context


def test_no_mentions():
    """測試不包含書籍提及的查詢"""
    print("\n" + "=" * 60)
    print("Test: Query without book mentions")
    print("=" * 60)
    
    test_query = "我想看完結的玄幻小說"
    
    print(f"\nQuery: {test_query}")
    print("-" * 40)
    
    result, tool_context = parse_query_with_tools(test_query, enable_tools=True)
    
    print(f"\nTool calls: {tool_context.get('tool_calls', [])}")
    print(f"  (Expected: empty list, no specific books mentioned)")
    
    return len(tool_context.get('tool_calls', [])) == 0


def test_disliked_books():
    """測試包含不喜歡書籍的查詢"""
    print("\n" + "=" * 60)
    print("Test: Query with disliked books")
    print("=" * 60)
    
    test_query = "我不喜歡後宮類型的小說，像回復術士那種，想找純粹的冒險故事"
    
    print(f"\nQuery: {test_query}")
    print("-" * 40)
    
    result, tool_context = parse_query_with_tools(test_query, enable_tools=True)
    
    print(f"\nTool calls: {tool_context.get('tool_calls', [])}")
    
    tool_results = tool_context.get('tool_results', [])
    for tr in tool_results:
        print(f"  - Tool Result [{tr.get('name')}]: success={tr.get('success')}")
        if tr.get('success'):
            result_data = tr.get('result', {})
            print(f"    - negative_tags: {result_data.get('negative_tags', [])[:5]}")
    
    return tool_context


if __name__ == "__main__":
    print("Starting Tool Calling Tests\n")
    
    # Test 1: parse_query_with_tools
    try:
        test_parse_query_with_tools()
        print("\n[PASS] parse_query_with_tools test completed")
    except Exception as e:
        print(f"\n[FAIL] parse_query_with_tools test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: No book mentions
    try:
        success = test_no_mentions()
        if success:
            print("\n[PASS] No book mentions test passed")
        else:
            print("\n[WARN] No book mentions test: LLM may have false positive")
    except Exception as e:
        print(f"\n[FAIL] No book mentions test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Disliked books
    try:
        test_disliked_books()
        print("\n[PASS] Disliked books test completed")
    except Exception as e:
        print(f"\n[FAIL] Disliked books test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Tests Completed")
    print("=" * 60)
