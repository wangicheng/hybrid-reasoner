"""
Hybrid Reasoner - Main Search Interface

Usage:
    python -m src.main
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

from src.core.engine import HybridEngine


def print_separator(char="=", length=60):
    print(char * length)


def display_results(result: dict):
    """Display search results in a user-friendly format."""
    
    print_separator()
    print(f"📝 Query: {result['query']}")
    print_separator("-")
    
    # Parsed Criteria
    print("\n🔍 AI 理解 (Parsed Criteria):")
    if result['parsed_criteria']:
        for i, c in enumerate(result['parsed_criteria'], 1):
            name = c.get('name', 'N/A')
            weight = c.get('weight', 0)
            params = c.get('parameters', {})
            # Filter out None values from params for cleaner display
            params_clean = {k: v for k, v in params.items() if v is not None}
            print(f"  {i}. {name} (權重: {weight:.2f})")
            if params_clean:
                print(f"     參數: {params_clean}")
    else:
        print("  (無具體條件，純語意搜尋)")
    
    print_separator("-")
    
    # Results
    print(f"\n📚 搜尋結果 (共 {len(result['results'])} 筆):\n")
    
    if not result['results']:
        print("  沒有找到符合條件的書籍。")
        return
    
    for i, res in enumerate(result['results'], 1):
        item = res['item']
        score = res['score']
        explanation = res.get('explanation')
        
        print(f"[第 {i} 名] 《{item.get('name', 'N/A')}》")
        print(f"  作者: {item.get('author', 'N/A')}")
        print(f"  分類: {item.get('classification', 'N/A')}")
        print(f"  標籤: {', '.join(item.get('tags', [])) if item.get('tags') else 'N/A'}")
        print(f"  總分: {score:.4f}")
        
        if explanation:
            print(f"\n  💡 AI 推薦理由:")
            # Wrap long explanations
            for line in explanation.split('\n'):
                print(f"     {line}")
        
        print()


def main():
    print_separator("=")
    print("🎯 Hybrid Reasoner - 小說推薦系統")
    print("   (輸入 'q' 或 'exit' 離開)")
    print_separator("=")

    # Check API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return
    print(f"✅ API Key found: {api_key[:5]}...")
    
    # Initialize Engine
    print("\n⏳ 正在初始化搜尋引擎...")
    engine = HybridEngine()
    print("✅ 初始化完成！\n")
    
    while True:
        try:
            query = input("請輸入查詢 (e.g., '奇幻小說有魔法的'): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('q', 'exit', 'quit'):
                print("\n👋 再見！")
                break
            
            print(f"\n⏳ 正在搜尋 \"{query}\"...\n")
            
            import asyncio
            result = asyncio.run(engine.search(query, limit=5))
            display_results(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 再見！")
            break
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
