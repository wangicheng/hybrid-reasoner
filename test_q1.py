import asyncio
import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.engine import HybridEngine

async def test():
    engine = HybridEngine()
    query = """現在對於小說的要求越來越高了，
而且慢慢的發現現在除了狗糧基本上都看不進去，
連以很喜歡的異世界主題都看不進去了，
好想念從前那個什麼都可以看得很快樂的自己

說回主題，希望可以找到沒看過的狗糧系小說，
可以接受小刀，但一定要無黃毛無牛頭人，
但背景自帶沈重的例如「戀愛光譜」「刮鬍」這一類的也不喜歡，
黨爭就...看情況吧但其實我也一般般"""

    res = await engine.search(query, limit=3, rerank_strategy="score_only", explain=False)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test())
