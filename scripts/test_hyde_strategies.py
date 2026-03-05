import os
import sys
import asyncio
from typing import List, Dict, Any
from tabulate import tabulate

# Fix windows encoding issue
sys.stdout.reconfigure(encoding='utf-8')

# 為了讓 script 能 import src，將專案根目錄加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.vector_store import VectorStore
from src.core.database import Database

async def test_hyde_strategies(query: str, generated_keywords: List[str], long_hyde: str, short_hyde: str):
    print(f"==================================================")
    print(f"🔍 測試查詢: {query}")
    print(f"🔑 擴展關鍵字: {', '.join(generated_keywords)}")
    print(f"==================================================\n")

    vs = VectorStore(collection_name="novels")
    db = Database()

    # 準備四種策略的檢索字串
    base_terms = query
    expansion_str = " ".join([kw.replace(" ", "") for kw in generated_keywords])
    
    strategies = {
        "Baseline (原始)": f"{base_terms} {expansion_str} {long_hyde}",
        "Strategy 1 (移除 HyDE)": f"{base_terms} {expansion_str}",
        "Strategy 2 (權重壓制)": f"{base_terms} {base_terms} {base_terms} {expansion_str} {long_hyde}",
        "Strategy 3 (簡短 HyDE)": f"{base_terms} {expansion_str} {short_hyde}"
    }

    results_map = {}
    top_k_to_fetch = 50 # 模擬 Qdrant 召回池的數量

    for strategy_name, expanded_query in strategies.items():
        print(f"▶️ 正在執行 {strategy_name}...")
        print(f"   [檢索字串長度]: {len(expanded_query)} 字")
        
        # 呼叫 Qdrant 搜尋 (不帶任何 Filter，純看語意召回)
        vector_results, _ = vs.search(
            expanded_query, 
            limit=top_k_to_fetch,
            query_filter=None,
            with_payload=False # 這裡為了速度我們只取 ID
        )
        
        # 將結果存起來 (ID -> Score)
        scored_books = []
        for hit in vector_results:
            book_id = str(hit["id"])
            item = db.get_item(book_id)
            if item:
                scored_books.append({
                    "id": book_id,
                    "name": item.get("name", "Unknown"),
                    "author": item.get("author", "Unknown"),
                    "score": hit["score"]
                })
        
        results_map[strategy_name] = scored_books
        print(f"   ✅ 成功召回 {len(scored_books)} 本書\n")

    # === 分析結果 ===
    
    # 1. 比較召回池 (Top-50) 的重疊率 (Intersect)
    print("\n📊 【召回池 (Top-50) 書單重疊率分析】")
    baseline_ids = set(book["id"] for book in results_map["Baseline (原始)"])
    if baseline_ids:
        table_data = []
        for name, books in results_map.items():
            if name == "Baseline (原始)": continue
            strategy_ids = set(b["id"] for b in books)
            overlap = len(baseline_ids.intersection(strategy_ids))
            overlap_ratio = overlap / top_k_to_fetch * 100
            
            # 看看排名前 10 的書，差距多大
            baseline_top10 = set(book["id"] for book in results_map["Baseline (原始)"][:10])
            strategy_top10 = set(b["id"] for b in books[:10])
            top10_overlap = len(baseline_top10.intersection(strategy_top10))
            
            table_data.append([
                name, 
                f"{overlap} / {top_k_to_fetch} ({overlap_ratio:.1f}%)",
                f"{top10_overlap} / 10 ({top10_overlap/10*100:.1f}%)"
            ])
            
        print(tabulate(table_data, headers=["策略名稱", "整體 Top-50 重疊", "頭部 Top-10 重疊"], tablefmt="grid"))
    
    # 2. 顯示 Top-5 的分數與書名變化
    print("\n🏆 【各策略 Top-5 書單比較】")
    top5_table = []
    
    for rank in range(5):
        row = [f"Rank {rank+1}"]
        for name in strategies.keys():
            books = results_map[name]
            if rank < len(books):
                b = books[rank]
                # 將書名截斷以免表格太寬
                title = b['name'][:10] + ".." if len(b['name']) > 10 else b['name']
                row.append(f"{title} ({b['score']:.3f})")
            else:
                row.append("-")
        top5_table.append(row)
        
    headers = ["排名"] + list(strategies.keys())
    print(tabulate(top5_table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    # --- 測試案例 ---
    # 您可以根據實際發生問題的查詢修改這裡
    
    test_query = "想看大男主修仙"
    test_keywords = ["修仙", "大男主", "升級", "爽文", "無敵", "奇遇", "丹藥", "法寶", "門派", "秘境"]
    test_long_hyde = "在天元大陸，修仙世家林立，宗門爭霸。主角秦雲本是家族棄子，卻意外獲得上古大能的傳承。從此，他修煉絕世功法，煉製逆天神丹，手持無上法寶，一路高歌猛進。無論是傲慢的天才，還是陰險的門派長老，都被他踩在腳下。在這殘酷的修仙世界，秦雲誓要踏破蒼穹，成就萬古第一仙尊，讓所有人都為他的名字而顫抖！這是一段充滿熱血與激情的修仙之旅，見證一個平凡少年如何逆襲成為修仙界的傳奇。"
    test_short_hyde = "這是一部關於大男主修仙的爽文，主角獲得奇遇後一路升級打怪，橫掃修仙門派的故事。"
    
    asyncio.run(test_hyde_strategies(
        query=test_query,
        generated_keywords=test_keywords,
        long_hyde=test_long_hyde,
        short_hyde=test_short_hyde
    ))
