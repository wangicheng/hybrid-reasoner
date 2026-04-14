import json
import asyncio
import sys
import os
import contextlib
from src.core.engine import HybridEngine

# 抑制外部函式庫日誌
import logging
logging.getLogger("jieba").setLevel(logging.ERROR)
import jieba
jieba.setLogLevel(logging.ERROR)

async def main():
    print("[INFO] 正在初始化成效評估引擎...")
    engine = HybridEngine()
    
    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    all_results = []
    total = len(queries)

    print(f"[PROGRESS] 開始執行批次測試 (總計 {total} 條查詢)...")
    
    for i, q_item in enumerate(queries):
        query_text = q_item if isinstance(q_item, str) else q_item.get("query")
        # 截斷過長的查詢顯示
        display_text = query_text.replace('\n', ' ')[:40] + "..."
        
        # 使用清除行末並覆寫的方式顯示動態進度
        print(f"\r[{i+1}/{total}] 正在處理: {display_text}", end='', flush=True)
        
        try:
            # 使用 redirect_stdout 到 devnull 來靜音搜尋過程中的詳細日誌
            with open(os.devnull, 'w', encoding='utf-8') as fnull:
                with contextlib.redirect_stdout(fnull):
                    response = await engine.search(query_text, limit=5)
            
            query_intent = response.get("query_intent") or {}
            hard_ex = [ex.get("term") for ex in query_intent.get("hard_exclusions", [])]
            soft_ex = [f"{ex.get('term')}({ex.get('weight')})" for ex in query_intent.get("soft_exclusions", [])]

            all_results.append({
                "id": q_item.get("id") if isinstance(q_item, dict) else f"q{i}",
                "query": query_text,
                "golden_rules": q_item.get("golden_rules", {}) if isinstance(q_item, dict) else {},
                "parsed_intent": {
                    "positive_terms": query_intent.get("positive_terms", []),
                    "hard_exclusions": hard_ex,
                    "soft_exclusions": soft_ex,
                    "sanitized_bm25_query": query_intent.get("sanitized_bm25_query", "")
                },
                "results": [
                    {
                        "title": r["item"].get("name"),
                        "score": round(r["score"], 4),
                        "tags": r["item"].get("tags")
                    } for r in response["results"]
                ]
            })
        except Exception as e:
            print(f"\n[FAIL] {display_text} 出錯: {e}")

    print("\n" + "-" * 50)
    with open("test_evaluation_results_modified.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("[SUCCESS] 測試完成！正式報表已輸出至 test_evaluation_results_modified.json")

if __name__ == "__main__":
    asyncio.run(main())


