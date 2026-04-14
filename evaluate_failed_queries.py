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
    # 定義要追蹤測試的 ID
    target_ids = ["q1", "q2", "q16", "q17", "q19"]
    
    print(f"[INFO] 正在初始化失敗案例回測引擎... 目標: {target_ids}")
    engine = HybridEngine()
    
    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    # 篩選出目標題目
    target_queries = [q for q in queries if isinstance(q, dict) and q.get("id") in target_ids]
    
    if not target_queries:
        print("[ERROR] 在 queries.json 中找不到指定的失敗 ID！")
        return

    all_results = []
    total = len(target_queries)

    print(f"[PROGRESS] 開始執行精準回測 (總計 {total} 條查詢)...")
    
    for i, q_item in enumerate(target_queries):
        query_text = q_item.get("query")
        qid = q_item.get("id")
        
        display_text = f"[{qid}] " + query_text.replace('\n', ' ')[:40] + "..."
        print(f"[{i+1}/{total}] 正在處理: {display_text}")
        
        try:
            # 靜音詳細日誌以保持輸出純淨
            with open(os.devnull, 'w', encoding='utf-8') as fnull:
                with contextlib.redirect_stdout(fnull):
                    response = await engine.search(query_text, limit=5)
            
            query_intent = response.get("query_intent") or {}
            hard_ex = [ex.get("term") for ex in query_intent.get("hard_exclusions", [])]
            soft_ex = [f"{ex.get('term')}({ex.get('weight')})" for ex in query_intent.get("soft_exclusions", [])]

            all_results.append({
                "id": qid,
                "query": query_text,
                "golden_rules": q_item.get("golden_rules", {}),
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
            print(f"[FAIL] {qid} 出錯: {e}")

    # 輸出至獨立的結果檔案，避免覆蓋全量測試結果
    output_file = "failed_queries_retest.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("-" * 50)
    print(f"[SUCCESS] 回測完成！結果已儲存至 {output_file}")
    print(f"[TIP] 你執行完後可以跑 `python calculate_pass_rate.py` (需稍微修改讀取檔案路徑) 來看這些補測有沒有過。")

if __name__ == "__main__":
    asyncio.run(main())
