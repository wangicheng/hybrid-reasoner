import asyncio
import json
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.engine import HybridEngine
from src.config import settings

async def run_latency_test():
    engine = HybridEngine()
    
    queries_path = "data/experiments/queries.json"
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        return
        
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
        
    strategies = ["score_only", "hybrid_fusion", "original_llm_reranker_top10"]
    limit_per_query = 10 
    
    print("\n" + "=" * 60)
    print(f"🚀 [第一階段] 純量測速 (Performance Run)")
    print(f"📚 題庫數量: {len(queries_data)} 題")
    print(f"⚙️  對比策略: {', '.join(strategies)}")
    print(f"⚠️  說明: 已強制關閉 explain=True，純測試「找書 + 排序」的硬實力速度。")
    print("=" * 60 + "\n")
    
    metrics = {s: {"avg_latency": 0.0, "total_queries": 0, "rule_pass_rate": 0.0, "total_passed": 0} for s in strategies}
    
    for idx, q_data in enumerate(queries_data, 1):
        query_text = q_data["query"]
        golden_rules = q_data.get("golden_rules", {})
        
        print(f"\n[{idx}/{len(queries_data)}] 測試查詢: {query_text[:40].replace(chr(10), ' ')}...")
        
        for strategy in strategies:
            start_time = time.time()
            
            try:
                # 執行搜尋 (強制 explain=False)
                response = await engine.search(
                    user_query=query_text,
                    limit=limit_per_query,
                    rerank_strategy=strategy,
                    explain=False  # <--- 關鍵：關閉寫作文干擾！
                )
                
                latency = time.time() - start_time
                results = response.get("results", [])
                
                # 檢查 Golden Rules (防守率)
                passed_rules = True
                for res in results:
                    book = res["item"]
                    tags = book.get("tags", [])
                    status = book.get("publish_status", "")
                    classification = book.get("classification", "")
                    
                    req_tags = golden_rules.get("required_tags", [])
                    blocked_tags = golden_rules.get("blocked_tags", [])
                    req_status = golden_rules.get("required_status")
                    
                    if blocked_tags:
                        for b_tag in blocked_tags:
                            if any(b_tag in tag for tag in tags) or b_tag in classification:
                                passed_rules = False
                                break
                    
                    if req_status == "completed" and status not in ["completed", "已完結", "完結"]:
                        passed_rules = False
                        
                    if not passed_rules:
                        break
                        
                metrics[strategy]["avg_latency"] += latency
                metrics[strategy]["total_queries"] += 1
                if passed_rules:
                    metrics[strategy]["total_passed"] += 1
                    
                status_mark = "✅" if passed_rules else "❌"
                print(f"  └─ [{strategy:<27}] 耗時: {latency:>5.2f}s | 守則: {status_mark}")
                
            except Exception as e:
                print(f"  └─ [{strategy:<27}] Error: {e}")
                
    # 輸出總結報告
    print("\n" + "═" * 50)
    print(" 📊 客觀指標評測總結 (Latency & Adherence)")
    print("═" * 50)
    
    # 按照速度排序輸出
    sorted_strategies = sorted(strategies, key=lambda s: metrics[s]["avg_latency"] / max(1, metrics[s]["total_queries"]))
    
    for strategy in sorted_strategies:
        stat = metrics[strategy]
        avg_lat = stat["avg_latency"] / max(1, stat["total_queries"])
        pass_rate = (stat["total_passed"] / max(1, stat["total_queries"])) * 100
        print(f"【{strategy}】")
        print(f"  ⏱️ 平均延遲: {avg_lat:.2f} 秒")
        print(f"  🛡️ 規則防守率: {pass_rate:.1f}% ({stat['total_passed']}/{stat['total_queries']})")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(run_latency_test())
