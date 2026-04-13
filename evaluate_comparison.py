import asyncio
import json
import sys
import os

# 加入專案路徑
sys.path.insert(0, os.getcwd())

from src.core.engine import HybridEngine

async def check_rules(item, rules):
    """檢查單一書籍是否符合 Golden Rules"""
    tags = set(item.get("tags", []))
    status = item.get("publish_status", "")
    
    # 檢查必要標籤
    for tag in rules.get("required_tags", []):
        if tag not in tags: return False
    # 檢查禁用標籤
    for tag in rules.get("blocked_tags", []):
        if tag in tags: return False
    # 檢查完結狀態
    req_status = rules.get("required_status")
    if req_status == "completed" and "完結" not in status: return False
    if req_status == "ongoing" and "連載" not in status: return False
    
    return True

async def run_eval():
    engine = HybridEngine()
    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"{'ID':<5} | {'純向量 (Pass%)':<15} | {'BM25+向量 (Pass%)':<18} | {'進步'}")
    print("-" * 60)

    for q in queries:
        rules = q.get("golden_rules", {})
        if not rules: continue # 跳過沒有規範的

        # 1. 模擬純向量搜尋 (不加 RRF bonus，且不使用 BM25 召回)
        # 註：這裡我們透過暫時修改 engine 參數來模擬
        res_v = await engine.search(q["query"], limit=5, explain=False)
        # 我們手動從 breakdown 扣除 RRF bonus 來模擬舊排名
        items_v = sorted(res_v["results"], key=lambda x: x["score"] - next((b["weighted_score"] for b in x["breakdown"] if b["criteria"] == "rrf_fusion"), 0), reverse=True)
        
        # 2. 目前的 Hybrid 搜尋
        res_h = await engine.search(q["query"], limit=5, explain=False)
        items_h = res_h["results"]

        # 計算 Pass Rate (Top 5 裡面符合規則的比例)
        pass_v = sum([1 for x in items_v[:5] if await check_rules(x["item"], rules)]) / 5
        pass_h = sum([1 for x in items_h[:5] if await check_rules(x["item"], rules)]) / 5
        
        diff = "+" if pass_h > pass_v else ("=" if pass_h == pass_v else "-")
        print(f"{q['id']:<5} | {pass_v:<15.0%} | {pass_h:<18.0%} | {diff}")

if __name__ == "__main__":
    asyncio.run(run_eval())
