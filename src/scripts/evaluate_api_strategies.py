import asyncio
import json
import time
import os
import requests

def run_evaluation():
    queries_path = "data/experiments/queries.json"
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found. Please run this from the project root.")
        return
        
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
        
    strategies = ["score_only", "hybrid_fusion", "original_llm_reranker_top10"]
    limit_per_query = 5
    api_url = "http://127.0.0.1:8000/api/search"
    
    # 減少測試數量為 2 題，因為 API 預設會生成解釋(explain=True)，非常耗時
    test_queries = queries_data[:2]
    
    print(f"Starting Evaluation for {len(test_queries)} test queries across {len(strategies)} strategies (via Web API)...\n")
    print("⚠️ 注意: 因為 Web API 預設會為結果生成「解釋(Explanation)」，所以這會花費相當長的時間 (每筆可能超過 3 分鐘)。")
    print("⚠️ 腳本已設定 timeout=None，將會無限期等待伺服器算完，不會再報錯中斷。\n")
    
    metrics = {s: {"avg_latency": 0.0, "total_queries": 0, "rule_pass_rate": 0.0, "total_passed": 0} for s in strategies}
    
    for q_data in test_queries:
        query_text = q_data["query"]
        golden_rules = q_data.get("golden_rules", {})
        
        print("-" * 60)
        print(f"Query: {query_text[:60]}...")
        
        for strategy in strategies:
            start_time = time.time()
            payload = {
                "query": query_text,
                "model_id": "gemma-3-27b-it",
                "rerank_strategy": strategy,
                "rerank_alpha": 0.3
            }
            
            try:
                # 設定 timeout=None 讓 client 永遠等待，避免 Read timed out
                resp = requests.post(api_url, json=payload, timeout=None)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    latency = time.time() - start_time
                    
                    # Check Golden Rules Adherence
                    passed_rules = True
                    for res in results:
                        item_dict = res.get("item", {})
                        tags = item_dict.get("tags", [])
                        status = item_dict.get("publish_status", "")
                        classification = item_dict.get("classification", "")
                        
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
                        
                    print(f"  [{strategy:<25}] Latency: {latency:>5.2f}s | Results: {len(results)} | Rule Adherence: {'Pass' if passed_rules else 'Fail'}")
                else:
                    print(f"  [{strategy:<25}] API Error {(resp.status_code)}")
            except Exception as e:
                print(f"  [{strategy:<25}] Request Error: {e}")
                
            # Add a delay between requests to avoid overloading the server/LLM rate limits
            time.sleep(10)
                    
    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (Test set of 5 queries)")
    print("=" * 50)
    for strategy in strategies:
        stat = metrics[strategy]
        avg_lat = stat["avg_latency"] / max(1, stat["total_queries"])
        pass_rate = (stat["total_passed"] / max(1, stat["total_queries"])) * 100
        print(f"Strategy: {strategy}")
        print(f"  Avg Latency  : {avg_lat:.2f} seconds")
        print(f"  Rule Pass Rate: {pass_rate:.1f}% ({stat['total_passed']}/{stat['total_queries']})")
        print("-" * 30)
    print("\nTip: To evaluate all queries, change `queries_data[:5]` to `queries_data` in the script.")

if __name__ == "__main__":
    run_evaluation()
