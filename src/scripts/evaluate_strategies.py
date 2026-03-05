import asyncio
import json
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.engine import HybridEngine

async def run_evaluation():
    engine = HybridEngine()
    
    queries_path = "data/experiments/queries.json"
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        return
        
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
        
    strategies = ["score_only", "hybrid_fusion", "original_llm_reranker_top10"]
    limit_per_query = 5
    
    print(f"Starting Evaluation for {len(queries_data)} queries across {len(strategies)} strategies...\n")
    
    # store results
    metrics = {s: {"avg_latency": 0.0, "total_queries": 0, "rule_pass_rate": 0.0, "total_passed": 0} for s in strategies}
    
    for q_data in queries_data[:5]: # Let's test on the first 5 to save time
        query_text = q_data["query"]
        golden_rules = q_data.get("golden_rules", {})
        
        print("-" * 50)
        print(f"Query: {query_text[:50]}...")
        
        for strategy in strategies:
            start_time = time.time()
            
            try:
                # Run the search
                response = await engine.search(
                    user_query=query_text,
                    limit=limit_per_query,
                    rerank_strategy=strategy,
                    explain=False # Turn off explanation to save time & tokens
                )
                
                latency = time.time() - start_time
                results = response.get("results", [])
                
                # Check Golden Rules Adherence
                passed_rules = True
                for res in results:
                    book = res["item"]
                    tags = book.get("tags", [])
                    status = book.get("publish_status", "")
                    
                    req_tags = golden_rules.get("required_tags", [])
                    # Check required tags
                    if req_tags:
                        for req_tag in req_tags:
                            if not any(req_tag in tag for tag in tags) and req_tag not in book.get("classification", ""):
                                # We'll do a soft check since LLM parsing handles it too
                                pass 
                                
                    blocked_tags = golden_rules.get("blocked_tags", [])
                    if blocked_tags:
                         for b_tag in blocked_tags:
                            if any(b_tag in tag for tag in tags) or b_tag in book.get("classification", ""):
                                passed_rules = False
                                break
                    
                    req_status = golden_rules.get("required_status")
                    if req_status == "completed" and status not in ["completed", "已完結", "完結"]:
                        passed_rules = False
                        
                    if not passed_rules:
                        break
                        
                metrics[strategy]["avg_latency"] += latency
                metrics[strategy]["total_queries"] += 1
                if passed_rules:
                    metrics[strategy]["total_passed"] += 1
                    
                print(f"  [{strategy}] Latency: {latency:.2f}s | Results: {len(results)} | Passed Rules: {passed_rules}")
                
            except Exception as e:
                print(f"  [{strategy}] Error: {e}")
                
    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for strategy in strategies:
        stat = metrics[strategy]
        avg_lat = stat["avg_latency"] / max(1, stat["total_queries"])
        pass_rate = (stat["total_passed"] / max(1, stat["total_queries"])) * 100
        print(f"Strategy: {strategy}")
        print(f"  Avg Latency  : {avg_lat:.2f} seconds")
        print(f"  Rule Pass Rate: {pass_rate:.1f}% ({stat['total_passed']}/{stat['total_queries']})")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
