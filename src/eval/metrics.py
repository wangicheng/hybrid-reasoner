import json
import csv
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def apply_strict_filters(golden_rules: Dict[str, Any], book_item: Dict[str, Any]) -> bool:
    """
    Returns False if the book explicitly violates hard constraints defined in the golden rules.
    這就是強硬條件仲裁：用程式無情抓出標註員（或者純向量引擎）無視字數或狀態的錯誤！
    """
    # Check numeric range
    min_words = golden_rules.get("min_words")
    max_words = golden_rules.get("max_words")
    words_total = book_item.get("words_total", 0)
    
    if min_words is not None and words_total < min_words:
        return False
    if max_words is not None and words_total > max_words:
        return False
        
    # Check status
    req_status = golden_rules.get("required_status")
    if req_status:
        status = str(book_item.get("publish_status", "")).lower()
        if req_status == "completed" and status not in ["completed", "已完結", "完結"]:
            return False
        if req_status == "ongoing" and status not in ["ongoing", "連載中", "連載"]:
            return False
            
    # Check animated
    must_be_animated = golden_rules.get("must_be_animated")
    if must_be_animated is not None:
        if bool(book_item.get("is_animated")) != bool(must_be_animated):
            return False
            
    # Check required tags
    req_tags = golden_rules.get("required_tags") or []
    if req_tags:
        book_tags = set(book_item.get("tags", []))
        for rt in req_tags:
            if rt not in book_tags:
                return False
                
    # Check blocked tags
    blocked_tags = golden_rules.get("blocked_tags") or []
    if blocked_tags:
        book_tags = set(book_item.get("tags", []))
        for bt in blocked_tags:
            if bt in book_tags:
                return False
                
    return True

def calculate_ndcg(ranked_scores: List[float], k: int) -> float:
    """計算給定 Top-K 分數陣列的 NDCG (Normalized Discounted Cumulative Gain)"""
    dcg = 0.0
    for i in range(min(k, len(ranked_scores))):
        rel = ranked_scores[i]
        dcg += (2**rel - 1) / math.log2(i + 2)
        
    ideal_scores = sorted(ranked_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_scores))):
        rel = ideal_scores[i]
        idcg += (2**rel - 1) / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(experiment_name: str, use_strict_filter: bool = True):
    base_dir = Path("data/experiments/pools")
    annotated_csv = base_dir / f"{experiment_name}_annotated.csv"
    truth_json = base_dir / f"{experiment_name}_truth.json"
    
    # 0. Load Golden Rules mapping from query -> rules
    with open("data/experiments/queries.json", "r", encoding="utf-8") as f:
        queries_config = json.load(f)
    golden_rules_map = {item["query"]: item["golden_rules"] for item in queries_config}
    
    # 0.5 Load all crawled books to build full metadata
    books_crawled_map = {}
    books_crawled_path = Path("data/books_crawled.json")
    if books_crawled_path.exists():
        with open(books_crawled_path, "r", encoding="utf-8") as f:
            try:
                crawled_data = json.load(f)
                for b in crawled_data:
                    books_crawled_map[str(b.get("id", ""))] = b
            except json.JSONDecodeError:
                print("Warning: Failed to parse books_crawled.json")

    # 1. Load Ground Truth tracking data (who found what!)
    with open(truth_json, "r", encoding="utf-8") as f:
        truth_data = json.load(f)
        
    # 2. Load the human (or mock) annotations
    annotations = {}  # { query: { book_id: score } }
    books_data = {}   # { book_id: dict info }
    
    with open(annotated_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["Query"]
            bid = row["Book ID"]
            try:
                score = float(row["Score (0-3)"])
            except ValueError:
                score = 0.0
                
            if q not in annotations:
                annotations[q] = {}
            annotations[q][bid] = score
            
            # Reconstruct dummy dict for strict filter
            if str(bid) in books_crawled_map:
                books_data[bid] = books_crawled_map[str(bid)]
            else:
                words_in_10k = float(row.get("Words (萬)") or 0)
                intro = row.get("Intro", "")
                
                # Parse tags from "[標籤: 架空, 穿越]"
                tags = []
                if "[標籤:" in intro:
                    start_idx = intro.find("[標籤:") + 4
                    end_idx = intro.find("]", start_idx)
                    if end_idx != -1:
                        tags_str = intro[start_idx:end_idx]
                        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                        
                books_data[bid] = {
                    "words_total": words_in_10k * 10000,
                    "publish_status": row.get("Status", ""),
                    "tags": tags,
                    "is_animated": False  # Default fallback
                }

    # 3. Apply Strict Filter (The Arbitrator)
    if use_strict_filter:
        print("\n🔍 [Strict Filter] 正在進行強制仲裁審查 (尋找字數/狀態/標籤/動畫的違規項目)...")
        for q, books_in_query in annotations.items():
            golden_rules = golden_rules_map.get(q)
            if not golden_rules:
                continue
                
            for bid, score in books_in_query.items():
                if score > 0:
                    passed = apply_strict_filters(golden_rules, books_data[bid])
                    if not passed:
                        print(f"  ❌ [降維打擊] Query: '{q}'\n      Book ID: {bid} 違反硬性條件，標註員分數: {score} -> 強制降為 0！")
                        annotations[q][bid] = 0.0

    # 4. Calculate NDCG per Engine per Query
    engine_ndcg = defaultdict(list)
    
    for truth_entry in truth_data:
        query = truth_entry["query"]
        candidates = truth_entry["candidates"]
        
        # Engine -> list of (rank, score)
        engine_results = defaultdict(list)
        
        for cand in candidates:
            bid = str(cand["book_id"])
            # 若書本有在該 Query 的評分表裡，就抓出分數，否則預設給 0 分
            score = annotations.get(query, {}).get(bid, 0.0)
            
            # 填入這本書在各個推薦引擎中的名次與分數
            for engine_name, rank in cand["original_ranks"].items():
                engine_results[engine_name].append((rank, score))
                
        for engine_name, results in engine_results.items():
            # sort by original engine rank ascending
            results.sort(key=lambda x: x[0])
            ranked_scores = [s for r, s in results]
            
            # Pad with 0s if engine returned fewer than 10 results but k=10
            ranked_scores += [0.0] * max(0, 10 - len(ranked_scores))
            
            ndcg_10 = calculate_ndcg(ranked_scores, k=10)
            engine_ndcg[engine_name].append(ndcg_10)

    # 5. Output Final Report
    print("\n" + "="*40)
    print("📊 實驗評估報告 (Experiment Evaluation)")
    print("="*40)
    print(f"🔹 實驗名稱: {experiment_name}")
    print(f"🔹 啟用強硬條件仲裁 (Strict Filter): {use_strict_filter}")
    print("-"*40)
    
    for engine_name, scores in engine_ndcg.items():
        avg_ndcg = sum(scores) / len(scores) if scores else 0
        print(f"  🏆 {engine_name:20s} | NDCG@10: {avg_ndcg:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Evaluation Metrics")
    parser.add_argument("--experiment", type=str, default="pilot_test", help="Experiment name")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict filtering")
    args = parser.parse_args()
    
    run_evaluation(args.experiment, use_strict_filter=not args.no_strict)
