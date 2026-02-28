import os
import csv
import json
import random
from typing import List, Dict, Any
from pathlib import Path

# Import the 4 engines
from src.core.engine import (
    ExactMatchEngine,
    PureVectorEngine,
    FilteredVectorEngine,
    HybridReasonerEngine
)

class PoolGenerator:
    """
    Treac Pooling 抽樣器
    負責對輸入的多個 Query 執行 4 個系統的檢索，合併 Top-K 候選書單，
    打亂順序並輸出為雙盲測試表單 (.csv)，同時保留對照解答本 (.json)。
    """
    def __init__(self, k_per_engine: int = 10):
        self.k = k_per_engine
        print("Initializing engines (Base & VectorStore)...")
        from src.core.database import Database
        from src.core.vector_store import VectorStore
        
        shared_db = Database()
        self.shared_vs = VectorStore(collection_name="novels")
        
        self.engines = {
            "ExactMatch": ExactMatchEngine(db=shared_db, vs=self.shared_vs),
            "PureVector": PureVectorEngine(db=shared_db, vs=self.shared_vs),
            "FilteredVector": FilteredVectorEngine(db=shared_db, vs=self.shared_vs),
            "HybridReasoner": HybridReasonerEngine(db=shared_db, vs=self.shared_vs)
        }
        
    def close(self):
        """Explicitly close Qdrant connection to avoid shutdown errors."""
        if hasattr(self, 'shared_vs') and self.shared_vs is not None:
            self.shared_vs.client.close()
        
    def generate_pool_for_query(self, query: str) -> List[Dict[str, Any]]:
        print(f"\nProcessing query: {query}")
        
        candidates = {}  # book_id -> dict with details and source_engines
        
        for engine_name, engine in self.engines.items():
            print(f"  Running {engine_name}...")
            # 每個 Engine 抽取 Top-K (關閉 AI 解釋以節省 API 成本)
            response = engine.search(query, limit=self.k, explain=False)
            
            # 從回傳中拿出 results
            results = response.get("results", [])
            for rank, res in enumerate(results):
                item = res.get("item", {})
                b_id = str(item.get("id"))
                
                if not b_id:
                    continue
                    
                # 處理新舊 Author Schema
                author_name = ""
                if isinstance(item.get('user'), dict):
                    author_name = item.get('user', {}).get('name', '')
                else:
                    author_name = item.get('author', '')
                    
                # 若為首次發現的書本，加入候選池
                if b_id not in candidates:
                    candidates[b_id] = {
                        "book_id": b_id,
                        "title": item.get("name", ""),
                        "author": author_name,
                        "intro": item.get("intro", ""),
                        "words_total": item.get("words_total", 0),
                        "publish_status": item.get("publish_status", ""),
                        "tags": item.get("tags", []),
                        "source_engines": [],
                        "original_ranks": {}
                    }
                
                # 紀錄這本書是被哪個系統找到的，以及名次
                if engine_name not in candidates[b_id]["source_engines"]:
                    candidates[b_id]["source_engines"].append(engine_name)
                    candidates[b_id]["original_ranks"][engine_name] = rank + 1  # 1-based rank
                    
        # 將字典轉為 List 並回傳
        pool = list(candidates.values())
        return pool

    def export_blind_test_batch(self, queries_config: List[Dict], experiment_name: str):
        """
        Runs queries, pools candidates, shuffles them per query, and exports:
        1. A blind test CSV for annotators.
        2. A ground truth JSON with engine sources and ranks.
        """
        output_dir = Path("data/experiments/pools")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        blind_csv_path = output_dir / f"{experiment_name}_blind.csv"
        truth_json_path = output_dir / f"{experiment_name}_truth.json"
        
        truth_data = []
        
        # 寫入提供給標註員的雙盲測試 CSV（加上 BOM 確保 Excel 開啟中文不會亂碼）
        with open(blind_csv_path, 'w', encoding='utf-8-sig', newline='') as f_csv:
            writer = csv.writer(f_csv)
            # CSV 標題，請標註員填寫 Score 與 Comment
            writer.writerow(["Query ID", "Query", "Book ID", "Title", "Author", "Words (萬)", "Status", "Intro", "Score (0-3)", "Comment"])
            
            for q_conf in queries_config:
                q_id = q_conf["id"]
                query = q_conf["query"]
                pool = self.generate_pool_for_query(query)
                
                # 打亂候選池，消除確認偏誤 (Confirmation Bias)
                random.shuffle(pool)
                
                truth_entry = {
                    "query_id": q_id,
                    "query": query,
                    "pool_size": len(pool), # 紀錄池化大小 (最大應為 K * 4，通常去重後會更少)
                    "candidates": []
                }
                
                for item in pool:
                    # 處理顯示欄位，例如字數轉為萬字以便閱讀
                    words_in_10k = round(item["words_total"] / 10000, 1) if item["words_total"] else 0
                    
                    # 簡介太長的話截斷，以防 Excel 爆格
                    short_intro = item["intro"][:250] + "..." if len(item["intro"]) > 250 else item["intro"]
                    # 處理可能是 array 的 tags
                    tags_str = item["tags"] if isinstance(item["tags"], str) else ", ".join([str(t) for t in item["tags"]])
                    
                    writer.writerow([
                        q_id,
                        query,
                        item["book_id"],
                        item["title"],
                        item["author"],
                        words_in_10k,
                        item["publish_status"],
                        f"[標籤: {tags_str}]\n{short_intro}",
                        "", # Score 留白給標註員填
                        ""  # Comment 留白給標註員填
                    ])
                    
                    # 紀錄到對照解答本 (Truth Data)
                    truth_entry["candidates"].append({
                        "book_id": item["book_id"],
                        "title": item["title"],
                        "source_engines": item["source_engines"],
                        "original_ranks": item["original_ranks"]
                    })
                    
                truth_data.append(truth_entry)
                
        # 寫入解答本
        with open(truth_json_path, 'w', encoding='utf-8') as f_json:
            json.dump(truth_data, f_json, ensure_ascii=False, indent=2)
            
        print(f"\n✅ Export Complete!")
        print(f"Blind Test CSV for annotator: {blind_csv_path}")
        print(f"Ground Truth tracking file:   {truth_json_path}")


if __name__ == "__main__":
    with open("data/experiments/queries.json", "r", encoding="utf-8") as f:
        sample_queries = json.load(f)
    
    # 每個 Engine 取 Top-10
    generator = PoolGenerator(k_per_engine=10)
    try:
        generator.export_blind_test_batch(sample_queries, "pilot_test")
    finally:
        generator.close()
