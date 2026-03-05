import os
import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# ==========================================
# ⚙️ 實驗設定區 (測試不同引擎時請修改這裡)
# ==========================================
# 在不同的分支或實驗中，載入你想要測試的引擎
from src.core.engine import HybridReasonerEngine as TestEngine

# 設定這個引擎的名稱，將會作為輸出的檔名 (e.g., HybridReasoner.json)
ENGINE_NAME = "HybridReasoner"
# ==========================================

class RunGenerator:
    """
    單一引擎執行器
    負責對輸入的多個 Query 執行「一個」指定的系統檢索，並將結果存成 JSON。
    """
    def __init__(self, k_per_engine: int = 10):
        self.k = k_per_engine
        print(f"Initializing {ENGINE_NAME} Engine...")
        from src.core.database import Database
        from src.core.vector_store import VectorStore
        
        self.db = Database()
        self.vs = VectorStore(collection_name="novels")
        
        self.engine = TestEngine(db=self.db, vs=self.vs)
        
    def close(self):
        """Explicitly close Qdrant connection to avoid shutdown errors."""
        if hasattr(self, 'vs') and self.vs is not None:
            self.vs.client.close()

    def generate_run(self, queries_config: List[Dict], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ENGINE_NAME}.json"
        
        run_data = []
        
        for q_conf in queries_config:
            q_id = q_conf["id"]
            query = q_conf["query"]
            print(f"\nProcessing query: {query}")
            
            # 使用引擎抽取 Top-K (關閉 AI 解釋以節省 API 成本)
            response = self.engine.search(query, limit=self.k, explain=False)
            if asyncio.iscoroutine(response):
                response = asyncio.run(response)
                
            results = response.get("results", [])
            extracted_results = []
            
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
                    
                extracted_results.append({
                    "book_id": b_id,
                    "title": item.get("name", ""),
                    "author": author_name,
                    "intro": item.get("intro", ""),
                    "words_total": item.get("words_total", 0),
                    "publish_status": item.get("publish_status", ""),
                    "tags": item.get("tags", []),
                    "rank": rank + 1  # 1-based rank
                })
                
            run_data.append({
                "query_id": q_id,
                "query": query,
                "results": extracted_results
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(run_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n✅ [{ENGINE_NAME}] Run complete! Saved to {output_path}")

if __name__ == "__main__":
    with open("data/experiments/queries.json", "r", encoding="utf-8") as f:
        sample_queries = json.load(f)
        
    generator = RunGenerator(k_per_engine=10)
    try:
        generator.generate_run(sample_queries, Path("data/experiments/runs"))
    finally:
        generator.close()
