import os
import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path

# ==========================================
# ⚙️ 實驗設定區
# ==========================================
from src.core.engine import HybridEngine
from src.core.database import Database
from src.core.vector_store import VectorStore

class RunGenerator:
    """
    多實驗執行器
    負責對輸入的多個 Query 跑遍所有指定的實驗模式。
    """
    def __init__(self, k_per_engine: int = 10):
        self.k = k_per_engine
        self.db = Database()
        
    def generate_run(self, queries_config: List[Dict], engine_name: str, retrieval_mode: str, output_dir: Path):
        print(f"\n🚀 [Batch] Starting Experiment: {engine_name} (Mode: {retrieval_mode})")
        
        # [USER-SET] Re-sync with Engine: Only Exp 4 (fused) uses the pre-fused collection.
        # Exp 1, 2, 3, 5 all use Multi-Vector Score Fusion on the 'novels' collection.
        if "fused" in retrieval_mode:
            collection = "novels_fused"
        else:
            collection = "novels"
            
        print(f"   Using collection: {collection}")
        vs = VectorStore(collection_name=collection)
        engine = HybridEngine(db=self.db, vs=vs, retrieval_mode=retrieval_mode)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{engine_name}.json"
        
        run_data = []
        processed_query_ids = set()
        
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for item in existing_data:
                        # 如果有 'error' 欄位，代表上次失敗了，我們不把它加入跳過名單讓它重跑
                        if "error" not in item:
                            run_data.append(item)
                            processed_query_ids.add(item.get("query_id"))
                print(f"   ► Loaded {len(processed_query_ids)} completed queries from existing file. Resuming...")
            except Exception as e:
                print(f"   ⚠️ Could not load existing file: {e}")

        try:
            for q_conf in queries_config:
                q_id = q_conf["id"]
                query = q_conf["query"]
                
                if q_id in processed_query_ids:
                    print(f"   - Skipping query: {q_id} (already completed)")
                    continue
                    
                print(f"   - Processing query: {query[:30]}...")
                
                try:
                    # 使用引擎抽取 Top-K (關閉 AI 解釋以節省 API 成本)
                    response = engine.search(query, limit=self.k, explain=False)
                    if asyncio.iscoroutine(response):
                        response = asyncio.run(response)
                        
                    results = response.get("results", [])
                    extracted_results = []
                    
                    for rank, res in enumerate(results):
                        item = res.get("item", {})
                        b_id = str(item.get("id"))
                        if not b_id: continue
                            
                        author_name = item.get('author') or item.get('user', {}).get('name', '')
                            
                        extracted_results.append({
                            "book_id": b_id,
                            "title": item.get("name", ""),
                            "author": author_name,
                            "intro": item.get("intro", ""),
                            "words_total": item.get("words_total", 0),
                            "publish_status": item.get("publish_status", ""),
                            "tags": item.get("tags", []),
                            "rank": rank + 1
                        })
                        
                    run_data.append({
                        "query_id": q_id,
                        "query": query,
                        "results": extracted_results
                    })
                except Exception as query_err:
                    print(f"     ⚠️ Error processing query {q_id}: {query_err}")
                    # 添加空的結果，確保評估時對應得到 query_id
                    run_data.append({
                        "query_id": q_id,
                        "query": query,
                        "results": [],
                        "error": str(query_err)
                    })
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(run_data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ [{engine_name}] Run complete! Saved to {output_path}")
        finally:
            # 關閉連線以防 Qdrant lock
            vs.client.close()

if __name__ == "__main__":
    # 讀取問題集
    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        print(f"❌ Error: {queries_path} not found!")
        exit(1)
        
    with open(queries_path, "r", encoding="utf-8") as f:
        sample_queries = json.load(f)
        
    # 定義所有要跑的實驗 (對應 docs/experiments/tag_processing.md)
    EXPERIMENTS = [
        {"name": "exp1_a_5-5", "mode": "baseline"},
        {"name": "exp2_a_5-5", "mode": "baseline_prompt"},
        {"name": "exp3_a_5-5", "mode": "multi_multiplicative_embedded_tags"},
        {"name": "exp4", "mode": "fused_multiplicative"},
        {"name": "exp5_a_5-5", "mode": "multi_multiplicative"},
    ]
    
    generator = RunGenerator(k_per_engine=10)
    output_folder = Path("data/experiments/runs")
    
    for exp in EXPERIMENTS:
        try:
            generator.generate_run(
                queries_config=sample_queries,
                engine_name=exp["name"],
                retrieval_mode=exp["mode"],
                output_dir=output_folder
            )
        except Exception as e:
            print(f"❌ Failed experiment {exp['name']}: {e}")

    print("\n🎉 All scheduled experiments finished!")

