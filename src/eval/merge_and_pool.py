import csv
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

class PoolMerger:
    """
    結果合併與池化器
    負責讀取 multiple runs (由不同引擎產出的 JSON)，合併同 Query 下的書籍，
    打亂順序並輸出為雙盲測試表單 (.csv) 與對照解答本 (.json)。
    """
    def __init__(self, experiment_dir: str = "data/experiments/runs"):
        self.runs_dir = Path(experiment_dir)

    def load_runs(self) -> Dict[str, List[Dict]]:
        """
        Reads all JSON files in the runs directory and returns a dictionary
        mapping engine names (filenames without extension) to their run data.
        """
        all_runs = {}
        if not self.runs_dir.exists():
            print(f"Directory {self.runs_dir} does not exist.")
            return all_runs

        for file_path in sorted(self.runs_dir.glob("*.json")):
            engine_name = file_path.stem
            with open(file_path, 'r', encoding='utf-8') as f:
                run_data = json.load(f)
                all_runs[engine_name] = run_data
        return all_runs

    def merge_and_export(self, experiment_name: str, queries_config: List[Dict]):
        all_runs = self.load_runs()
        if not all_runs:
            print("No run files found to merge. Please execute generate_run.py first.")
            return

        print(f"Found {len(all_runs)} engine runs in {self.runs_dir}: {list(all_runs.keys())}")

        output_dir = self.runs_dir / "pools"
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
                
                candidates = {}  # book_id -> dict with details and source_engines
                
                # 走訪所有的 Engine 回報結果
                for engine_name, engine_run in all_runs.items():
                    # 找出該引擎對於此 Query 的結果
                    q_run = next((item for item in engine_run if item["query_id"] == q_id), None)
                    if not q_run:
                        continue
                        
                    for res in q_run["results"]:
                        b_id = res["book_id"]
                        rank = res["rank"]
                        
                        # 若為首次發現的書本，加入候選池
                        if b_id not in candidates:
                            candidates[b_id] = {
                                "book_id": b_id,
                                "title": res["title"],
                                "author": res.get("author", ""),
                                "intro": res.get("intro", ""),
                                "words_total": res.get("words_total", 0),
                                "publish_status": res.get("publish_status", ""),
                                "tags": res.get("tags", []),
                                "source_engines": [],
                                "original_ranks": {}
                            }
                            
                        # 紀錄這本書是被哪個系統找到的，以及名次
                        if engine_name not in candidates[b_id]["source_engines"]:
                            candidates[b_id]["source_engines"].append(engine_name)
                            candidates[b_id]["original_ranks"][engine_name] = rank
                            
                # 轉為 list 並打亂候選池，消除確認偏誤 (Confirmation Bias)
                pool = list(candidates.values())
                
                truth_entry = {
                    "query_id": q_id,
                    "query": query,
                    "pool_size": len(pool), # 紀錄池化大小
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
            
        print(f"\nMerge and Export Complete!")
        print(f"Blind Test CSV for annotator: {blind_csv_path}")
        print(f"Ground Truth tracking file:   {truth_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge experiment runs into a blind pool")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/runs/batch_YYYYMMDD_HHMMSS",
        help="Batch directory containing run JSON files",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="pilot_test",
        help="Experiment name for output files",
    )
    args = parser.parse_args()

    with open("data/experiments/queries.json", "r", encoding="utf-8") as f:
        sample_queries = json.load(f)
        
    merger = PoolMerger(experiment_dir=args.experiment_dir)
    merger.merge_and_export(args.experiment, sample_queries)
