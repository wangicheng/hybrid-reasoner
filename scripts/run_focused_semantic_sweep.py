"""
Focused Semantic Weight Sweep (W_s: 60%, 65%, 70%)
Runs two iterations to test the stability of LLM parsing.
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.generate_run import RunGenerator

def load_annotations(annotation_path: Path) -> dict[str, dict[str, float]]:
    annotations = {}
    with annotation_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = str(row.get("Query ID", "")).strip()
            book_id = str(row.get("Book ID", "")).strip()
            if not query_id or not book_id:
                continue
            try:
                score = float(row.get("Score (0-3)", "") or 0)
            except ValueError:
                score = 0.0
            annotations.setdefault(query_id, {})[book_id] = score
    return annotations

def evaluate_run(run_path: Path, annotations: dict[str, dict[str, float]]) -> dict:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    stricts, avgs = [], []
    for entry in data:
        qid = str(entry.get("query_id", ""))
        results = entry.get("results", [])[:10]
        rels = [annotations.get(qid, {}).get(str(item.get("book_id", "")), 0.0) for item in results]
        
        avgs.append((sum(rels) / 10.0) if rels else 0.0)
        stricts.append(float(sum(1 for s in rels if s >= 3.0)))
        
    return {
        "run_file": run_path.name,
        "mean_avg_at_10": mean(avgs) if avgs else 0.0,
        "mean_strict_at_10": mean(stricts) if stricts else 0.0,
    }

def main() -> None:
    queries_path = Path("data/experiments/queries.json")
    annotations_path = Path("data/experiments/annotations/annotated.csv")
    output_root = Path("data/experiments/runs")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    annotations = load_annotations(annotations_path)

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_focused_sweep")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for ws_int in [60, 65, 70]:
        ws = ws_int / 100.0
        wa = round(1.0 - ws, 2)
        configs.append({
            "name": f"ws{ws_int}_wa{int(wa*100)}",
            "semantic_weight": ws,
            "attribute_weight": wa,
        })

    print(f"\nStarting Focused Sweep. Output dir: {output_dir}\n")

    for iteration in [1, 2]:
        run_suffix = f"_iter{iteration}"
        print(f"========== ITERATION {iteration} ==========")
        
        for cfg in configs:
            print(f"  Running: {cfg['name']} (W_s={cfg['semantic_weight']:.2f})")
            generator = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
            generator.generate_run(
                queries_config=queries,
                engine_name=cfg["name"],
                output_dir=output_dir,
                semantic_weight=cfg["semantic_weight"],
                attribute_weight=cfg["attribute_weight"],
                run_suffix=run_suffix,
                enable_bm25=True,
                bm25_weight=0.1,
                bm25_bonus_max=0.0,
            )

    print("\n========== EVALUATION RESULTS ==========")
    for iteration in [1, 2]:
        print(f"\n--- Iteration {iteration} ---")
        run_suffix = f"_iter{iteration}"
        for cfg in configs:
            run_path = output_dir / f"{cfg['name']}{run_suffix}.json"
            if run_path.exists():
                metrics = evaluate_run(run_path, annotations)
                print(f"{cfg['name']:>12} | Avg@10: {metrics['mean_avg_at_10']:.4f} | Strong@10: {(metrics['mean_strict_at_10']*10):.1f}%")

if __name__ == "__main__":
    main()
