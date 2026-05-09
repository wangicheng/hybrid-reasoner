"""
RRF vs Weighted Fusion A/B Test

Compares:
  1. RRF fusion (k=60) with BM25 enabled
  2. RRF fusion (k=20) with BM25 enabled — more aggressive rank boosting
  3. Best weighted baseline (ws60_wa40, bm25_bonus=0.0) for reference

All runs share the same LLM parse cache.
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    avgs, stricts = [], []
    for entry in data:
        qid = str(entry.get("query_id", ""))
        results = entry.get("results", [])[:10]
        rels = [
            annotations.get(qid, {}).get(str(item.get("book_id", "")), 0.0)
            for item in results
        ]
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

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_rrf_test")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_sameparse"

    configs = [
        # ── RRF configs ──
        {
            "name": "rrf_k60",
            "fusion_strategy": "rrf",
            "rrf_k": 60,
            "semantic_weight": 0.5,  # ignored in RRF, but needed for init
            "attribute_weight": 0.5,
            "enable_bm25": True,
            "bm25_bonus_max": 0.0,
        },
        {
            "name": "rrf_k20",
            "fusion_strategy": "rrf",
            "rrf_k": 20,
            "semantic_weight": 0.5,
            "attribute_weight": 0.5,
            "enable_bm25": True,
            "bm25_bonus_max": 0.0,
        },
        {
            "name": "rrf_k60_no_bm25",
            "fusion_strategy": "rrf",
            "rrf_k": 60,
            "semantic_weight": 0.5,
            "attribute_weight": 0.5,
            "enable_bm25": False,
            "bm25_bonus_max": 0.0,
        },
        # ── Weighted baselines for comparison ──
        {
            "name": "weighted_ws60_wa40",
            "fusion_strategy": "weighted",
            "rrf_k": 60,
            "semantic_weight": 0.60,
            "attribute_weight": 0.40,
            "enable_bm25": True,
            "bm25_bonus_max": 0.0,
        },
        {
            "name": "weighted_ws35_wa65",
            "fusion_strategy": "weighted",
            "rrf_k": 60,
            "semantic_weight": 0.35,
            "attribute_weight": 0.65,
            "enable_bm25": True,
            "bm25_bonus_max": 0.0,
        },
    ]

    print(f"\n{'█' * 60}")
    print("  RRF vs Weighted Fusion A/B Test")
    print(f"  {len(configs)} configurations")
    print(f"  Output: {output_dir}")
    print(f"{'█' * 60}")

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  {cfg['name']}  fusion={cfg['fusion_strategy']}  rrf_k={cfg['rrf_k']}")
        print(f"{'=' * 60}")

        generator = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
        generator.generate_run(
            queries_config=queries,
            engine_name=cfg["name"],
            output_dir=output_dir,
            semantic_weight=cfg["semantic_weight"],
            attribute_weight=cfg["attribute_weight"],
            run_suffix=cache_suffix,
            enable_bm25=cfg["enable_bm25"],
            bm25_weight=0.1,
            bm25_bonus_max=cfg["bm25_bonus_max"],
            fusion_strategy=cfg["fusion_strategy"],
            rrf_k=cfg["rrf_k"],
        )

    # ── Evaluate ──
    print(f"\n{'#' * 60}")
    print("  RESULTS: RRF vs Weighted Fusion")
    print(f"{'#' * 60}")
    print(f"  {'Config':<25} | {'Avg@10':>8} | {'Strong@10':>9}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*9}")

    for cfg in configs:
        run_path = output_dir / f"{cfg['name']}{cache_suffix}.json"
        if not run_path.exists():
            print(f"  {cfg['name']:<25} | {'MISSING':>8} |")
            continue
        m = evaluate_run(run_path, annotations)
        print(
            f"  {cfg['name']:<25} | {m['mean_avg_at_10']:>8.4f} | "
            f"{m['mean_strict_at_10'] * 10:>8.1f}%"
        )

    print()


if __name__ == "__main__":
    main()
