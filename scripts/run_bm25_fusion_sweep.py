"""
BM25 Fusion Mode Sweep Experiment

Compares 4 different BM25 scoring formulas:
  1. multiplicative : total = base × (1 + α × bm25_metric)       [original]
  2. additive       : total = base + α × bm25_metric              [flat boost]
  3. log_dampened   : total = base + α × log(1 + bm25_metric)     [diminishing returns]
  4. tiebreaker     : total = base + ε × bm25_metric              [micro-boost only]

Baseline: b000 (no BM25 scoring, recall-only) using W_s=0.3, W_a=0.7.

Each mode is tested with multiple α/ε values to find the sweet spot.
All runs share the same LLM parse cache.
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.eval.generate_run import RunGenerator


# ── Evaluation helpers ─────────────────────────────────────────────

def load_annotations(annotation_path: Path) -> dict[str, dict[str, float]]:
    annotations: dict[str, dict[str, float]] = {}
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


def dcg(rels: list[float]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_10(query_id: str, results: list[dict], annotations: dict[str, dict[str, float]]) -> float:
    rels = [annotations.get(query_id, {}).get(str(item.get("book_id", "")), 0.0) for item in results[:10]]
    ideal = sorted(annotations.get(query_id, {}).values(), reverse=True)[:10]
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0


def avg_at_10(query_id: str, results: list[dict], annotations: dict[str, dict[str, float]]) -> float:
    rels = [annotations.get(query_id, {}).get(str(item.get("book_id", "")), 0.0) for item in results[:10]]
    return (sum(rels) / 10.0) if rels else 0.0


def strict_at_10(query_id: str, results: list[dict], annotations: dict[str, dict[str, float]]) -> float:
    rels = [annotations.get(query_id, {}).get(str(item.get("book_id", "")), 0.0) for item in results[:10]]
    return float(sum(1 for s in rels if s >= 3.0))


def evaluate_run(run_path: Path, annotations: dict[str, dict[str, float]]) -> dict:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    ndcgs, avgs, stricts = [], [], []
    for entry in data:
        qid = str(entry.get("query_id", ""))
        results = entry.get("results", [])
        ndcgs.append(ndcg_at_10(qid, results, annotations))
        avgs.append(avg_at_10(qid, results, annotations))
        stricts.append(strict_at_10(qid, results, annotations))
    return {
        "run_file": run_path.name,
        "n_queries": len(data),
        "mean_ndcg_at_10": mean(ndcgs) if ndcgs else 0.0,
        "mean_avg_at_10": mean(avgs) if avgs else 0.0,
        "mean_strict_at_10": mean(stricts) if stricts else 0.0,
    }


# ── Experiment configs ─────────────────────────────────────────────

# Fixed weights: using the proven best
W_S = 0.3
W_A = 0.7

EXPERIMENT_CONFIGS: list[dict[str, Any]] = []

# ── Baseline: no BM25 scoring ──
EXPERIMENT_CONFIGS.append({
    "name": "baseline_b000",
    "fusion_mode": "multiplicative",
    "bonus_max": 0.0,
    "description": "Recall-only baseline (no BM25 scoring)",
})

# ── Multiplicative (original formula) ──
for alpha in [0.005, 0.01, 0.02, 0.05]:
    EXPERIMENT_CONFIGS.append({
        "name": f"mult_a{str(alpha).replace('.', '')}",
        "fusion_mode": "multiplicative",
        "bonus_max": alpha,
        "description": f"Multiplicative: base × (1 + {alpha} × metric)",
    })

# ── Additive (flat boost) ──
for alpha in [0.005, 0.01, 0.02, 0.05]:
    EXPERIMENT_CONFIGS.append({
        "name": f"add_a{str(alpha).replace('.', '')}",
        "fusion_mode": "additive",
        "bonus_max": alpha,
        "description": f"Additive: base + {alpha} × metric",
    })

# ── Log-dampened ──
for alpha in [0.005, 0.01, 0.02, 0.05]:
    EXPERIMENT_CONFIGS.append({
        "name": f"log_a{str(alpha).replace('.', '')}",
        "fusion_mode": "log_dampened",
        "bonus_max": alpha,
        "description": f"Log-dampened: base + {alpha} × log(1 + metric)",
    })

# ── Tie-breaker (micro-boost) ──
for epsilon in [0.0005, 0.001, 0.002, 0.005]:
    EXPERIMENT_CONFIGS.append({
        "name": f"tie_e{str(epsilon).replace('.', '')}",
        "fusion_mode": "tiebreaker",
        "bonus_max": epsilon,
        "description": f"Tie-breaker: base + {epsilon} × metric",
    })


def run_single_config(
    cfg: dict[str, Any],
    queries: list,
    output_dir: Path,
    cache_suffix: str,
) -> None:
    """Execute a single experiment configuration."""
    print(
        f"\n{'='*60}\n"
        f"[{cfg['fusion_mode']}] {cfg['name']}\n"
        f"  W_s={W_S:.2f}  W_a={W_A:.2f}  "
        f"α/ε={cfg['bonus_max']:.4f}  mode={cfg['fusion_mode']}\n"
        f"  {cfg['description']}\n"
        f"{'='*60}"
    )
    generator = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
    generator.generate_run(
        queries_config=queries,
        engine_name=cfg["name"],
        output_dir=output_dir,
        semantic_weight=W_S,
        attribute_weight=W_A,
        run_suffix=cache_suffix,
        enable_bm25=True,
        bm25_weight=0.1,
        bm25_bonus_max=cfg["bonus_max"],
        bm25_fusion_mode=cfg["fusion_mode"],
    )


def print_results_table(
    rows: list[dict],
    title: str,
) -> None:
    """Print formatted results table."""
    if not rows:
        print(f"  [{title}] No results.")
        return

    rows.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)
    best_ndcg = rows[0]["mean_ndcg_at_10"]

    print(f"\n{'#'*100}")
    print(f"  {title}")
    print(f"{'#'*100}")
    print(
        f"  {'#':>3} | {'Name':<22} | {'Mode':<14} | {'α/ε':>8} | "
        f"{'NDCG@10':>8} | {'Avg@10':>8} | {'Strict':>6} | {'Δ NDCG':>8}"
    )
    print(
        f"  {'-'*3}-+-{'-'*22}-+-{'-'*14}-+-{'-'*8}-+-"
        f"{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}"
    )
    for i, r in enumerate(rows, 1):
        delta = r["mean_ndcg_at_10"] - best_ndcg
        marker = " ★" if i == 1 else ""
        print(
            f"  {i:>3} | {r['name']:<22} | {r['fusion_mode']:<14} | "
            f"{r['bonus_max']:>8.4f} | "
            f"{r['mean_ndcg_at_10']:>8.4f} | {r['mean_avg_at_10']:>8.4f} | "
            f"{r['mean_strict_at_10']:>6.1f} | "
            f"{delta:>+8.4f}{marker}"
        )
    print(f"{'#'*100}")

    best = rows[0]
    print(
        f"\n  ★ BEST: {best['name']} "
        f"(mode={best['fusion_mode']}, α/ε={best['bonus_max']:.4f})"
    )
    print(
        f"    NDCG@10={best['mean_ndcg_at_10']:.4f}  "
        f"Avg@10={best['mean_avg_at_10']:.4f}  "
        f"Strict@10={best['mean_strict_at_10']:.1f}"
    )


def main() -> None:
    queries_path = Path("data/experiments/queries.json")
    annotations_path = Path("data/experiments/annotations/annotated.csv")
    output_root = Path("data/experiments/runs")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing: {queries_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing: {annotations_path}")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    annotations = load_annotations(annotations_path)

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_fusion_sweep")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_sameparse"

    print("█" * 80)
    print("  BM25 FUSION MODE SWEEP")
    print(f"  Fixed: W_s={W_S}, W_a={W_A}")
    print(f"  Modes: multiplicative, additive, log_dampened, tiebreaker")
    print(f"  Total configs: {len(EXPERIMENT_CONFIGS)}")
    print(f"  Output: {output_dir}")
    print("█" * 80)

    # ── Run all configs ──
    for cfg in EXPERIMENT_CONFIGS:
        run_single_config(cfg, queries, output_dir, cache_suffix)

    # ── Evaluate all ──
    all_rows: list[dict] = []
    for cfg in EXPERIMENT_CONFIGS:
        run_path = output_dir / f"{cfg['name']}{cache_suffix}.json"
        if not run_path.exists():
            print(f"[WARN] Missing: {run_path.name}")
            continue
        metrics = evaluate_run(run_path, annotations)
        metrics["name"] = cfg["name"]
        metrics["fusion_mode"] = cfg["fusion_mode"]
        metrics["bonus_max"] = cfg["bonus_max"]
        metrics["description"] = cfg["description"]
        all_rows.append(metrics)

    # ── Print per-mode summaries ──
    modes = ["multiplicative", "additive", "log_dampened", "tiebreaker"]
    for mode in modes:
        mode_rows = [r for r in all_rows if r["fusion_mode"] == mode]
        # Include baseline in each mode table for comparison
        baseline = [r for r in all_rows if r["name"] == "baseline_b000"]
        combined = baseline + [r for r in mode_rows if r["name"] != "baseline_b000"]
        print_results_table(combined, f"Mode: {mode}")

    # ── Print combined final ranking ──
    print_results_table(all_rows, "FINAL COMBINED RANKING (All Modes)")

    # ── Write results ──
    fieldnames = [
        "name", "fusion_mode", "bonus_max", "description",
        "n_queries", "mean_ndcg_at_10", "mean_avg_at_10", "mean_strict_at_10",
    ]

    all_rows.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)

    results_csv = output_dir / "fusion_sweep_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    results_json = output_dir / "fusion_sweep_results.json"
    results_json.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[Done] Output: {output_dir.as_posix()}")
    print(f"[Done] CSV:  {results_csv.as_posix()}")
    print(f"[Done] JSON: {results_json.as_posix()}")


if __name__ == "__main__":
    main()
