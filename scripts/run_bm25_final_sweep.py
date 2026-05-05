"""
BM25 Final Sweep: Test β_max with the two best weight configurations.

Based on strict-only evaluation results:
  🥇 W_s=0.4, W_a=0.6  (strict Avg 2.1338)
  🥈 W_s=0.3, W_a=0.7  (strict Avg 2.1285)

Sweep β_max ∈ {0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20}
across both weight pairs → 2 × 7 = 14 runs total.
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

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_bm25_final")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_sameparse"

    # ══════════════════════════════════════════════════════════════
    # Experiment matrix: 2 weight pairs × 7 β_max values = 14 runs
    # ══════════════════════════════════════════════════════════════
    weight_pairs = [
        (0.4, 0.6, "s4a6"),   # strict-only 🥇
        (0.3, 0.7, "s3a7"),   # strict-only 🥈
    ]
    bmax_values = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    configs = []
    for ws, wa, label in weight_pairs:
        for bmax in bmax_values:
            name = f"{label}_b{int(bmax * 100):03d}"
            configs.append({
                "name": name,
                "semantic_weight": ws,
                "attribute_weight": wa,
                "bm25_bonus_max": bmax,
            })

    print("\n" + "█" * 70)
    print("  BM25 Final Sweep")
    print("  Weight pairs: (0.4/0.6) and (0.3/0.7)")
    print(f"  β_max ∈ {{{', '.join(f'{v:.2f}' for v in bmax_values)}}}")
    print(f"  Total: {len(configs)} configurations")
    print("█" * 70)

    # ── Run experiments ────────────────────────────────────────────
    for cfg in configs:
        print(
            f"\n{'='*60}\n"
            f"  {cfg['name']}  W_s={cfg['semantic_weight']:.2f}  "
            f"W_a={cfg['attribute_weight']:.2f}  β_max={cfg['bm25_bonus_max']:.2f}\n"
            f"{'='*60}"
        )
        generator = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
        generator.generate_run(
            queries_config=queries,
            engine_name=cfg["name"],
            output_dir=output_dir,
            semantic_weight=cfg["semantic_weight"],
            attribute_weight=cfg["attribute_weight"],
            run_suffix=cache_suffix,
            enable_bm25=True,
            bm25_weight=0.1,
            bm25_bonus_max=cfg["bm25_bonus_max"],
        )

    # ── Evaluate ───────────────────────────────────────────────────
    rows: list[dict] = []
    for cfg in configs:
        run_path = output_dir / f"{cfg['name']}{cache_suffix}.json"
        if not run_path.exists():
            print(f"[WARN] Missing: {run_path.name}")
            continue
        metrics = evaluate_run(run_path, annotations)
        metrics["semantic_weight"] = cfg["semantic_weight"]
        metrics["attribute_weight"] = cfg["attribute_weight"]
        metrics["bm25_bonus_max"] = cfg["bm25_bonus_max"]
        metrics["weight_pair"] = f"{cfg['semantic_weight']:.1f}/{cfg['attribute_weight']:.1f}"
        rows.append(metrics)

    # ── Print per-group summary ────────────────────────────────────
    for ws, wa, label in weight_pairs:
        group = [r for r in rows if r["semantic_weight"] == ws]
        group.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)

        if group:
            best_ndcg = group[0]["mean_ndcg_at_10"]
            for r in group:
                r["delta_ndcg"] = r["mean_ndcg_at_10"] - best_ndcg

        print(f"\n{'#'*80}")
        print(f"  W_s={ws:.1f} / W_a={wa:.1f}  —  BM25 β_max sweep")
        print(f"{'#'*80}")
        print(
            f"  {'#':>3} | {'β_max':>5} | "
            f"{'NDCG@10':>9} | {'Avg@10':>8} | {'Strict@10':>9} | {'Δ NDCG':>9}"
        )
        print(
            f"  {'-'*3}-+-{'-'*5}-+-"
            f"{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}"
        )
        for i, r in enumerate(group, 1):
            marker = " ★" if i == 1 else ""
            print(
                f"  {i:>3} | {r['bm25_bonus_max']:>5.2f} | "
                f"{r['mean_ndcg_at_10']:>9.4f} | {r['mean_avg_at_10']:>8.4f} | "
                f"{r['mean_strict_at_10']:>9.1f} | "
                f"{r.get('delta_ndcg', 0):>+9.4f}{marker}"
            )
        print(f"{'#'*80}")

    # ── Combined ranking ───────────────────────────────────────────
    rows.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)
    print(f"\n{'#'*90}")
    print(f"  COMBINED RANKING (All 14 Configurations)")
    print(f"{'#'*90}")
    print(
        f"  {'#':>3} | {'Weights':>7} | {'β_max':>5} | "
        f"{'NDCG@10':>9} | {'Avg@10':>8} | {'Strict@10':>9}"
    )
    print(
        f"  {'-'*3}-+-{'-'*7}-+-{'-'*5}-+-"
        f"{'-'*9}-+-{'-'*8}-+-{'-'*9}"
    )
    for i, r in enumerate(rows, 1):
        marker = " ★" if i == 1 else ""
        print(
            f"  {i:>3} | {r['weight_pair']:>7} | {r['bm25_bonus_max']:>5.2f} | "
            f"{r['mean_ndcg_at_10']:>9.4f} | {r['mean_avg_at_10']:>8.4f} | "
            f"{r['mean_strict_at_10']:>9.1f}{marker}"
        )
    print(f"{'#'*90}")

    if rows:
        best = rows[0]
        print(
            f"\n  ★ BEST: W_s={best['semantic_weight']:.2f}, "
            f"W_a={best['attribute_weight']:.2f}, "
            f"β_max={best['bm25_bonus_max']:.2f}"
        )
        print(
            f"    NDCG@10={best['mean_ndcg_at_10']:.4f}  "
            f"Avg@10={best['mean_avg_at_10']:.4f}  "
            f"Strict@10={best['mean_strict_at_10']:.1f}"
        )

    # ── Write results ──────────────────────────────────────────────
    fieldnames = [
        "run_file", "weight_pair", "semantic_weight", "attribute_weight",
        "bm25_bonus_max", "n_queries", "mean_ndcg_at_10",
        "mean_avg_at_10", "mean_strict_at_10",
    ]
    results_csv = output_dir / "bm25_final_sweep_results.csv"
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    results_json = output_dir / "bm25_final_sweep_results.json"
    results_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[Done] Output: {output_dir.as_posix()}")
    print(f"[Done] CSV:  {results_csv.as_posix()}")
    print(f"[Done] JSON: {results_json.as_posix()}")


if __name__ == "__main__":
    main()
