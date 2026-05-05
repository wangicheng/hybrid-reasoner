"""
BM25 Three-Phase Weight Sweep Experiment

Phase A: BM25 Bonus Max sweep (fixed W_s=0.3, W_a=0.7)
  β_max ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}

Phase B: Semantic / Attribute weight sweep (fixed β_max=0.15)
  (W_s, W_a) ∈ {(0.2,0.8), (0.4,0.6), (0.5,0.5), (0.6,0.4)}
  Note: (0.3,0.7) is shared with A4, not re-run.

Phase C: Cross-validation of top combos from A & B
  3 runs combining the best results from both phases.

Total: ~14 runs, all sharing the same LLM parse cache.
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


def run_config(
    cfg: dict[str, Any],
    queries: list,
    output_dir: Path,
    cache_suffix: str,
) -> None:
    """Execute a single experiment configuration."""
    print(
        f"\n{'='*60}\n"
        f"[{cfg['phase']}] {cfg['name']}\n"
        f"  W_s={cfg['semantic_weight']:.2f}  W_a={cfg['attribute_weight']:.2f}  "
        f"β_max={cfg['bm25_bonus_max']:.2f}\n"
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


def evaluate_phase(
    configs: list[dict],
    output_dir: Path,
    cache_suffix: str,
    annotations: dict,
    phase_label: str,
) -> list[dict]:
    """Evaluate all runs in a phase and print summary."""
    rows = []
    for cfg in configs:
        run_path = output_dir / f"{cfg['name']}{cache_suffix}.json"
        if not run_path.exists():
            print(f"[WARN] Missing: {run_path.name}")
            continue
        metrics = evaluate_run(run_path, annotations)
        metrics["phase"] = cfg["phase"]
        metrics["semantic_weight"] = cfg["semantic_weight"]
        metrics["attribute_weight"] = cfg["attribute_weight"]
        metrics["bm25_bonus_max"] = cfg["bm25_bonus_max"]
        rows.append(metrics)

    if not rows:
        print(f"[{phase_label}] No results to evaluate!")
        return rows

    rows.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)

    # Delta vs first row in this phase (best becomes reference after sort)
    baseline_ndcg = rows[-1]["mean_ndcg_at_10"]  # worst as reference
    baseline_strict = rows[-1]["mean_strict_at_10"]

    print(f"\n{'='*90}")
    print(f"  {phase_label} Results (sorted by NDCG@10)")
    print(f"{'='*90}")
    print(
        f"  {'Config':<30} | {'W_s':>5} | {'W_a':>5} | {'β_max':>5} | "
        f"{'NDCG@10':>8} | {'Avg@10':>8} | {'Strict':>6}"
    )
    print(f"  {'-'*30}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for r in rows:
        print(
            f"  {r['run_file']:<30} | {r['semantic_weight']:>5.2f} | "
            f"{r['attribute_weight']:>5.2f} | {r['bm25_bonus_max']:>5.2f} | "
            f"{r['mean_ndcg_at_10']:>8.4f} | {r['mean_avg_at_10']:>8.4f} | "
            f"{r['mean_strict_at_10']:>6.1f}"
        )
    print(f"{'='*90}")

    return rows


def print_final_summary(all_rows: list[dict]) -> None:
    """Print a combined summary across all phases."""
    all_rows.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)
    print(f"\n{'#'*90}")
    print(f"  FINAL COMBINED RANKING (All Phases)")
    print(f"{'#'*90}")
    print(
        f"  {'#':>3} | {'Phase':<8} | {'W_s':>5} | {'W_a':>5} | {'β_max':>5} | "
        f"{'NDCG@10':>8} | {'Avg@10':>8} | {'Strict':>6}"
    )
    print(
        f"  {'-'*3}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-"
        f"{'-'*8}-+-{'-'*8}-+-{'-'*6}"
    )
    for i, r in enumerate(all_rows, 1):
        marker = " ★" if i == 1 else ""
        print(
            f"  {i:>3} | {r['phase']:<8} | {r['semantic_weight']:>5.2f} | "
            f"{r['attribute_weight']:>5.2f} | {r['bm25_bonus_max']:>5.2f} | "
            f"{r['mean_ndcg_at_10']:>8.4f} | {r['mean_avg_at_10']:>8.4f} | "
            f"{r['mean_strict_at_10']:>6.1f}{marker}"
        )
    print(f"{'#'*90}")

    best = all_rows[0]
    print(
        f"\n  ★ BEST CONFIG: W_s={best['semantic_weight']:.2f}, "
        f"W_a={best['attribute_weight']:.2f}, β_max={best['bm25_bonus_max']:.2f}"
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

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_3phase_sweep")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_sameparse"

    all_rows: list[dict] = []

    # ══════════════════════════════════════════════════════════════
    # Phase A: BM25 Bonus Max sweep (fixed W_s=0.3, W_a=0.7)
    # ══════════════════════════════════════════════════════════════
    phase_a_configs = []
    for bmax in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        label = f"A_bonus{int(bmax * 100):03d}"
        phase_a_configs.append({
            "phase": "PhaseA",
            "name": label,
            "semantic_weight": 0.30,
            "attribute_weight": 0.70,
            "bm25_bonus_max": bmax,
        })

    print("\n" + "█" * 60)
    print("  PHASE A: BM25 Bonus Max Sweep")
    print("  Fixed: W_s=0.30, W_a=0.70")
    print("  Sweep: β_max ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}")
    print("█" * 60)

    for cfg in phase_a_configs:
        run_config(cfg, queries, output_dir, cache_suffix)

    phase_a_rows = evaluate_phase(phase_a_configs, output_dir, cache_suffix, annotations, "Phase A")
    all_rows.extend(phase_a_rows)

    # Determine best β_max from Phase A
    best_a = phase_a_rows[0] if phase_a_rows else None
    best_bmax = best_a["bm25_bonus_max"] if best_a else 0.15
    second_best_a = phase_a_rows[1] if len(phase_a_rows) > 1 else None
    second_bmax = second_best_a["bm25_bonus_max"] if second_best_a else 0.10
    print(f"\n  → Phase A best β_max = {best_bmax:.2f}")

    # ══════════════════════════════════════════════════════════════
    # Phase B: Semantic/Attribute weight sweep (fixed β_max=0.15)
    # ══════════════════════════════════════════════════════════════
    # (0.3, 0.7) is already covered by A4, skip it
    phase_b_configs = []
    for ws, wa in [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4)]:
        label = f"B_s{int(ws*10)}a{int(wa*10)}"
        phase_b_configs.append({
            "phase": "PhaseB",
            "name": label,
            "semantic_weight": ws,
            "attribute_weight": wa,
            "bm25_bonus_max": 0.15,
        })

    print("\n" + "█" * 60)
    print("  PHASE B: Semantic / Attribute Weight Sweep")
    print("  Fixed: β_max=0.15")
    print("  Sweep: (W_s, W_a) ∈ {(0.2,0.8), (0.4,0.6), (0.5,0.5), (0.6,0.4)}")
    print("  Note: (0.3,0.7) reuses Phase A result A_bonus015")
    print("█" * 60)

    for cfg in phase_b_configs:
        run_config(cfg, queries, output_dir, cache_suffix)

    # Include A4 (0.3/0.7/0.15) in Phase B evaluation for comparison
    phase_b_eval_configs = list(phase_b_configs)
    a4_cfg = next((c for c in phase_a_configs if c["bm25_bonus_max"] == 0.15), None)
    if a4_cfg:
        phase_b_eval_configs.append(a4_cfg)

    phase_b_rows = evaluate_phase(phase_b_eval_configs, output_dir, cache_suffix, annotations, "Phase B")
    # Only add the new B configs to all_rows (A4 is already there)
    for r in phase_b_rows:
        if r["phase"] == "PhaseB":
            all_rows.append(r)

    # Determine best (W_s, W_a) from Phase B
    best_b = phase_b_rows[0] if phase_b_rows else None
    best_ws = best_b["semantic_weight"] if best_b else 0.30
    best_wa = best_b["attribute_weight"] if best_b else 0.70
    second_best_b = phase_b_rows[1] if len(phase_b_rows) > 1 else None
    second_ws = second_best_b["semantic_weight"] if second_best_b else 0.40
    second_wa = second_best_b["attribute_weight"] if second_best_b else 0.60
    print(f"\n  → Phase B best (W_s, W_a) = ({best_ws:.2f}, {best_wa:.2f})")

    # ══════════════════════════════════════════════════════════════
    # Phase C: Cross-validation of top combos
    # ══════════════════════════════════════════════════════════════
    # Generate 3 cross-validation configs from A & B bests
    cross_combos = set()

    def _add_cross(ws, wa, bmax):
        key = (ws, wa, bmax)
        # Check if this combo was already run in A or B
        for existing in phase_a_configs + phase_b_configs:
            if (existing["semantic_weight"] == ws and
                existing["attribute_weight"] == wa and
                existing["bm25_bonus_max"] == bmax):
                return  # Already exists
        cross_combos.add(key)

    # C1: Best weights from B × Best β from A
    _add_cross(best_ws, best_wa, best_bmax)
    # C2: Second-best weights from B × Best β from A
    _add_cross(second_ws, second_wa, best_bmax)
    # C3: Best weights from B × Second-best β from A
    _add_cross(best_ws, best_wa, second_bmax)

    phase_c_configs = []
    for i, (ws, wa, bmax) in enumerate(sorted(cross_combos), 1):
        label = f"C{i}_s{int(ws*10)}a{int(wa*10)}_b{int(bmax*100):03d}"
        phase_c_configs.append({
            "phase": "PhaseC",
            "name": label,
            "semantic_weight": ws,
            "attribute_weight": wa,
            "bm25_bonus_max": bmax,
        })

    if phase_c_configs:
        print("\n" + "█" * 60)
        print("  PHASE C: Cross-Validation")
        print(f"  {len(phase_c_configs)} new combos from Phase A × Phase B bests")
        print("█" * 60)

        for cfg in phase_c_configs:
            run_config(cfg, queries, output_dir, cache_suffix)

        phase_c_rows = evaluate_phase(phase_c_configs, output_dir, cache_suffix, annotations, "Phase C")
        all_rows.extend(phase_c_rows)
    else:
        print("\n[Phase C] All cross-combos already covered by A/B — skipping.")

    # ══════════════════════════════════════════════════════════════
    # Final combined summary
    # ══════════════════════════════════════════════════════════════
    print_final_summary(all_rows)

    # ── Write results ──────────────────────────────────────────────
    fieldnames = [
        "phase", "run_file", "semantic_weight", "attribute_weight",
        "bm25_bonus_max", "n_queries", "mean_ndcg_at_10",
        "mean_avg_at_10", "mean_strict_at_10",
    ]

    results_csv = output_dir / "3phase_sweep_results.csv"
    all_rows.sort(key=lambda r: r["mean_ndcg_at_10"], reverse=True)
    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    results_json = output_dir / "3phase_sweep_results.json"
    results_json.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[Done] Output directory: {output_dir.as_posix()}")
    print(f"[Done] CSV:  {results_csv.as_posix()}")
    print(f"[Done] JSON: {results_json.as_posix()}")


if __name__ == "__main__":
    main()
