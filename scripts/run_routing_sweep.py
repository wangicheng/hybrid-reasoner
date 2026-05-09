"""
Dynamic Routing Parameter Sweep

Based on the first-round data insights:
  - Strict-only: auto (2.5192) > ws35_wa65 (2.4775) > ws60_wa40 (2.3385) > rrf (2.0135)
  - No-strict:   ws35_wa65 (2.4350) > auto (2.3538) > rrf (2.3208) > ws60_wa40 (2.2542)

Key observations driving this experiment design:
  1. Gemma 4 31B favours high attribute weight (ws35_wa65) even in no-strict scenarios
  2. auto succeeded in strict-only because it correctly routed constraint-heavy queries
  3. RRF without BM25 still wins for pure-atmosphere queries
  4. The routing threshold (pos_tags >= 2) may be too conservative

Experiment axes:
  A. Tag threshold: 1 vs 2 vs 3  (when does the router switch to weighted?)
  B. Weighted-path weights: ws25_wa75 / ws35_wa65 / ws45_wa55
  C. RRF-path BM25: on vs off
  D. Control baselines: best static weighted + best static RRF
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.eval.generate_run import RunGenerator


def main() -> None:
    queries_path = Path("data/experiments/queries.json")
    output_root = Path("data/experiments/runs")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_routing_sweep")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_sameparse"

    # ── All configs ──
    configs = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Axis A: Tag threshold sweep (fix weights at best: ws35_wa65)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for threshold in [1, 2, 3]:
        configs.append({
            "name": f"auto_t{threshold}_ws35wa65",
            "fusion_strategy": "auto",
            "rrf_k": 60,
            "routing_tag_threshold": threshold,
            "routing_weighted_ws": 0.35,
            "routing_weighted_wa": 0.65,
            "routing_weighted_bm25": True,
            "routing_rrf_bm25": False,
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Axis B: Weighted-path weight sweep (fix threshold at 2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for ws, wa in [(0.25, 0.75), (0.45, 0.55)]:
        configs.append({
            "name": f"auto_t2_ws{int(ws*100)}wa{int(wa*100)}",
            "fusion_strategy": "auto",
            "rrf_k": 60,
            "routing_tag_threshold": 2,
            "routing_weighted_ws": ws,
            "routing_weighted_wa": wa,
            "routing_weighted_bm25": True,
            "routing_rrf_bm25": False,
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Axis C: RRF-path BM25 toggle (fix threshold=2, weights=35/65)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    configs.append({
        "name": "auto_t2_ws35wa65_rrf_bm25on",
        "fusion_strategy": "auto",
        "rrf_k": 60,
        "routing_tag_threshold": 2,
        "routing_weighted_ws": 0.35,
        "routing_weighted_wa": 0.65,
        "routing_weighted_bm25": True,
        "routing_rrf_bm25": True,  # <-- key difference
    })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Axis D: Aggressive routing - threshold=1 + strong weights
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    configs.append({
        "name": "auto_t1_ws25wa75",
        "fusion_strategy": "auto",
        "rrf_k": 60,
        "routing_tag_threshold": 1,
        "routing_weighted_ws": 0.25,
        "routing_weighted_wa": 0.75,
        "routing_weighted_bm25": True,
        "routing_rrf_bm25": False,
    })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Control baselines
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    configs.append({
        "name": "baseline_ws35wa65",
        "fusion_strategy": "weighted",
        "rrf_k": 60,
        "routing_tag_threshold": 2,
        "routing_weighted_ws": 0.35,
        "routing_weighted_wa": 0.65,
        "routing_weighted_bm25": True,
        "routing_rrf_bm25": False,
        # Static overrides
        "_static_ws": 0.35,
        "_static_wa": 0.65,
        "_static_bm25": True,
    })
    configs.append({
        "name": "baseline_rrf_k60_nobm25",
        "fusion_strategy": "rrf",
        "rrf_k": 60,
        "routing_tag_threshold": 2,
        "routing_weighted_ws": 0.35,
        "routing_weighted_wa": 0.65,
        "routing_weighted_bm25": True,
        "routing_rrf_bm25": False,
        # Static overrides
        "_static_ws": 0.50,
        "_static_wa": 0.50,
        "_static_bm25": False,
    })

    print(f"\n{'█' * 60}")
    print("  Dynamic Routing Parameter Sweep")
    print(f"  {len(configs)} configurations")
    print(f"  Output: {output_dir}")
    print(f"{'█' * 60}\n")

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{len(configs)}] {cfg['name']}")
        print(f"  fusion={cfg['fusion_strategy']}  threshold={cfg['routing_tag_threshold']}")
        print(f"  weighted_path: ws={cfg['routing_weighted_ws']} wa={cfg['routing_weighted_wa']}")
        print(f"  rrf_path: bm25={cfg['routing_rrf_bm25']}")
        print(f"{'=' * 60}")

        # Determine static-level params
        static_ws = cfg.get("_static_ws", cfg["routing_weighted_ws"])
        static_wa = cfg.get("_static_wa", cfg["routing_weighted_wa"])
        static_bm25 = cfg.get("_static_bm25", True)

        generator = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
        generator.generate_run(
            queries_config=queries,
            engine_name=cfg["name"],
            output_dir=output_dir,
            semantic_weight=static_ws,
            attribute_weight=static_wa,
            run_suffix=cache_suffix,
            enable_bm25=static_bm25,
            bm25_weight=0.1,
            bm25_bonus_max=0.0,
            fusion_strategy=cfg["fusion_strategy"],
            rrf_k=cfg["rrf_k"],
            routing_tag_threshold=cfg["routing_tag_threshold"],
            routing_weighted_ws=cfg["routing_weighted_ws"],
            routing_weighted_wa=cfg["routing_weighted_wa"],
            routing_weighted_bm25=cfg["routing_weighted_bm25"],
            routing_rrf_bm25=cfg["routing_rrf_bm25"],
        )

    print(f"\n{'#' * 60}")
    print("  Batch complete!")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Run evaluation with:")
    print(f"    python -m src.eval.llm_judge --experiment-dir {output_dir}")
    print(f"    python -m src.eval.metrics --experiment-dir {output_dir}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
