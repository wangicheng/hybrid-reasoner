"""
Dynamic Routing A/B Test

Compares:
  1. auto (dynamic routing) — the new intent-aware architecture
  2. weighted_ws35_wa65 — best strict baseline
  3. weighted_ws60_wa40 — best no-strict baseline
  4. rrf_k60_no_bm25 — best RRF baseline

All runs share the same LLM parse cache for fair comparison.
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

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_dynamic_routing")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = "_sameparse"

    configs = [
        # ── Dynamic Routing (the new architecture) ──
        {
            "name": "auto_dynamic_routing",
            "fusion_strategy": "auto",
            "rrf_k": 60,
            "semantic_weight": 0.5,   # fallback defaults (overridden per-query)
            "attribute_weight": 0.5,
            "enable_bm25": True,      # lexical_store always initialized for auto
            "bm25_bonus_max": 0.0,
        },
        # ── Static baselines for comparison ──
        {
            "name": "weighted_ws35_wa65",
            "fusion_strategy": "weighted",
            "rrf_k": 60,
            "semantic_weight": 0.35,
            "attribute_weight": 0.65,
            "enable_bm25": True,
            "bm25_bonus_max": 0.0,
        },
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
            "name": "rrf_k60_no_bm25",
            "fusion_strategy": "rrf",
            "rrf_k": 60,
            "semantic_weight": 0.5,
            "attribute_weight": 0.5,
            "enable_bm25": False,
            "bm25_bonus_max": 0.0,
        },
    ]

    print(f"\n{'█' * 60}")
    print("  Dynamic Routing A/B Test")
    print(f"  {len(configs)} configurations")
    print(f"  Output: {output_dir}")
    print(f"{'█' * 60}")

    for cfg in configs:
        print(f"\n{'=' * 60}")
        print(f"  {cfg['name']}  fusion={cfg['fusion_strategy']}")
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

    print(f"\n{'#' * 60}")
    print("  Batch complete!")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Run evaluation with:")
    print(f"    python -m src.eval.llm_judge --experiment-dir {output_dir}")
    print(f"    python -m src.eval.metrics --experiment-dir {output_dir}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
