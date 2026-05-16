"""Run a single dual_track experiment to capture tag mapping details into the run JSON.

This script invokes the existing RunGenerator but forces `rerank=False` to reduce extra LLM calls
and saves the run output in `data/experiments/runs/batch_tagmap_<timestamp>/gemma4_dual_track_tagmap.json`.
"""
from datetime import datetime
from pathlib import Path
import json

from src.eval.generate_run import RunGenerator
from src.core.model_catalog import normalize_model_id


def main():
    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        return

    with open(queries_path, "r", encoding="utf-8") as f:
        sample_queries = json.load(f)

    model_id = normalize_model_id("gemma-4-31b-it")
    output_root = Path("data/experiments/runs")
    batch_name = datetime.now().strftime("batch_tagmap_%Y%m%d_%H%M%S")
    output_folder = output_root / batch_name
    print(f"Batch output directory: {output_folder}")

    generator = RunGenerator(k_per_engine=10, model_id=model_id, rerank=False)
    generator.generate_run(
        queries_config=sample_queries,
        engine_name="gemma4_dual_track_tagmap",
        output_dir=output_folder,
        semantic_weight=0.4,
        attribute_weight=0.6,
        run_suffix="",
        enable_bm25=False,
        bm25_weight=0.3,
        use_schema_constraint=True,
        disable_tag_embedding=False,
        max_tags_per_term=3,
    )

    print("Run finished. Inspect the JSON in the output folder.")


if __name__ == "__main__":
    main()
