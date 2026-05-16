"""Run two generate_run experiments (dual-track k=1 and k=3) using raw mapping cutoff >= 0.7.

This script programmatically invokes RunGenerator.generate_run for two variants
and writes outputs under data/experiments/runs/batch_raw70_<TS>.
"""
from pathlib import Path
from datetime import datetime
from src.eval.generate_run import RunGenerator
from src.core.model_catalog import normalize_model_id


def main():
    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        raise SystemExit(f"Queries file not found: {queries_path}")

    batch_name = datetime.now().strftime("batch_raw70_%Y%m%d_%H%M%S")
    output_root = Path("data/experiments/runs")
    output_folder = output_root / batch_name
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Batch output directory: {output_folder}")

    with open(queries_path, "r", encoding="utf-8") as f:
        import json
        sample_queries = json.load(f)

    experiments = [
        {
            "name": "gemma4_dual_track_k1_raw70",
            "model_id": "gemma-4-31b-it",
            "disable_tag_embedding": False,
            "use_schema_constraint": True,
            "rerank": True,
            "max_tags_per_term": 1,
        },
        {
            "name": "gemma4_dual_track_k3_raw70",
            "model_id": "gemma-4-31b-it",
            "disable_tag_embedding": False,
            "use_schema_constraint": True,
            "rerank": True,
            "max_tags_per_term": 3,
        }
    ]

    for exp in experiments:
        model_id = normalize_model_id(exp.get("model_id"))
        generator = RunGenerator(k_per_engine=10, model_id=model_id, rerank=exp.get("rerank", True))
        try:
            generator.generate_run(
                queries_config=sample_queries,
                engine_name=exp["name"],
                output_dir=output_folder,
                use_schema_constraint=exp.get("use_schema_constraint", True),
                disable_tag_embedding=exp.get("disable_tag_embedding", False),
                max_tags_per_term=exp.get("max_tags_per_term", 3),
            )
        except Exception as exc:
            print(f"Experiment {exp['name']} failed: {exc}")

    print("All experiments finished.")

if __name__ == '__main__':
    main()
