import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.core.database import Database
from src.core.model_catalog import normalize_model_id, sanitize_model_tag
from src.eval.generate_run import RunGenerator
from src.eval.pool_data import load_queries
from src.eval.subset_manifest import (
    build_subset_manifest,
    load_subset_manifest,
    resolve_subset_book_ids,
    save_subset_manifest,
)


DEFAULT_SUBSET_SIZES = [50, 100, 250, 500, 1000]


def parse_subset_sizes(raw_value: str) -> List[int]:
    sizes: List[int] = []
    for part in str(raw_value or "").split(","):
        text = part.strip()
        if not text:
            continue
        size = int(text)
        if size <= 0:
            continue
        sizes.append(size)
    deduped = sorted(set(sizes))
    if not deduped:
        raise ValueError("subset_sizes must contain at least one positive integer")
    return deduped


def build_or_load_manifest(
    *,
    manifest_path: Path,
    base_seed: int,
    repeats: int,
    book_ids: List[str],
) -> Dict[str, Any]:
    if manifest_path.exists():
        print(f"Loading existing subset manifest: {manifest_path}")
        return load_subset_manifest(manifest_path)

    manifest = build_subset_manifest(
        book_ids=book_ids,
        base_seed=base_seed,
        repeats=repeats,
    )
    save_subset_manifest(manifest, manifest_path)
    print(f"Created subset manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subset-size comparison runs")
    parser.add_argument(
        "--subset-sizes",
        type=str,
        default=",".join(str(size) for size in DEFAULT_SUBSET_SIZES),
        help="Comma-separated subset sizes, e.g. 50,100,250,500,1000",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated random permutations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for subset permutation generation",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gemma-4-31b-it",
        help="Model id shared by both engines",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/subset_runs",
        help="Root directory for generated subset run batches",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="Optional explicit manifest path. If omitted, use a seed-based default path.",
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="data/experiments/queries.json",
        help="Path to query config JSON",
    )
    args = parser.parse_args()

    subset_sizes = parse_subset_sizes(args.subset_sizes)
    repeats = max(1, args.repeats)
    model_id = normalize_model_id(args.model_id)

    queries_path = Path(args.queries_path)
    queries_config = load_queries(queries_path)

    db = Database()
    all_items = db.get_all_items()
    all_book_ids = [
        str(item.get("id", "")).strip()
        for item in all_items
        if str(item.get("id", "")).strip()
    ]
    if not all_book_ids:
        raise RuntimeError("No books found in database; cannot generate subset runs.")

    max_subset_size = max(subset_sizes)
    if max_subset_size > len(all_book_ids):
        raise ValueError(
            f"Requested subset size {max_subset_size} exceeds catalog size {len(all_book_ids)}"
        )

    manifest_path = (
        Path(args.manifest_path)
        if str(args.manifest_path).strip()
        else Path("data/experiments/subsets") / f"subset_manifest_seed{args.seed}_r{repeats}.json"
    )
    manifest = build_or_load_manifest(
        manifest_path=manifest_path,
        base_seed=args.seed,
        repeats=repeats,
        book_ids=all_book_ids,
    )

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    model_tag = sanitize_model_tag(model_id)
    output_dir = Path(args.experiment_dir) / f"{batch_name}_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_snapshot_path = output_dir / "subset_manifest.json"
    save_subset_manifest(manifest, manifest_snapshot_path)

    print(f"Subset batch output directory: {output_dir}")
    print(
        json.dumps(
            {
                "model_id": model_id,
                "catalog_size": len(all_book_ids),
                "subset_sizes": subset_sizes,
                "repeats": repeats,
                "seed": args.seed,
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    experiments = [
        # {
        #     "name_prefix": "hybrid",
        #     "engine_type": "hybrid",
        # },
        # {
        #     "name_prefix": "single_prompt",
        #     "engine_type": "single_prompt_llm",
        # },
        {
            "name_prefix": "hybrid_rerank",
            "engine_type": "hybrid_rerank",
        },
    ]

    generator = RunGenerator(
        k_per_engine=10,
        model_id=model_id,
    )

    for repeat_index in range(1, repeats + 1):
        print(f"\n=== Subset Repeat {repeat_index}/{repeats} ===")
        for subset_size in subset_sizes:
            subset_book_ids = resolve_subset_book_ids(
                manifest,
                repeat_index=repeat_index,
                subset_size=subset_size,
            )
            allowed_book_ids = set(subset_book_ids)
            subset_seed = args.seed + repeat_index - 1
            subset_id = f"seed{args.seed}_size{subset_size:04d}_run{repeat_index:02d}"

            print(
                f"\n--- subset_size={subset_size} repeat={repeat_index} "
                f"subset_seed={subset_seed} books={len(allowed_book_ids)} ---"
            )

            for exp in experiments:
                engine_name = (
                    f"{exp['name_prefix']}_size{subset_size:04d}_run{repeat_index:02d}"
                )
                generator.generate_run(
                    queries_config=queries_config,
                    engine_name=engine_name,
                    output_dir=output_dir,
                    engine_type=exp["engine_type"],
                    semantic_weight=0.4,
                    attribute_weight=0.6,
                    engine_kwargs={
                        "allowed_book_ids": allowed_book_ids,
                    },
                    run_metadata={
                        "experiment_kind": "subset_size_comparison",
                        "subset_size": subset_size,
                        "repeat_index": repeat_index,
                        "subset_seed": subset_seed,
                        "subset_id": subset_id,
                        "catalog_size_before_subset": len(all_book_ids),
                        "catalog_size_after_subset": len(allowed_book_ids),
                        "manifest_path": str(manifest_snapshot_path),
                        "model_id": model_id,
                    },
                )

    print("\nSubset-size experiments finished!")


if __name__ == "__main__":
    main()
