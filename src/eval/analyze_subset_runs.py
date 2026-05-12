import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.eval.metrics import (
    _mean,
    _resolve_candidate_score,
    _stdev,
    calculate_set_quality,
)
from src.eval.paths import resolve_annotation_path
from src.eval.pool_data import load_queries


def _load_run_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Run file must contain a list: {path}")
    return data


def _load_annotations(annotation_path: Path) -> Dict[str, Dict[str, float]]:
    annotations: Dict[str, Dict[str, float]] = {}
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


def _extract_run_metadata(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for row in rows:
        metadata = row.get("run_metadata", {})
        if isinstance(metadata, dict) and metadata:
            return metadata
    return {}


def _evaluate_run(
    rows: List[Dict[str, Any]],
    queries_map: Dict[str, Dict[str, Any]],
    annotations: Dict[str, Dict[str, float]],
    *,
    use_strict_filter: bool,
    strict_only: bool,
) -> Dict[str, float]:
    query_scores: List[Dict[str, float]] = []
    query_successes = 0
    empty_results = 0

    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            continue

        if "error" not in row:
            query_successes += 1

        results = row.get("results", []) or []
        if not results:
            empty_results += 1

        query_conf = queries_map.get(query_id, {})
        golden_rules = query_conf.get("golden_rules", {}) if isinstance(query_conf, dict) else {}

        candidate_scores: List[float] = []
        for result in results:
            book_id = str(result.get("book_id", "")).strip()
            base_score = annotations.get(query_id, {}).get(book_id, 0.0)
            candidate_scores.append(
                _resolve_candidate_score(
                    base_score=base_score,
                    golden_rules=golden_rules,
                    candidate=result,
                    use_strict_filter=use_strict_filter,
                    strict_only=strict_only,
                )
            )

        query_scores.append(calculate_set_quality(candidate_scores))

    avg_at_10 = _mean([item["avg"] for item in query_scores])
    good_rate = _mean([item["good_rate"] for item in query_scores])
    strong_rate = _mean([item["strong_rate"] for item in query_scores])
    best_at_10 = _mean([item["best"] for item in query_scores])
    total_queries = len(rows) or 1

    return {
        "avg_at_10": avg_at_10,
        "good_rate": good_rate,
        "strong_rate": strong_rate,
        "best_at_10": best_at_10,
        "query_success_rate": query_successes / total_queries,
        "empty_result_rate": empty_results / total_queries,
    }


def analyze_subset_runs(
    experiment_dir: str,
    queries_path: str,
    annotations_dir: str,
) -> None:
    run_dir = Path(experiment_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing experiment directory: {run_dir}")

    queries = load_queries(Path(queries_path))
    queries_map = {str(item["id"]): item for item in queries}
    annotation_path = resolve_annotation_path(annotations_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    annotations = _load_annotations(annotation_path)

    grouped_metrics: Dict[Tuple[str, int, str], List[Dict[str, float]]] = defaultdict(list)
    run_files = sorted(run_dir.glob("*.json"))
    if not run_files:
        raise FileNotFoundError(f"No run JSON files found in {run_dir}")

    for run_file in run_files:
        try:
            rows = _load_run_file(run_file)
        except ValueError:
            print(f"Skipping non-run JSON file: {run_file.name}")
            continue
        run_metadata = _extract_run_metadata(rows)
        engine_type = str(rows[0].get("engine_type", "") or run_metadata.get("engine_type", "")).strip()
        if not engine_type:
            engine_type = "unknown"
        subset_size = int(run_metadata.get("subset_size", 0) or 0)
        if subset_size <= 0:
            print(f"Skipping {run_file.name}: missing subset_size metadata")
            continue

        for mode_name, use_strict_filter, strict_only in [
            ("no-strict", False, False),
            ("strict-only", True, True),
        ]:
            grouped_metrics[(engine_type, subset_size, mode_name)].append(
                _evaluate_run(
                    rows,
                    queries_map,
                    annotations,
                    use_strict_filter=use_strict_filter,
                    strict_only=strict_only,
                )
            )

    for mode_name in ["no-strict", "strict-only"]:
        print("\n" + "=" * 72)
        print(f"{mode_name} subset summary")
        print("=" * 72)

        mode_rows: List[Tuple[str, int, List[Dict[str, float]]]] = []
        for (engine_type, subset_size, grouped_mode), metrics_list in grouped_metrics.items():
            if grouped_mode != mode_name:
                continue
            mode_rows.append((engine_type, subset_size, metrics_list))

        for engine_type, subset_size, metrics_list in sorted(
            mode_rows,
            key=lambda item: (item[1], item[0]),
        ):
            avg_values = [item["avg_at_10"] for item in metrics_list]
            success_values = [item["query_success_rate"] for item in metrics_list]
            empty_values = [item["empty_result_rate"] for item in metrics_list]
            print(
                f"{engine_type:18s} size={subset_size:4d} "
                f"| Avg@10 {_mean(avg_values):.4f}+/-{_stdev(avg_values):.4f} "
                f"| Success {_mean(success_values):.1%}+/-{_stdev(success_values):.1%} "
                f"| Empty {_mean(empty_values):.1%}+/-{_stdev(empty_values):.1%} "
                f"| Runs {len(metrics_list)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate subset-size comparison runs")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing subset run JSON files",
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="data/experiments/queries.json",
        help="Path to query config JSON",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/experiments/annotations",
        help="Directory containing annotated.csv",
    )
    args = parser.parse_args()
    analyze_subset_runs(
        experiment_dir=args.experiment_dir,
        queries_path=args.queries_path,
        annotations_dir=args.annotations_dir,
    )


if __name__ == "__main__":
    main()
