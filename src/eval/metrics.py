import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.eval.paths import resolve_annotation_path
from src.eval.pool_data import load_experiment_pool
from src.eval.tag_rules import apply_hard_filters, score_required_tags


def calculate_set_quality(scores: List[float], good_threshold: float = 2.0) -> Dict[str, float]:
    if not scores:
        return {"avg": 0.0, "good_rate": 0.0, "strong_rate": 0.0, "best": 0.0}

    total = len(scores)
    avg_score = sum(scores) / total
    good_rate = sum(1 for score in scores if score >= good_threshold) / total
    strong_rate = sum(1 for score in scores if score >= 3.0) / total
    best_score = max(scores)
    return {
        "avg": avg_score,
        "good_rate": good_rate,
        "strong_rate": strong_rate,
        "best": best_score,
    }


def _run_family(engine_name: str) -> str:
    if "_run" not in engine_name:
        return engine_name
    prefix, _, suffix = engine_name.rpartition("_run")
    return prefix if suffix.isdigit() else engine_name


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _candidate_metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "words_total": candidate.get("words_total", 0),
        "publish_status": candidate.get("publish_status", ""),
        "tags": candidate.get("tags", []),
        "is_animated": False,
    }


def _resolve_candidate_score(
    base_score: float,
    golden_rules: Dict[str, Any],
    candidate: Dict[str, Any],
    use_strict_filter: bool,
    strict_only: bool,
) -> float:
    if not use_strict_filter and not strict_only:
        return base_score

    if not golden_rules:
        return 0.0 if strict_only else base_score

    metadata = _candidate_metadata(candidate)
    if not apply_hard_filters(golden_rules, metadata):
        return 0.0

    required_tags = golden_rules.get("required_tags") or []
    if strict_only:
        if not required_tags:
            return 3.0
        strict_score, _, _, _ = score_required_tags(golden_rules, metadata)
        return strict_score

    if not required_tags:
        return base_score

    strict_score, _, _, _ = score_required_tags(golden_rules, metadata)
    return strict_score


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


def _print_mode_summary(
    mode_name: str,
    engine_query_quality: Dict[str, List[Dict[str, float]]],
    show_detail: bool = True,
) -> None:
    summary = []
    for engine_name, query_scores in engine_query_quality.items():
        if query_scores:
            avg_score = sum(item["avg"] for item in query_scores) / len(query_scores)
            good_rate = sum(item["good_rate"] for item in query_scores) / len(query_scores)
            strong_rate = sum(item["strong_rate"] for item in query_scores) / len(query_scores)
            best_score = sum(item["best"] for item in query_scores) / len(query_scores)
        else:
            avg_score = 0.0
            good_rate = 0.0
            strong_rate = 0.0
            best_score = 0.0
        summary.append((engine_name, avg_score, good_rate, strong_rate, best_score))

    summary.sort(key=lambda item: item[1], reverse=True)

    if show_detail:
        print("\n" + "-" * 40)
        print(f"{mode_name} Summary")
        print("-" * 40)
        for engine_name, avg_score, good_rate, strong_rate, best_score in summary:
            print(
                f"  {engine_name:20s} | Avg@10: {avg_score:.4f}"
                f" | Good@10: {good_rate:.1%}"
                f" | Strong@10: {strong_rate:.1%}"
                f" | Best@10: {best_score:.4f}"
            )

    family_summary = defaultdict(list)
    for engine_name, avg_score, good_rate, strong_rate, best_score in summary:
        family_summary[_run_family(engine_name)].append(
            {
                "avg_score": avg_score,
                "good_rate": good_rate,
                "strong_rate": strong_rate,
                "best_score": best_score,
            }
        )

    grouped_families = dict(family_summary)
    repeated_families = {family: runs for family, runs in grouped_families.items() if len(runs) > 1}
    families_to_print = grouped_families if not show_detail else repeated_families

    if families_to_print:
        if show_detail:
            print("\n" + "-" * 40)
            print(f"{mode_name} Grouped Summary")
            print("-" * 40)
        else:
            print(mode_name)

        for family, runs in sorted(
            families_to_print.items(),
            key=lambda item: _mean([run["avg_score"] for run in item[1]]),
            reverse=True,
        ):
            avg_scores = [item["avg_score"] for item in runs]
            good_rates = [item["good_rate"] for item in runs]
            strong_rates = [item["strong_rate"] for item in runs]
            best_scores = [item["best_score"] for item in runs]
            if show_detail:
                print(
                    f"  {family:20s} | Avg@10: {_mean(avg_scores):.4f} +/- {_stdev(avg_scores):.4f}"
                    f" | Good@10: {_mean(good_rates):.1%} +/- {_stdev(good_rates):.1%}"
                    f" | Strong@10: {_mean(strong_rates):.1%} +/- {_stdev(strong_rates):.1%}"
                    f" | Best@10: {_mean(best_scores):.4f} +/- {_stdev(best_scores):.4f}"
                    f" | Runs: {len(runs)}"
                )
            else:
                print(
                    f"  {family:20s} | Avg {_mean(avg_scores):.4f}+/-{_stdev(avg_scores):.4f}"
                    f" | Good {_mean(good_rates):.1%}+/-{_stdev(good_rates):.1%}"
                    f" | Strong {_mean(strong_rates):.1%}+/-{_stdev(strong_rates):.1%}"
                    f" | Runs {len(runs)}"
                )
    elif not show_detail:
        print(f"{mode_name}\n  (No grouped runs)")


def run_evaluation(
    experiment_dir: str,
    experiment_name: Optional[str] = None,
    annotations_dir: str = "data/experiments/annotations",
    group_only: bool = False,
) -> None:
    pooled_queries = load_experiment_pool(experiment_dir)
    annotation_path = resolve_annotation_path(annotations_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    annotations = _load_annotations(annotation_path)
    label = experiment_name or Path(experiment_dir).name

    print("\n" + "=" * 40)
    print("Experiment Evaluation")
    print("=" * 40)
    print(f"Experiment: {label}")
    print(f"Run directory: {Path(experiment_dir)}")
    print(f"Annotation file: {annotation_path}")

    modes = [
        ("no-strict", False, False),
        ("strict-only", True, True),
    ]
    for mode_name, use_strict_filter, strict_only in modes:
        engine_query_quality: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for pooled_query in pooled_queries:
            query_id = pooled_query["query_id"]
            golden_rules = pooled_query.get("golden_rules", {})
            engine_results: Dict[str, List[float]] = defaultdict(list)

            for candidate in pooled_query["candidates"]:
                book_id = str(candidate.get("book_id", ""))
                base_score = annotations.get(query_id, {}).get(book_id, 0.0)
                score = _resolve_candidate_score(
                    base_score=base_score,
                    golden_rules=golden_rules,
                    candidate=candidate,
                    use_strict_filter=use_strict_filter,
                    strict_only=strict_only,
                )

                for engine_name in candidate.get("original_ranks", {}):
                    engine_results[engine_name].append(score)

            for engine_name, scores in engine_results.items():
                engine_query_quality[engine_name].append(calculate_set_quality(scores))

        _print_mode_summary(mode_name, engine_query_quality, show_detail=not group_only)

    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate run quality from pooled run data")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/runs",
        help="Directory containing run JSON files",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional label shown in output",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/experiments/annotations",
        help="Directory containing the shared annotation CSV",
    )
    parser.add_argument(
        "--group-only",
        action="store_true",
        help="Only show grouped summary output",
    )
    args = parser.parse_args()

    run_evaluation(
        experiment_dir=args.experiment_dir,
        experiment_name=args.experiment,
        annotations_dir=args.annotations_dir,
        group_only=args.group_only,
    )
