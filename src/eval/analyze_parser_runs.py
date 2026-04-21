import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


BRANCH_NAMES = [
    "semantic_understanding",
    "tag_projection",
    "structured",
]


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_run_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Run file must contain a list: {path}")
    return data


def _format_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator):.1%} ({numerator}/{denominator})"


def _print_run_summary(run_name: str, rows: List[Dict[str, Any]]) -> None:
    total_queries = len(rows)
    successful_queries = [row for row in rows if "error" not in row]
    query_execution = [
        row.get("execution_metadata", {})
        for row in rows
        if isinstance(row.get("execution_metadata", {}), dict) and row.get("execution_metadata")
    ]
    query_first_attempt_successes = sum(
        1 for meta in query_execution if bool(meta.get("first_attempt_success"))
    )
    query_retry_counts = [
        float(meta.get("query_retry_count", 0) or 0)
        for meta in query_execution
    ]

    print("\n" + "=" * 60)
    print(run_name)
    print("=" * 60)
    print(f"Queries: {total_queries}")
    print(f"Query success rate: {_format_rate(len(successful_queries), total_queries)}")
    print(
        "Query first-attempt success rate: "
        f"{_format_rate(query_first_attempt_successes, len(query_execution))}"
    )
    print(f"Avg query retry count: {_safe_mean(query_retry_counts):.2f}")

    parser_modes = Counter(
        str(row.get("parse_metadata", {}).get("parser_mode", "")).strip()
        for row in rows
        if str(row.get("parse_metadata", {}).get("parser_mode", "")).strip()
    )
    if parser_modes:
        print(f"Parser modes: {dict(parser_modes)}")

    for branch_name in BRANCH_NAMES:
        branch_rows = []
        parse_sources = Counter()
        for row in rows:
            branch = row.get("parse_metadata", {}).get("branches", {}).get(branch_name)
            if isinstance(branch, (int, float)):
                branch = {
                    "success": True,
                    "latency_ms": float(branch),
                    "request_count": 0,
                    "retry_count": 0,
                    "first_attempt_success": False,
                    "parse_source": "legacy_latency_only",
                    "recovered_from_raw_text": False,
                }
            if not isinstance(branch, dict):
                continue
            branch_rows.append(branch)
            parse_source = str(branch.get("parse_source", "")).strip()
            if parse_source:
                parse_sources[parse_source] += 1

        attempted = len(branch_rows)
        successful = sum(1 for branch in branch_rows if bool(branch.get("success")))
        first_attempt_successes = sum(
            1 for branch in branch_rows if bool(branch.get("first_attempt_success"))
        )
        recovered_from_raw_text = sum(
            1 for branch in branch_rows if bool(branch.get("recovered_from_raw_text"))
        )
        retry_counts = [
            float(branch.get("retry_count", 0) or 0)
            for branch in branch_rows
        ]
        request_counts = [
            float(branch.get("request_count", 0) or 0)
            for branch in branch_rows
        ]
        latencies = [
            float(branch.get("latency_ms", 0) or 0)
            for branch in branch_rows
        ]

        print(f"\n[{branch_name}]")
        print(f"Attempted: {attempted}")
        print(f"Success rate: {_format_rate(successful, attempted)}")
        print(
            "First-attempt success rate: "
            f"{_format_rate(first_attempt_successes, attempted)}"
        )
        print(
            "Recovered from raw text: "
            f"{_format_rate(recovered_from_raw_text, attempted)}"
        )
        print(f"Avg request count: {_safe_mean(request_counts):.2f}")
        print(f"Avg retry count: {_safe_mean(retry_counts):.2f}")
        print(f"Avg latency (ms): {_safe_mean(latencies):.2f}")
        if parse_sources:
            print(f"Parse sources: {dict(parse_sources)}")


def analyze_runs(experiment_dir: str) -> None:
    run_dir = Path(experiment_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing experiment directory: {run_dir}")

    run_files = sorted(run_dir.glob("*.json"))
    if not run_files:
        raise FileNotFoundError(f"No run JSON files found in {run_dir}")

    for run_file in run_files:
        _print_run_summary(run_file.stem, _load_run_file(run_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize parser branch telemetry from run JSON files")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing run JSON files",
    )
    args = parser.parse_args()
    analyze_runs(args.experiment_dir)
