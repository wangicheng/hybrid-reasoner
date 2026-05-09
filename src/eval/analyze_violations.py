import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.eval.tag_rules import normalize_tags, tag_matches


# Fix Windows console encoding issues.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def resolve_run_dir(experiment_dir: Path) -> Path:
    """Return the directory that actually contains raw run JSON files."""
    return experiment_dir


def get_experiment_group_name(run_name: str) -> str:
    """Group repeated runs like exp_name_run01 under the same experiment key."""
    return re.sub(r"_run\d+$", "", run_name)


def check_rules(golden_rules, book):
    violations = []

    # Check words
    words_total = book.get("words_total", 0)
    min_words = golden_rules.get("min_words")
    max_words = golden_rules.get("max_words")
    if min_words is not None and words_total < min_words:
        violations.append(f"Words too few: {words_total} < {min_words}")
    if max_words is not None and words_total > max_words:
        violations.append(f"Words too many: {words_total} > {max_words}")

    # Check status
    req_status = golden_rules.get("required_status")
    if req_status:
        status = str(book.get("publish_status", "")).lower()
        if req_status == "completed":
            if status not in ["completed", "completed_status", "finished", "done", "已完結", "完結"]:
                violations.append(f"Status mismatch: expected completed, got {status}")
        elif req_status == "ongoing":
            if status not in ["ongoing", "in_progress", "running", "連載中", "連載"]:
                violations.append(f"Status mismatch: expected ongoing, got {status}")

    # Check animated
    must_be_animated = golden_rules.get("must_be_animated")
    if must_be_animated is not None:
        is_animated = bool(book.get("is_animated", False))
        if is_animated != bool(must_be_animated):
            violations.append(
                f"Animation mismatch: expected {must_be_animated}, got {is_animated}"
            )

    # Check required tags
    req_tags = golden_rules.get("required_tags") or []
    if req_tags:
        book_tags = normalize_tags(book.get("tags", []))
        for rt in req_tags:
            if not any(tag_matches(rt, bt) for bt in book_tags):
                violations.append(f"Missing required tag: {rt}")

    # Check blocked tags
    blocked_tags = golden_rules.get("blocked_tags") or []
    if blocked_tags:
        book_tags = normalize_tags(book.get("tags", []))
        for bt in blocked_tags:
            if any(tag_matches(bt, b_tag) for b_tag in book_tags):
                violations.append(f"Contains blocked tag: {bt}")

    return violations


def main(experiment_dir: str = "data/experiments/runs"):
    print("Loading experiment data and ground truth...")

    # Load queries
    queries_path = Path("data/experiments/queries.json")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    query_map = {q["id"]: q for q in queries}

    # Load books for full metadata if possible
    books_path = Path("data/books_crawled.json")
    books_db = {}
    if books_path.exists():
        with open(books_path, "r", encoding="utf-8") as f:
            for b in json.load(f):
                books_db[str(b.get("id"))] = b

    run_root = Path(experiment_dir)
    run_dir = resolve_run_dir(run_root)
    run_files = sorted(run_dir.glob("*.json"))

    if not run_files:
        raise FileNotFoundError(
            f"No run JSON files found under {run_root}"
        )

    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total_samples = defaultdict(lambda: defaultdict(int))
    violation_details = defaultdict(lambda: defaultdict(list))
    group_runs = defaultdict(list)

    print(f"Found {len(run_files)} run files in {run_dir}")

    for run_file in run_files:
        run_name = run_file.stem
        exp_name = get_experiment_group_name(run_name)
        group_runs[exp_name].append(run_name)
        with open(run_file, "r", encoding="utf-8") as f:
            try:
                run_data = json.load(f)
            except json.JSONDecodeError:
                print(f"  Skipping invalid JSON file: {run_file.name}")
                continue

        for query_run in run_data:
            q_id = query_run.get("query_id")
            q_conf = query_map.get(q_id)
            if not q_conf:
                q_text = query_run.get("query")
                q_conf = next((q for q in queries if q["query"] == q_text), None)

            if not q_conf:
                continue

            golden_rules = q_conf.get("golden_rules", {})
            query_label = q_conf.get("query", query_run.get("query", ""))
            results = query_run.get("results", [])[:10]

            for rank, res in enumerate(results):
                total_samples[exp_name][q_id] += 1
                b_id = str(res.get("book_id"))

                # Use DB metadata if available, otherwise fallback to run result data
                book_meta = books_db.get(b_id, res)

                violations = check_rules(golden_rules, book_meta)
                if violations:
                    for v in violations:
                        summary[exp_name][q_id][v] += 1

                    if len(violation_details[exp_name][q_id]) < 20:
                        violation_details[exp_name][q_id].append(
                            {
                                "run": run_name,
                                "query_id": q_id,
                                "query": query_label,
                                "rank": rank + 1,
                                "book": res.get("title", b_id),
                                "violations": violations,
                            }
                        )

    # Output results
    print("\n" + "=" * 60)
    print("Hard Constraint Violation Analysis")
    print("=" * 60)

    sorted_exps = sorted(summary.keys())

    for exp in sorted_exps:
        query_ids = sorted(summary[exp].keys())
        total_v = sum(sum(v_counts.values()) for v_counts in summary[exp].values())
        samples = sum(total_samples[exp].values())

        print(f"\nExperiment: {exp}")
        print(f"Runs merged: {', '.join(group_runs[exp])}")
        print(f"Top-10 samples: {samples} | Total violations: {total_v}")

        if total_v == 0:
            print("  No violations found.")
        else:
            ranked_violations = []
            for q_id in query_ids:
                query_samples = total_samples[exp][q_id]
                for v_msg, count in summary[exp][q_id].items():
                    percentage = (count / query_samples) * 100 if query_samples else 0.0
                    ranked_violations.append((count, percentage, q_id, v_msg))

            ranked_violations.sort(key=lambda item: (-item[0], item[2], item[3]))

            for count, percentage, q_id, v_msg in ranked_violations:
                print(
                    f"  - [{count:3d} ({percentage:5.1f}%)] Query {q_id}: {v_msg}"
                )

    # Summary Ranking
    print("\n" + "=" * 60)
    print("Violation Summary Ranking")
    print("=" * 60)
    
    ranking_data = []
    for exp in sorted_exps:
        total_v = sum(sum(v_counts.values()) for v_counts in summary[exp].values())
        blocked_tag_v = 0
        for q_id in summary[exp]:
            for v_msg, count in summary[exp][q_id].items():
                if v_msg.startswith("Contains blocked tag:"):
                    blocked_tag_v += count
        
        ranking_data.append({
            "exp": exp,
            "total": total_v,
            "blocked": blocked_tag_v
        })
    
    # Sort by total violations ascending (lower is better)
    ranking_data.sort(key=lambda x: (x["total"], x["blocked"]))
    
    print(f"{'Experiment':<45} | {'Total':<7} | {'Blocked':<7}")
    print("-" * 65)
    for row in ranking_data:
        print(f"{row['exp']:<45} | {row['total']:<7} | {row['blocked']:<7}")

    print("\n" + "=" * 60)
    print("Analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze hard constraint violations from run files"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/runs/batch_YYYYMMDD_HHMMSS",
        help="Batch directory containing run JSON files",
    )
    args = parser.parse_args()

    main(experiment_dir=args.experiment_dir)
