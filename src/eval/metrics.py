import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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


def _build_fallback_book(row: Dict[str, str]) -> Dict[str, Any]:
    words_in_10k = float(row.get("Words (萬)") or 0)
    intro = row.get("Intro", "")

    tags: List[str] = []
    if "[標籤:" in intro:
        start_idx = intro.find("[標籤:") + 4
        end_idx = intro.find("]", start_idx)
        if end_idx != -1:
            tags_str = intro[start_idx:end_idx]
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    return {
        "words_total": words_in_10k * 10000,
        "publish_status": row.get("Status", ""),
        "tags": tags,
        "is_animated": False,
    }


def _load_books_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    books_data: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return books_data

    with path.open("r", encoding="utf-8") as f:
        try:
            crawled_data = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Failed to parse books_crawled.json")
            return books_data

    for book in crawled_data:
        book_id = str(book.get("id", "")).strip()
        if book_id:
            books_data[book_id] = book
    return books_data


def _resolve_candidate_score(
    query: str,
    book_id: str,
    base_score: float,
    books_data: Dict[str, Dict[str, Any]],
    golden_rules_map: Dict[str, Dict[str, Any]],
    use_strict_filter: bool,
    strict_only: bool,
) -> float:
    if not use_strict_filter and not strict_only:
        return base_score

    golden_rules = golden_rules_map.get(query, {})
    if not golden_rules:
        return 0.0 if strict_only else base_score

    metadata = books_data.get(book_id, {})
    if not apply_hard_filters(golden_rules, metadata):
        return 0.0

    if strict_only:
        required_tags = golden_rules.get("required_tags") or []
        if not required_tags:
            return 3.0
        strict_score, _, _, _ = score_required_tags(golden_rules, metadata)
        return strict_score

    required_tags = golden_rules.get("required_tags") or []
    if not required_tags:
        return base_score

    strict_score, _, _, _ = score_required_tags(golden_rules, metadata)
    return strict_score


def run_evaluation(
    experiment_name: str,
    use_strict_filter: bool = True,
    strict_only: bool = False,
) -> None:
    strict_mode = use_strict_filter or strict_only
    base_dir = Path("data/experiments/pools")
    annotated_csv = base_dir / f"{experiment_name}_annotated.csv"
    truth_json = base_dir / f"{experiment_name}_truth.json"

    with open("data/experiments/queries.json", "r", encoding="utf-8") as f:
        queries_config = json.load(f)
    golden_rules_map = {item["query"]: item.get("golden_rules", {}) for item in queries_config}

    books_data = _load_books_metadata(Path("data/books_crawled.json"))

    with open(truth_json, "r", encoding="utf-8") as f:
        truth_data = json.load(f)

    annotations: Dict[str, Dict[str, float]] = {}
    with annotated_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["Query"]
            book_id = row["Book ID"]
            try:
                score = float(row["Score (0-3)"])
            except ValueError:
                score = 0.0

            annotations.setdefault(query, {})[book_id] = score

            if str(book_id) not in books_data:
                books_data[str(book_id)] = _build_fallback_book(row)

    engine_query_quality = defaultdict(list)

    for truth_entry in truth_data:
        query = truth_entry["query"]
        candidates = truth_entry["candidates"]

        engine_results = defaultdict(list)

        for cand in candidates:
            book_id = str(cand["book_id"])
            base_score = annotations.get(query, {}).get(book_id, 0.0)
            score = _resolve_candidate_score(
                query=query,
                book_id=book_id,
                base_score=base_score,
                books_data=books_data,
                golden_rules_map=golden_rules_map,
                use_strict_filter=strict_mode,
                strict_only=strict_only,
            )

            for engine_name, _rank in cand["original_ranks"].items():
                engine_results[engine_name].append(score)

        for engine_name, scores in engine_results.items():
            quality = calculate_set_quality(scores)
            engine_query_quality[engine_name].append(quality)

    print("\n" + "=" * 40)
    print("Experiment Evaluation")
    print("=" * 40)
    print(f"Experiment: {experiment_name}")
    print(f"Score mode: {'strict-only' if strict_only else ('strict' if strict_mode else 'llm-only')}")
    print("-" * 40)

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

    summary.sort(key=lambda x: x[1], reverse=True)
    for engine_name, avg_score, good_rate, strong_rate, best_score in summary:
        print(
            f"  {engine_name:20s} | Avg@10: {avg_score:.4f}"
            f" | Good@10: {good_rate:.1%}"
            f" | Strong@10: {strong_rate:.1%}"
            f" | Best@10: {best_score:.4f}"
        )
    print("=" * 40 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Evaluation Metrics")
    parser.add_argument("--experiment", type=str, default="pilot_test", help="Experiment name")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict filtering")
    parser.add_argument("--strict-only", action="store_true", help="Use strict-only scoring")
    args = parser.parse_args()

    run_evaluation(
        args.experiment,
        use_strict_filter=not args.no_strict,
        strict_only=args.strict_only,
    )
