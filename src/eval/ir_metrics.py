import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from src.eval.paths import resolve_annotation_path
from src.eval.pool_data import load_queries, load_runs
from src.eval.tag_rules import normalize_tags, tag_matches


ScoreMap = Dict[str, Dict[str, float]]
ApDenominatorMode = Literal["relevant", "truncated"]


@dataclass
class EngineMetrics:
    query_count: int
    unjudged_count: int
    unjudged_rate: float
    precision_at_k: Dict[int, float]
    map_at_k: Dict[int, float]
    violation_rate_at_k: Dict[int, float]
    clean_rate_at_k: Dict[int, float]
    violation_breakdown_rate: Dict[str, float]
    # --- Fault Tolerance & Survival metrics ---
    zero_result_rate: float
    semantic_recall_at_k: Dict[int, float]
    # --- Advanced metrics ---
    penalized_ndcg_at_k: Dict[int, float]
    rbv_at_k: Dict[int, float]
    rc_f1_at_k: Dict[int, float]


def _load_annotations(annotation_path: Path) -> ScoreMap:
    annotations: ScoreMap = {}

    with annotation_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = str(row.get("Query ID", "")).strip()
            book_id = str(row.get("Book ID", "")).strip()
            if not query_id or not book_id:
                continue

            try:
                score = float(row.get("Score (0-3)", "") or 0.0)
            except ValueError:
                score = 0.0

            annotations.setdefault(query_id, {})[book_id] = score

    return annotations


def _status_is_valid(required_status: str, publish_status: str) -> bool:
    required = str(required_status or "").strip().lower()
    status = str(publish_status or "").strip().lower()

    if required == "completed":
        return status in ["completed", "completed_status", "finished", "done", "已完結", "完結"]
    if required == "ongoing":
        return status in ["ongoing", "in_progress", "running", "連載中", "連載"]
    return True


def _check_structural_violations(golden_rules: Dict[str, Any], book: Dict[str, Any]) -> List[str]:
    violations: List[str] = []

    words_total = book.get("words_total", 0)
    min_words = golden_rules.get("min_words")
    max_words = golden_rules.get("max_words")
    if min_words is not None and words_total < min_words:
        violations.append("min_words")
    if max_words is not None and words_total > max_words:
        violations.append("max_words")

    req_status = golden_rules.get("required_status")
    if req_status and not _status_is_valid(str(req_status), str(book.get("publish_status", ""))):
        violations.append("required_status")

    must_be_animated = golden_rules.get("must_be_animated")
    if must_be_animated is not None:
        is_animated = bool(book.get("is_animated", False))
        if is_animated != bool(must_be_animated):
            violations.append("must_be_animated")

    req_tags = normalize_tags(golden_rules.get("required_tags") or [])
    blocked_tags = normalize_tags(golden_rules.get("blocked_tags") or [])
    book_tags = normalize_tags(book.get("tags", []))

    for rt in req_tags:
        if not any(tag_matches(rt, bt) for bt in book_tags):
            violations.append("required_tags")
            break

    for bt in blocked_tags:
        if any(tag_matches(bt, tag) for tag in book_tags):
            violations.append("blocked_tags")
            break

    return violations


def _to_sorted_results(results: List[dict], k_max: int) -> List[dict]:
    ranked: List[Tuple[int, dict]] = []
    for idx, item in enumerate(results, start=1):
        book_id = str(item.get("book_id", "")).strip()
        if not book_id:
            continue

        rank = item.get("rank", idx)
        try:
            numeric_rank = int(rank)
        except (TypeError, ValueError):
            numeric_rank = idx

        ranked.append((numeric_rank, item))

    ranked.sort(key=lambda pair: pair[0])
    return [item for _, item in ranked[:k_max]]


def _binary_relevance(
    ranked_book_ids: Iterable[str],
    query_scores: Dict[str, float],
    relevance_threshold: float,
) -> Tuple[List[int], int]:
    binary: List[int] = []
    unjudged = 0
    for book_id in ranked_book_ids:
        if book_id not in query_scores:
            unjudged += 1
        score = query_scores.get(book_id, 0.0)
        binary.append(1 if score >= relevance_threshold else 0)
    return binary, unjudged


def _precision_at_k(binary_relevance: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(binary_relevance[:k]) / float(k)


def _recall_at_k(binary_relevance: List[int], total_relevant: int, k: int) -> float:
    if k <= 0 or total_relevant <= 0:
        return 0.0
    return sum(binary_relevance[:k]) / float(total_relevant)


def _ap_at_k(
    binary_relevance: List[int],
    total_relevant: int,
    k: int,
    denominator_mode: ApDenominatorMode,
) -> float:
    if k <= 0 or total_relevant <= 0:
        return 0.0

    hit_count = 0
    precision_sum = 0.0
    for i, rel in enumerate(binary_relevance[:k], start=1):
        if rel:
            hit_count += 1
            precision_sum += hit_count / float(i)

    if denominator_mode == "relevant":
        denom = total_relevant
    else:
        denom = min(total_relevant, k)

    return precision_sum / float(denom) if denom > 0 else 0.0


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Advanced Metric Helpers
# ---------------------------------------------------------------------------

def _dcg(gains: List[float], k: int) -> float:
    """Discounted Cumulative Gain up to position k."""
    score = 0.0
    for i, g in enumerate(gains[:k], start=1):
        score += g / math.log2(i + 1)
    return score


def _penalized_ndcg_at_k(
    graded_scores: List[float],
    violations_by_rank: List[List[str]],
    all_possible_scores: List[float],
    k: int,
    alpha_map: Dict[str, float],
) -> float:
    """NDCG@K with stratified multiplicative penalty for violated positions.

    For each rank position i, the gain is multiplied by (1 - alpha_type)
    for each distinct violation type present at that rank position.
    The result is normalised against the ideal ranking (sorted scores,
    no violations).

    Args:
        graded_scores: raw relevance scores (0-3 scale) per rank position.
        violations_by_rank: list of violation strings per rank position.
        all_possible_scores: all known relevance scores for the query (for ideal DCG).
        k: cutoff depth.
        alpha_map: mapping of violation type to penalty weight (0-1).
    """
    if k <= 0:
        return 0.0

    penalized_gains: List[float] = []
    for i in range(min(k, len(graded_scores))):
        g = graded_scores[i]
        if i < len(violations_by_rank) and violations_by_rank[i]:
            # Apply stratified penalties multiplicatively for each distinct violation type
            for v_type in set(violations_by_rank[i]):
                alpha = alpha_map.get(v_type, alpha_map.get("other", 0.5))
                # ── Semantic Immunity for Benign Tag Omissions ──
                # If content is highly relevant semantically (score >= 2.5), missing an official tag
                # is a benign omission rather than a quality flaw. We apply graceful softening.
                if v_type == "required_tags" and graded_scores[i] >= 2.5:
                    alpha *= 0.25
                g *= (1.0 - alpha)
        penalized_gains.append(g)

    actual_dcg = _dcg(penalized_gains, k)

    # Ideal: best possible scores sorted descending from the ENTIRE annotation pool
    ideal_gains = sorted(all_possible_scores, reverse=True)[:k]
    ideal_dcg = _dcg(ideal_gains, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _rbv_at_k(
    violations_flags: List[bool],
    k: int,
    persistence: float = 0.8,
) -> float:
    """Rank-Biased Violation rate at K.

    Uses a geometric weight decay so violations at top ranks are
    exponentially more costly.  A persistence of 0.8 means rank 1
    carries ~5x the weight of rank 10.

    Returns a value in [0, 1] where 0 = no violations, 1 = all violated.
    """
    if k <= 0:
        return 0.0

    weighted_sum = 0.0
    normalizer = 0.0
    for i in range(min(k, len(violations_flags))):
        weight = persistence ** i          # rank-1 → weight 1.0
        normalizer += weight
        if violations_flags[i]:
            weighted_sum += weight

    return weighted_sum / normalizer if normalizer > 0 else 0.0


def _rc_f1_at_k(
    precision_k: float,
    clean_k: float,
) -> float:
    """Relevance-Compliance F1: harmonic mean of P@K and Clean@K.

    Returns 0 if either dimension collapses, harshly penalising
    single-dimension failure.
    """
    if precision_k + clean_k <= 0:
        return 0.0
    return 2.0 * precision_k * clean_k / (precision_k + clean_k)


def evaluate_ir(
    experiment_dir: str,
    annotations_dir: str,
    ks: List[int],
    relevance_threshold: float,
    denominator_mode: ApDenominatorMode,
    queries_path: str = "data/experiments/queries.json",
    penalty_alpha_blocked: float = 1.0,
    penalty_alpha_status: float = 0.7,
    penalty_alpha_tag: float = 0.4,
    penalty_alpha_other: float = 0.5,
    rbv_persistence: float = 0.8,
) -> Dict[str, EngineMetrics]:
    k_values = sorted(set(k for k in ks if k > 0))
    if not k_values:
        raise ValueError("At least one positive K value is required.")

    alpha_map = {
        "blocked_tags": penalty_alpha_blocked,
        "required_status": penalty_alpha_status,
        "required_tags": penalty_alpha_tag,
        "other": penalty_alpha_other,
    }

    queries = load_queries(queries_path=Path(queries_path))
    runs = load_runs(experiment_dir)

    annotation_path = resolve_annotation_path(annotations_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    annotations = _load_annotations(annotation_path)

    query_ids = [str(q.get("id", "")).strip() for q in queries if str(q.get("id", "")).strip()]
    query_rules = {
        str(q.get("id", "")).strip(): q.get("golden_rules", {})
        for q in queries
        if str(q.get("id", "")).strip()
    }
    max_k = max(k_values)

    results_by_engine: Dict[str, EngineMetrics] = {}
    for engine_name, run_items in runs.items():
        run_by_query_id = {str(item.get("query_id", "")).strip(): item for item in run_items}

        p_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        ap_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        recall_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        violation_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        pndcg_acc: Dict[int, List[float]] = {k: [] for k in k_values}
        rbv_acc: Dict[int, List[float]] = {k: [] for k in k_values}

        total_unjudged = 0
        total_positions = 0
        zero_result_queries = 0
        violation_breakdown_counts: Dict[str, int] = {}

        for query_id in query_ids:
            query_scores = annotations.get(query_id, {})
            total_relevant = sum(1 for score in query_scores.values() if score >= relevance_threshold)

            item = run_by_query_id.get(query_id, {})
            ranked_results = _to_sorted_results(item.get("results", []), k_max=max_k)
            ranked_ids = [str(res.get("book_id", "")).strip() for res in ranked_results]

            binary, unjudged_count = _binary_relevance(ranked_ids, query_scores, relevance_threshold)
            total_unjudged += unjudged_count
            total_positions += len(ranked_ids)
            if len(ranked_ids) == 0:
                zero_result_queries += 1

            # Graded scores per rank (for NDCG)
            graded_by_rank: List[float] = [
                query_scores.get(bid, 0.0) for bid in ranked_ids
            ]

            violation_flags_by_rank: List[bool] = []
            violation_types_by_rank: List[List[str]] = []
            rules = query_rules.get(query_id, {})
            for res in ranked_results:
                violations = _check_structural_violations(rules, res)
                violation_flags_by_rank.append(bool(violations))
                violation_types_by_rank.append(violations)
                for reason in violations:
                    violation_breakdown_counts[reason] = violation_breakdown_counts.get(reason, 0) + 1

            for k in k_values:
                p_acc[k].append(_precision_at_k(binary, k))
                ap_acc[k].append(
                    _ap_at_k(
                        binary_relevance=binary,
                        total_relevant=total_relevant,
                        k=k,
                        denominator_mode=denominator_mode,
                    )
                )
                recall_acc[k].append(_recall_at_k(binary, total_relevant, k))

                top_k_violations = violation_flags_by_rank[:k]
                has_violation = 1.0 if any(top_k_violations) else 0.0
                violation_acc[k].append(has_violation)

                # Advanced metrics per query
                pndcg_acc[k].append(
                    _penalized_ndcg_at_k(
                        graded_by_rank, 
                        violation_types_by_rank, 
                        list(query_scores.values()),
                        k, 
                        alpha_map=alpha_map
                    )
                )
                rbv_acc[k].append(
                    _rbv_at_k(violation_flags_by_rank, k, persistence=rbv_persistence)
                )

        precision = {k: _safe_mean(vals) for k, vals in p_acc.items()}
        map_scores = {k: _safe_mean(vals) for k, vals in ap_acc.items()}
        recalls = {k: _safe_mean(vals) for k, vals in recall_acc.items()}
        violation_rate = {k: _safe_mean(vals) for k, vals in violation_acc.items()}
        clean_rate = {k: (1.0 - violation_rate[k]) for k in k_values}
        pndcg = {k: _safe_mean(vals) for k, vals in pndcg_acc.items()}
        rbv = {k: _safe_mean(vals) for k, vals in rbv_acc.items()}
        rc_f1 = {k: _rc_f1_at_k(precision[k], clean_rate[k]) for k in k_values}

        total_violation_events = sum(violation_breakdown_counts.values())
        if total_violation_events > 0:
            violation_breakdown_rate = {
                reason: count / float(total_violation_events)
                for reason, count in sorted(violation_breakdown_counts.items(), key=lambda kv: kv[1], reverse=True)
            }
        else:
            violation_breakdown_rate = {}

        results_by_engine[engine_name] = EngineMetrics(
            query_count=len(query_ids),
            unjudged_count=total_unjudged,
            unjudged_rate=(total_unjudged / total_positions if total_positions else 0.0),
            precision_at_k=precision,
            map_at_k=map_scores,
            violation_rate_at_k=violation_rate,
            clean_rate_at_k=clean_rate,
            violation_breakdown_rate=violation_breakdown_rate,
            zero_result_rate=(zero_result_queries / len(query_ids) if query_ids else 0.0),
            semantic_recall_at_k=recalls,
            penalized_ndcg_at_k=pndcg,
            rbv_at_k=rbv,
            rc_f1_at_k=rc_f1,
        )

    return results_by_engine


def _format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_report(
    metrics_by_engine: Dict[str, EngineMetrics],
    ks: List[int],
    baseline: Optional[str],
) -> None:
    k_values = sorted(set(ks))
    col_width = max(12, max((len(name) for name in metrics_by_engine.keys()), default=12))

    sorted_engines = sorted(
        metrics_by_engine.items(),
        key=lambda item: (
            item[1].violation_rate_at_k[max(k_values)],
            -item[1].map_at_k[max(k_values)],
        ),
    )

    print("\n" + "=" * 120)
    print("Ranking Quality Report (P@K, AP@K, mAP@K)")
    print("=" * 120)

    header_1 = ["Engine", "Queries", "Unjudged"] + [f"P@{k}" for k in k_values] + [f"mAP@{k}" for k in k_values]
    print(" | ".join(
        [
            header_1[0].ljust(col_width),
            header_1[1].rjust(7),
            header_1[2].rjust(8),
            *[col.rjust(8) for col in header_1[3:]],
        ]
    ))
    print("-" * 120)

    for engine_name, metric in sorted_engines:
        cells = [
            engine_name.ljust(col_width),
            str(metric.query_count).rjust(7),
            _format_pct(metric.unjudged_rate).rjust(8),
        ]
        cells.extend(_format_pct(metric.precision_at_k[k]).rjust(8) for k in k_values)
        cells.extend(_format_pct(metric.map_at_k[k]).rjust(8) for k in k_values)
        print(" | ".join(cells))

    print("\n" + "=" * 120)
    print("Constraint Fidelity Report (Structural Violations)")
    print("=" * 120)

    header_2 = ["Engine", "Viol@{}".format(k_values[0]), "Viol@{}".format(k_values[-1]), "Clean@{}".format(k_values[-1]), "Top Violation Types"]
    print(" | ".join([
        header_2[0].ljust(col_width),
        header_2[1].rjust(10),
        header_2[2].rjust(10),
        header_2[3].rjust(10),
        header_2[4],
    ]))
    print("-" * 120)

    for engine_name, metric in sorted_engines:
        top_breakdown = list(metric.violation_breakdown_rate.items())[:3]
        if top_breakdown:
            reason_text = ", ".join(f"{reason}:{rate*100:.1f}%" for reason, rate in top_breakdown)
        else:
            reason_text = "none"

        print(" | ".join([
            engine_name.ljust(col_width),
            _format_pct(metric.violation_rate_at_k[k_values[0]]).rjust(10),
            _format_pct(metric.violation_rate_at_k[k_values[-1]]).rjust(10),
            _format_pct(metric.clean_rate_at_k[k_values[-1]]).rjust(10),
            reason_text,
        ]))

    # --- Fault Tolerance & Survival Report ---
    print("\n" + "=" * 120)
    print("Fault Tolerance & Survival Report (ZRR, Recall@K)")
    print("=" * 120)

    surv_header = ["Engine", "ZRR (零結果率)"] + [f"Recall@{k}" for k in k_values]
    print(" | ".join([
        surv_header[0].ljust(col_width),
        surv_header[1].rjust(15),
        *[col.rjust(10) for col in surv_header[2:]],
    ]))
    print("-" * 120)

    for engine_name, metric in sorted_engines:
        cells = [
            engine_name.ljust(col_width),
            _format_pct(metric.zero_result_rate).rjust(15),
        ]
        cells.extend(_format_pct(metric.semantic_recall_at_k[k]).rjust(10) for k in k_values)
        print(" | ".join(cells))

    # --- Advanced Metrics Report ---
    print("\n" + "=" * 120)
    print("Advanced Metrics Report (Primary: pNDCG | Guardrails: RBV, RC-F1)")
    print("=" * 120)

    adv_header = ["Engine"]
    for k in k_values:
        adv_header.extend([f"pNDCG@{k}", f"RBV@{k}", f"RC-F1@{k}"])
    print(" | ".join([
        adv_header[0].ljust(col_width),
        *[col.rjust(9) for col in adv_header[1:]],
    ]))
    print("-" * 120)

    for engine_name, metric in sorted_engines:
        cells = [engine_name.ljust(col_width)]
        for k in k_values:
            cells.append(_format_pct(metric.penalized_ndcg_at_k[k]).rjust(9))
            cells.append(_format_pct(metric.rbv_at_k[k]).rjust(9))
            cells.append(_format_pct(metric.rc_f1_at_k[k]).rjust(9))
        print(" | ".join(cells))

    if baseline:
        if baseline not in metrics_by_engine:
            print(f"\n[Warning] Baseline '{baseline}' not found. Skip delta report.")
            return

        base = metrics_by_engine[baseline]
        print("\nDelta vs Baseline:", baseline)
        for engine_name, metric in sorted_engines:
            if engine_name == baseline:
                continue

            p_deltas = [metric.precision_at_k[k] - base.precision_at_k[k] for k in k_values]
            map_deltas = [metric.map_at_k[k] - base.map_at_k[k] for k in k_values]
            viol_deltas = [metric.violation_rate_at_k[k] - base.violation_rate_at_k[k] for k in k_values]
            clean_deltas = [metric.clean_rate_at_k[k] - base.clean_rate_at_k[k] for k in k_values]

            p_text = ", ".join(f"P@{k}:{delta:+.4f}" for k, delta in zip(k_values, p_deltas))
            map_text = ", ".join(f"mAP@{k}:{delta:+.4f}" for k, delta in zip(k_values, map_deltas))
            viol_text = ", ".join(f"Viol@{k}:{delta:+.4f}" for k, delta in zip(k_values, viol_deltas))
            clean_text = ", ".join(f"Clean@{k}:{delta:+.4f}" for k, delta in zip(k_values, clean_deltas))

            print(f"  {engine_name}")
            print(f"    {p_text}")
            print(f"    {map_text}")
            print(f"    {viol_text}")
            print(f"    {clean_text}")


def _to_json_serializable(metrics_by_engine: Dict[str, EngineMetrics]) -> Dict[str, dict]:
    return {engine_name: asdict(metric) for engine_name, metric in metrics_by_engine.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and structural constraints")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing run JSON files",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/experiments/queries.json",
        help="Path to queries JSON file",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/experiments/annotations",
        help="Directory containing annotation CSV (annotated.csv)",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for Precision@K / mAP@K / Violation@K",
    )
    parser.add_argument(
        "--penalty-alpha-blocked",
        type=float,
        default=1.0,
        help="Penalty weight for blocked_tags violations (default 1.0, dealbreaker)",
    )
    parser.add_argument(
        "--penalty-alpha-status",
        type=float,
        default=0.7,
        help="Penalty weight for required_status violations (default 0.7, high friction)",
    )
    parser.add_argument(
        "--penalty-alpha-tag",
        type=float,
        default=0.4,
        help="Penalty weight for required_tags violations (default 0.4, vibe substitutability)",
    )
    parser.add_argument(
        "--penalty-alpha-other",
        type=float,
        default=0.5,
        help="Penalty weight for other violations (default 0.5)",
    )
    parser.add_argument(
        "--rbv-persistence",
        type=float,
        default=0.8,
        help="RBV persistence parameter (higher = more top-heavy weighting)",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=2.0,
        help="Score threshold (0-3 scale) to treat an item as relevant",
    )
    parser.add_argument(
        "--ap-denominator",
        type=str,
        choices=["relevant", "truncated"],
        default="truncated",
        help="AP@K denominator mode: total relevant docs or min(total_relevant, K)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline engine name for delta reporting",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional file path to save metric report as JSON",
    )
    args = parser.parse_args()

    metrics_by_engine = evaluate_ir(
        experiment_dir=args.experiment_dir,
        annotations_dir=args.annotations_dir,
        queries_path=args.queries,
        ks=args.ks,
        relevance_threshold=args.relevance_threshold,
        denominator_mode=args.ap_denominator,
        penalty_alpha_blocked=args.penalty_alpha_blocked,
        penalty_alpha_status=args.penalty_alpha_status,
        penalty_alpha_tag=args.penalty_alpha_tag,
        penalty_alpha_other=args.penalty_alpha_other,
        rbv_persistence=args.rbv_persistence,
    )
    print_report(metrics_by_engine=metrics_by_engine, ks=args.ks, baseline=args.baseline)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(metrics_by_engine), f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON report to: {output_path}")


if __name__ == "__main__":
    main()
