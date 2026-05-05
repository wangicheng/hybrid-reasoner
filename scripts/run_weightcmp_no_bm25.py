import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.generate_run import RunGenerator


def load_annotations(annotation_path: Path) -> dict[str, dict[str, float]]:
    annotations: dict[str, dict[str, float]] = {}
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


def dcg(rels: list[float]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_10(query_id: str, results: list[dict], annotations: dict[str, dict[str, float]]) -> float:
    rels = [annotations.get(query_id, {}).get(str(item.get("book_id", "")), 0.0) for item in results[:10]]
    ideal = sorted(annotations.get(query_id, {}).values(), reverse=True)[:10]
    idcg = dcg(ideal)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0


def avg_at_10(query_id: str, results: list[dict], annotations: dict[str, dict[str, float]]) -> float:
    rels = [annotations.get(query_id, {}).get(str(item.get("book_id", "")), 0.0) for item in results[:10]]
    return (sum(rels) / 10.0) if rels else 0.0


def strict_at_10(query_id: str, results: list[dict], annotations: dict[str, dict[str, float]]) -> float:
    rels = [annotations.get(query_id, {}).get(str(item.get("book_id", "")), 0.0) for item in results[:10]]
    return float(sum(1 for score in rels if score >= 3.0))


def evaluate_run(run_path: Path, annotations: dict[str, dict[str, float]]) -> dict:
    data = json.loads(run_path.read_text(encoding="utf-8"))
    ndcgs: list[float] = []
    avgs: list[float] = []
    stricts: list[float] = []

    for entry in data:
        query_id = str(entry.get("query_id", ""))
        results = entry.get("results", [])
        ndcgs.append(ndcg_at_10(query_id, results, annotations))
        avgs.append(avg_at_10(query_id, results, annotations))
        stricts.append(strict_at_10(query_id, results, annotations))

    return {
        "run_file": run_path.name,
        "n_queries": len(data),
        "mean_ndcg_at_10": mean(ndcgs) if ndcgs else 0.0,
        "mean_avg_at_10": mean(avgs) if avgs else 0.0,
        "mean_strict_at_10": mean(stricts) if stricts else 0.0,
    }


def main() -> None:
    queries_path = Path("data/experiments/queries.json")
    annotations_path = Path("data/experiments/annotations/annotated.csv")
    output_root = Path("data/experiments/runs")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    annotations = load_annotations(annotations_path)

    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S_weightcmp_nobm25")
    output_dir = output_root / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"name": "gemma4_default_parser_attr60", "semantic_weight": 0.40, "attribute_weight": 0.60},
        {"name": "gemma4_default_parser_attr65", "semantic_weight": 0.35, "attribute_weight": 0.65},
        {"name": "gemma4_default_parser_attr70", "semantic_weight": 0.30, "attribute_weight": 0.70},
    ]
    shared_cache_suffix = "_sameparse"

    for cfg in configs:
        print(
            "\n[WeightSweep-NoBM25] Running "
            f"{cfg['name']} (semantic={cfg['semantic_weight']}, attribute={cfg['attribute_weight']})"
        )
        generator = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
        generator.generate_run(
            queries_config=queries,
            engine_name=cfg["name"],
            output_dir=output_dir,
            semantic_weight=cfg["semantic_weight"],
            attribute_weight=cfg["attribute_weight"],
            run_suffix=shared_cache_suffix,
            enable_bm25=False,
            bm25_weight=0.0,
        )

    rows: list[dict] = []
    for cfg in configs:
        run_path = output_dir / f"{cfg['name']}{shared_cache_suffix}.json"
        metrics = evaluate_run(run_path, annotations)
        metrics["semantic_weight"] = cfg["semantic_weight"]
        metrics["attribute_weight"] = cfg["attribute_weight"]
        rows.append(metrics)

    baseline = next((r for r in rows if r["run_file"] == f"gemma4_default_parser_attr60{shared_cache_suffix}.json"), None)
    for row in rows:
        if baseline is None:
            row["delta_ndcg_vs_attr60"] = 0.0
            row["delta_strict_vs_attr60"] = 0.0
        else:
            row["delta_ndcg_vs_attr60"] = row["mean_ndcg_at_10"] - baseline["mean_ndcg_at_10"]
            row["delta_strict_vs_attr60"] = row["mean_strict_at_10"] - baseline["mean_strict_at_10"]

    rows.sort(key=lambda item: item["mean_ndcg_at_10"], reverse=True)

    decision_csv = output_dir / "final_decision_table.csv"
    with decision_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_file",
            "semantic_weight",
            "attribute_weight",
            "n_queries",
            "mean_ndcg_at_10",
            "mean_avg_at_10",
            "mean_strict_at_10",
            "delta_ndcg_vs_attr60",
            "delta_strict_vs_attr60",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    decision_json = output_dir / "final_decision_table.json"
    decision_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[WeightSweep-NoBM25] Completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
