from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse components from the original experiment script if possible, 
# but for clarity in this new experiment, we redefine or import them.
from scripts.tag_template_experiment import (
    LabelRecord, MappingExample, GeminiEmbedder, 
    normalize_term, safe_mean, load_source_records, 
    build_dataset, load_candidate_tags, build_mapping_examples,
    cosine_similarity, compute_f1_metrics, get_rendered_vectors,
    get_model_run_name, normalize_model_name
)

DEFAULT_EXPERIMENT_ROOT = Path("data/experiments/tag_template")
DEFAULT_SOURCE = DEFAULT_EXPERIMENT_ROOT / "datasets" / "tag_template_eval_dataset.json"
DEFAULT_CANDIDATE_TAGS_SOURCE = Path("data/all_tags.json")
DEFAULT_TAG_DESCRIPTIONS_SOURCE = DEFAULT_EXPERIMENT_ROOT / "datasets" / "tag_descriptions.json"

EXPERIMENT_CONFIG = {
    "source": DEFAULT_SOURCE,
    "candidate_tags_source": DEFAULT_CANDIDATE_TAGS_SOURCE,
    "tag_descriptions_source": DEFAULT_TAG_DESCRIPTIONS_SOURCE,
    "output_dir": DEFAULT_EXPERIMENT_ROOT / "runs_with_desc",
    "query_template": "這部作品的類型偏向{label}",
    "candidate_template": "{label}：{description}",
    "use_symmetric_templates": False,
    "model": "gemini-embedding-001",
    "batch_size": 64,
    "query_task_type": "RETRIEVAL_QUERY",
    "candidate_task_type": "RETRIEVAL_DOCUMENT",
}

def load_tag_descriptions(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"[tag-desc] Warning: Description file {path} not found.")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[tag-desc] Error loading descriptions: {e}")
        return {}

def render_desc_text(template_text: str, label: str, description: str = "") -> str:
    # Handle cases where description might be missing
    return template_text.format(label=label, description=description).strip("： ")

def compute_cmc(examples_results: List[Dict[str, object]], max_k: int = 10) -> List[float]:
    cmc = []
    total = len(examples_results)
    if total == 0:
        return [0.0] * max_k
    for k in range(1, max_k + 1):
        hits = sum(1 for res in examples_results if res["rank"] <= k)
        cmc.append(hits / total)
    return cmc

def evaluate_with_desc(
    candidate_labels: Sequence[str],
    tag_descriptions: Dict[str, str],
    candidate_vectors: Dict[str, List[float]],
    query_vectors: Dict[str, List[float]],
    query_template: str,
    candidate_template: str,
    examples: Sequence[MappingExample],
) -> Dict[str, object]:
    
    candidate_render_map = {
        label: render_desc_text(candidate_template, label, tag_descriptions.get(label, ""))
        for label in candidate_labels
    }

    example_results: List[Dict[str, object]] = []
    top1_predictions: List[str] = []
    true_labels: List[str] = []

    for example in examples:
        query_rendered = query_template.format(label=example.query_text)
        query_vector = query_vectors[query_rendered]

        ranked = sorted(
            (
                (
                    label,
                    cosine_similarity(query_vector, candidate_vectors[candidate_render_map[label]]),
                )
                for label in candidate_labels
            ),
            key=lambda item: (-item[1], item[0]),
        )

        ranked_labels = [label for label, _ in ranked]
        target_rank = ranked_labels.index(example.target_label) + 1
        top1_label = ranked_labels[0]

        top1_predictions.append(top1_label)
        true_labels.append(example.target_label)

        example_results.append(
            {
                "query_text": example.query_text,
                "target_label": example.target_label,
                "group_id": example.group_id,
                "source_type": example.source_type,
                "rank": target_rank,
                "reciprocal_rank": 1.0 / target_rank,
                "top1_prediction": top1_label,
                "top1_hit": target_rank == 1,
            }
        )

    f1_metrics = compute_f1_metrics(true_labels, top1_predictions)
    cmc = compute_cmc(example_results, 10)

    per_label_results: List[Dict[str, object]] = []
    grouped_examples: Dict[str, List[Dict[str, object]]] = {}
    for result in example_results:
        grouped_examples.setdefault(str(result["target_label"]), []).append(result)

    for label, label_results in grouped_examples.items():
        wrong_predictions = Counter(
            str(result["top1_prediction"])
            for result in label_results
            if not result["top1_hit"]
        )
        per_label_results.append(
            {
                "label": label,
                "group_id": int(label_results[0]["group_id"]),
                "query_count": len(label_results),
                "top1_accuracy": safe_mean(1.0 if result["top1_hit"] else 0.0 for result in label_results),
                "mrr": safe_mean(float(result["reciprocal_rank"]) for result in label_results),
                "top_confusions": [
                    {"label": wrong_label, "count": count}
                    for wrong_label, count in wrong_predictions.most_common(3)
                ],
            }
        )

    return {
        "top1": cmc[0],
        "top3": cmc[2],
        "top5": cmc[4],
        "top10": cmc[9],
        "mrr": safe_mean(float(result["reciprocal_rank"]) for result in example_results),
        "micro_f1": f1_metrics["micro_f1"],
        "macro_f1": f1_metrics["macro_f1"],
        "cmc": cmc,
        "per_label": per_label_results,
        "examples": example_results,
    }

def run_experiment(
    name: str,
    query_template: str,
    candidate_template: str,
    use_desc: bool,
    config: Dict[str, object]
) -> Dict[str, object]:
    model = config["model"]
    output_dir = config["output_dir"] / name
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(config["source"])
    candidate_tags_source = Path(config["candidate_tags_source"])
    tag_descriptions_source = Path(config["tag_descriptions_source"])
    
    query_task_type = config["query_task_type"]
    candidate_task_type = config["candidate_task_type"]
    batch_size = config["batch_size"]

    # Load data
    parsed_records = load_source_records(source)
    dataset = build_dataset(parsed_records=parsed_records, max_items=None)
    candidate_labels = load_candidate_tags(candidate_tags_source, dataset)
    examples = build_mapping_examples(dataset=dataset, include_canonical_label_queries=False)
    tag_descriptions = load_tag_descriptions(tag_descriptions_source) if use_desc else {}

    print(f"\n[Run: {name}] Labels: {len(dataset)}, Queries: {len(examples)}, Candidates: {len(candidate_labels)}")

    # Embed queries
    query_rendered_texts = [query_template.format(label=example.query_text) for example in examples]
    query_vectors = get_rendered_vectors(
        rendered_texts=query_rendered_texts,
        output_dir=output_dir,
        cache_name="queries",
        template_text=query_template,
        role="query",
        model=model,
        task_type=query_task_type,
        batch_size=batch_size,
    )

    # Embed candidates
    candidate_rendered_texts = [
        render_desc_text(candidate_template, label, tag_descriptions.get(label, "")) 
        for label in candidate_labels
    ]
    candidate_vectors = get_rendered_vectors(
        rendered_texts=candidate_rendered_texts,
        output_dir=output_dir,
        cache_name="candidates",
        template_text=candidate_template,
        role="candidate",
        model=model,
        task_type=candidate_task_type,
        batch_size=batch_size,
    )

    # Evaluate
    result = evaluate_with_desc(
        candidate_labels=candidate_labels,
        tag_descriptions=tag_descriptions,
        candidate_vectors=candidate_vectors,
        query_vectors=query_vectors,
        query_template=query_template,
        candidate_template=candidate_template,
        examples=examples,
    )

    # Save results
    report_payload = {
        "model": model,
        "query_template": query_template,
        "candidate_template": candidate_template,
        "use_description": use_desc,
        "metrics": {
            "top1": result["top1"],
            "top3": result["top3"],
            "top5": result["top5"],
            "top10": result["top10"],
            "mrr": result["mrr"],
            "macro_f1": result["macro_f1"],
            "cmc": result["cmc"]
        },
        "per_label": result["per_label"]
    }
    
    (output_dir / "results.json").write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload

def evaluate_hybrid(
    candidate_labels: Sequence[str],
    tag_descriptions: Dict[str, str],
    candidate_vectors_base: Dict[str, List[float]],
    candidate_vectors_desc: Dict[str, List[float]],
    query_vectors: Dict[str, List[float]],
    query_template: str,
    candidate_template_base: str,
    candidate_template_desc: str,
    examples: Sequence[MappingExample],
    weight_base: float = 0.5
) -> Dict[str, object]:
    
    candidate_render_map_base = {
        label: render_desc_text(candidate_template_base, label, "")
        for label in candidate_labels
    }
    candidate_render_map_desc = {
        label: render_desc_text(candidate_template_desc, label, tag_descriptions.get(label, ""))
        for label in candidate_labels
    }

    example_results: List[Dict[str, object]] = []
    top1_predictions: List[str] = []
    true_labels: List[str] = []

    for example in examples:
        query_rendered = query_template.format(label=example.query_text)
        query_vector = query_vectors[query_rendered]

        ranked = []
        for label in candidate_labels:
            score_base = cosine_similarity(query_vector, candidate_vectors_base[candidate_render_map_base[label]])
            score_desc = cosine_similarity(query_vector, candidate_vectors_desc[candidate_render_map_desc[label]])
            final_score = weight_base * score_base + (1.0 - weight_base) * score_desc
            ranked.append((label, final_score))
            
        ranked.sort(key=lambda item: (-item[1], item[0]))

        ranked_labels = [label for label, _ in ranked]
        target_rank = ranked_labels.index(example.target_label) + 1
        top1_label = ranked_labels[0]
        top5_labels = ranked_labels[:5]

        top1_predictions.append(top1_label)
        true_labels.append(example.target_label)

        example_results.append(
            {
                "query_text": example.query_text,
                "target_label": example.target_label,
                "top1_hit": top1_label == example.target_label,
                "top5_hit": example.target_label in top5_labels,
                "rank": target_rank,
                "reciprocal_rank": 1.0 / target_rank,
            }
        )

    f1_metrics = compute_f1_metrics(true_labels, top1_predictions)
    cmc = compute_cmc(example_results, 10)

    return {
        "top1": cmc[0],
        "top3": cmc[2],
        "top5": cmc[4],
        "top10": cmc[9],
        "mrr": safe_mean(float(result["reciprocal_rank"]) for result in example_results),
        "micro_f1": f1_metrics["micro_f1"],
        "macro_f1": f1_metrics["macro_f1"],
        "cmc": cmc
    }

def main():
    model = normalize_model_name(os.getenv("TAG_TEMPLATE_MODEL", EXPERIMENT_CONFIG["model"]))
    base_output_dir = Path(EXPERIMENT_CONFIG["output_dir"]) / get_model_run_name(model)
    
    config = dict(EXPERIMENT_CONFIG)
    config["model"] = model
    config["output_dir"] = base_output_dir

    # Common Templates
    QUERY_TPL = "這部作品的類型偏向{label}"
    CAND_TPL_RAW = "{label}"
    CAND_TPL_SYM = "這部作品的類型偏向{label}"
    CAND_TPL_DESC = "{label}：{description}"

    # 0. Raw Label (Baseline)
    raw_baseline = run_experiment(
        name="baseline_raw",
        query_template="{label}",
        candidate_template="{label}",
        use_desc=False,
        config=config
    )

    # 1. Symmetric novel_genre
    baseline = run_experiment(
        name="baseline_symmetric",
        query_template=QUERY_TPL,
        candidate_template=CAND_TPL_SYM,
        use_desc=False,
        config=config
    )

    # 2. Experiment: Asymmetric (With Description)
    experiment = run_experiment(
        name="experiment_with_desc",
        query_template=QUERY_TPL,
        candidate_template=CAND_TPL_DESC,
        use_desc=True,
        config=config
    )

    # 3. Hybrid: Weighted (0.7 symmetric + 0.3 desc)
    print("\n[Run: Hybrid 0.7/0.3]")
    # Reload data
    source = Path(config["source"])
    candidate_tags_source = Path(config["candidate_tags_source"])
    tag_descriptions_source = Path(config["tag_descriptions_source"])
    parsed_records = load_source_records(source)
    dataset = build_dataset(parsed_records=parsed_records, max_items=None)
    candidate_labels = load_candidate_tags(candidate_tags_source, dataset)
    examples = build_mapping_examples(dataset=dataset, include_canonical_label_queries=False)
    tag_descriptions = load_tag_descriptions(tag_descriptions_source)

    # Get cached vectors
    query_vectors = get_rendered_vectors(
        rendered_texts=[QUERY_TPL.format(label=e.query_text) for e in examples],
        output_dir=base_output_dir / "baseline_symmetric",
        cache_name="queries",
        template_text=QUERY_TPL,
        role="query",
        model=model,
        task_type=config["query_task_type"],
        batch_size=config["batch_size"],
    )
    
    candidate_vectors_sym = get_rendered_vectors(
        rendered_texts=[CAND_TPL_SYM.format(label=l) for l in candidate_labels],
        output_dir=base_output_dir / "baseline_symmetric",
        cache_name="candidates",
        template_text=CAND_TPL_SYM,
        role="candidate",
        model=model,
        task_type=config["candidate_task_type"],
        batch_size=config["batch_size"],
    )
    
    candidate_vectors_desc = get_rendered_vectors(
        rendered_texts=[render_desc_text(CAND_TPL_DESC, l, tag_descriptions.get(l, "")) for l in candidate_labels],
        output_dir=base_output_dir / "experiment_with_desc",
        cache_name="candidates",
        template_text=CAND_TPL_DESC,
        role="candidate",
        model=model,
        task_type=config["candidate_task_type"],
        batch_size=config["batch_size"],
    )

    hybrid = evaluate_hybrid(
        candidate_labels=candidate_labels,
        tag_descriptions=tag_descriptions,
        candidate_vectors_base=candidate_vectors_sym,
        candidate_vectors_desc=candidate_vectors_desc,
        query_vectors=query_vectors,
        query_template=QUERY_TPL,
        candidate_template_base=CAND_TPL_SYM,
        candidate_template_desc=CAND_TPL_DESC,
        examples=examples,
        weight_base=0.7
    )

    # Final Comparison
    print("\n" + "="*90)
    print(f"Comparison for {model}")
    print("="*90)
    print(f"{'Metric':<10} | {'Raw':<12} | {'Symmetric':<12} | {'With Desc':<12} | {'Hybrid (0.7)':<12}")
    print("-" * 95)
    
    for key in ["top1", "top3", "top5", "top10", "mrr", "macro_f1"]:
        v_raw = raw_baseline["metrics"].get(key, 0)
        v_base = baseline["metrics"].get(key, 0)
        v_exp = experiment["metrics"].get(key, 0)
        v_hybrid = hybrid.get(key, 0)
        print(f"{key:<10} | {v_raw:<12.4f} | {v_base:<12.4f} | {v_exp:<12.4f} | {v_hybrid:<12.4f}")
    print("="*95)

    print("\nCMC Data (1-10):")
    def format_cmc(cmc):
        return "[" + ", ".join([f"{x*100:.2f}" for x in cmc]) + "]"

    print(f"Raw:       {format_cmc(raw_baseline['metrics']['cmc'])}")
    print(f"Symmetric: {format_cmc(baseline['metrics']['cmc'])}")
    print(f"Hybrid:    {format_cmc(hybrid['cmc'])}")

if __name__ == "__main__":
    main()
