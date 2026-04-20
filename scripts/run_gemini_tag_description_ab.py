from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm import parse_query


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class GeminiEmbedder:
    def __init__(
        self,
        model: str,
        task_type: str = "RETRIEVAL_QUERY",
        batch_size: int = 64,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "google-genai is required for embedding mapping."
            ) from exc

        from src.core.api_utils import get_current_api_key

        self.model = model
        self.task_type = task_type
        self.batch_size = batch_size
        self._genai = genai
        self._types = types
        self._api_utils = __import__("src.core.api_utils", fromlist=["placeholder"])
        self.client = genai.Client(api_key=get_current_api_key())

    def _rotate_key(self) -> None:
        new_key = self._api_utils.get_api_key_rotator().on_rate_limit_error()
        self.client = self._genai.Client(api_key=new_key)

    def _embed_batch_with_retry(self, texts: Sequence[str]) -> Dict[str, List[float]]:
        attempt = 0
        while True:
            attempt += 1
            try:
                self._api_utils.get_rate_limiter().wait()
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=list(texts),
                    config=self._types.EmbedContentConfig(task_type=self.task_type),
                )
                embeddings = list(getattr(response, "embeddings", []) or [])
                if len(embeddings) != len(texts):
                    if len(texts) == 1:
                        raise ValueError(
                            f"Expected 1 embedding for single text, got {len(embeddings)}"
                        )
                    vectors: Dict[str, List[float]] = {}
                    for text in texts:
                        vectors.update(self._embed_batch_with_retry([text]))
                    return vectors

                return {
                    text: list(embedding.values)
                    for text, embedding in zip(texts, embeddings)
                }
            except Exception as exc:
                if not self._api_utils._is_retryable(exc) or attempt >= 4:
                    raise
                self._rotate_key()
                time.sleep(min(2 * attempt, 10))

    def embed_many(self, texts: Sequence[str]) -> Dict[str, List[float]]:
        deduped = list(dict.fromkeys([str(text).strip() for text in texts if str(text).strip()]))
        vectors: Dict[str, List[float]] = {}
        for index in range(0, len(deduped), self.batch_size):
            batch = deduped[index : index + self.batch_size]
            vectors.update(self._embed_batch_with_retry(batch))
        return vectors


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-16") as f:
            return json.load(f)


def normalize_tags(tags: Sequence[str], allowed_set: set[str]) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    outside: List[str] = []
    seen = set()

    for tag in tags:
        t = str(tag).strip()
        if not t:
            continue
        if t in allowed_set:
            if t not in seen:
                kept.append(t)
                seen.add(t)
        else:
            outside.append(t)

    return kept, outside


def micro_prf(pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def macro_prf(pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]], labels: Sequence[str]) -> Dict[str, float]:
    per_label = []

    for label in labels:
        tp = fp = fn = 0
        for pred, gold in zip(pred_sets, gold_sets):
            pred_has = label in pred
            gold_has = label in gold
            if pred_has and gold_has:
                tp += 1
            elif pred_has and not gold_has:
                fp += 1
            elif (not pred_has) and gold_has:
                fn += 1

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label.append((p, r, f))

    if not per_label:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = sum(x[0] for x in per_label) / len(per_label)
    recall = sum(x[1] for x in per_label) / len(per_label)
    f1 = sum(x[2] for x in per_label) / len(per_label)
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match_rate(pred_sets: Sequence[set[str]], gold_sets: Sequence[set[str]]) -> float:
    if not pred_sets:
        return 0.0
    matches = sum(1 for pred, gold in zip(pred_sets, gold_sets) if pred == gold)
    return matches / len(pred_sets)


def resolve_queries_path(repo_root: Path, provided: Optional[Path]) -> Path:
    if provided is not None:
        return provided if provided.is_absolute() else (repo_root / provided).resolve()

    candidates = [
        repo_root / "data/experiments/queries.json",
        repo_root / "queries.json",
        repo_root / "data/raw_repo_data/experiments/queries.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    checked = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"queries.json not found. Checked:\n  - {checked}")


def load_description_items(
    descriptions_path: Path,
    all_tags: Sequence[str],
) -> Tuple[Tuple[Tuple[str, str], ...], List[str], List[str]]:
    payload = load_json(descriptions_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Tag descriptions file must be a JSON object: {descriptions_path}")

    missing: List[str] = []
    items: List[Tuple[str, str]] = []
    for tag in all_tags:
        raw_desc = payload.get(tag)
        if raw_desc is None:
            missing.append(tag)
            continue
        desc = str(raw_desc).strip()
        if not desc:
            missing.append(tag)
            continue
        items.append((tag, desc))

    extras = [str(key) for key in payload.keys() if str(key) not in set(all_tags)]
    return tuple(items), missing, extras


def run_condition(
    run_name: str,
    queries: Sequence[Dict[str, object]],
    all_tags: Sequence[str],
    allowed_set: set[str],
    model_id: str,
    max_samples: int,
    use_tag_descriptions: bool,
    tag_descriptions: Optional[Tuple[Tuple[str, str], ...]],
) -> List[Dict[str, object]]:
    data: List[Dict[str, object]] = []
    tag_list_tuple = tuple(all_tags)

    selected_queries = list(queries)
    if max_samples > 0:
        selected_queries = selected_queries[:max_samples]

    print(
        f"[ab] {run_name}: processing {len(selected_queries)} queries "
        f"(use_tag_descriptions={use_tag_descriptions})"
    )

    for idx, item in enumerate(selected_queries, start=1):
        qid = str(item.get("id", f"q{idx}"))
        query_text = str(item.get("query", "")).strip()
        golden_rules = item.get("golden_rules") or {}
        required_gold, _ = normalize_tags(golden_rules.get("required_tags") or [], allowed_set)
        blocked_gold, _ = normalize_tags(golden_rules.get("blocked_tags") or [], allowed_set)

        parse_success = True
        parse_error = ""
        raw_extracted_tags: List[str] = []
        generated_keywords: List[str] = []

        try:
            parsed = parse_query(
                query_text,
                model_id=model_id,
                tag_list=tag_list_tuple,
                tag_descriptions=tag_descriptions if use_tag_descriptions else None,
                use_tag_descriptions=use_tag_descriptions,
            )
            raw_extracted_tags = list(parsed.extracted_tags or [])
            generated_keywords = list(parsed.generated_keywords or [])
        except Exception as exc:
            parse_success = False
            parse_error = str(exc)

        pred_tags, outside_tags = normalize_tags(raw_extracted_tags, allowed_set)

        pred_set = set(pred_tags)
        req_set = set(required_gold)
        blocked_set = set(blocked_gold)

        data.append(
            {
                "query_id": qid,
                "query": query_text,
                "required_tags": required_gold,
                "blocked_tags": blocked_gold,
                "pred_tags": pred_tags,
                "outside_tags_raw": outside_tags,
                "raw_extracted_tags": raw_extracted_tags,
                "generated_keywords": generated_keywords,
                "parse_success": parse_success,
                "parse_error": parse_error,
                "required_exact_cover": bool(req_set and req_set.issubset(pred_set)),
                "blocked_clean": bool(not blocked_set or pred_set.isdisjoint(blocked_set)),
            }
        )

    return data


def map_outside_tags_with_embedding(
    run_data: Sequence[Dict[str, object]],
    all_tags: Sequence[str],
    embedding_model: str,
    similarity_threshold: float,
    batch_size: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    outside_terms = sorted(
        {
            str(term).strip()
            for row in run_data
            for term in (row.get("outside_tags_raw") or [])
            if str(term).strip()
        }
    )

    rows = [dict(row) for row in run_data]
    if not outside_terms:
        for row in rows:
            row["mapped_pred_tags"] = list(row.get("pred_tags") or [])
            row["outside_tag_mapping"] = []
        return rows, {
            "embedding_model": embedding_model,
            "outside_term_count": 0,
            "mapped_term_count": 0,
            "mapping_rate": 0.0,
            "similarity_threshold": similarity_threshold,
        }

    print(
        f"[ab] embedding mapping: model={embedding_model}, "
        f"outside_terms={len(outside_terms)}, threshold={similarity_threshold:.2f}"
    )

    embedder = GeminiEmbedder(model=embedding_model, batch_size=batch_size)
    canonical_vectors = embedder.embed_many(all_tags)
    outside_vectors = embedder.embed_many(outside_terms)

    mapping_table: Dict[str, Dict[str, object]] = {}
    mapped_term_count = 0
    for term, term_vector in outside_vectors.items():
        best_label = ""
        best_score = -1.0
        for label in all_tags:
            score = cosine_similarity(term_vector, canonical_vectors[label])
            if score > best_score:
                best_score = score
                best_label = label
        accepted = best_score >= similarity_threshold
        if accepted:
            mapped_term_count += 1
        mapping_table[term] = {
            "mapped_label": best_label,
            "similarity": best_score,
            "accepted": accepted,
        }

    for row in rows:
        mapped_pred_tags = list(row.get("pred_tags") or [])
        mapping_details: List[Dict[str, object]] = []
        seen = set(mapped_pred_tags)
        for term in row.get("outside_tags_raw") or []:
            key = str(term).strip()
            if not key:
                continue
            mapping_info = mapping_table.get(key)
            if not mapping_info:
                continue
            if bool(mapping_info.get("accepted")):
                mapped_label = str(mapping_info.get("mapped_label") or "").strip()
                if mapped_label and mapped_label not in seen:
                    mapped_pred_tags.append(mapped_label)
                    seen.add(mapped_label)
            mapping_details.append(
                {
                    "source": key,
                    "mapped_label": mapping_info.get("mapped_label", ""),
                    "similarity": float(mapping_info.get("similarity", 0.0)),
                    "accepted": bool(mapping_info.get("accepted")),
                }
            )

        row["mapped_pred_tags"] = mapped_pred_tags
        row["outside_tag_mapping"] = mapping_details

    mapping_rate = mapped_term_count / len(outside_terms) if outside_terms else 0.0
    return rows, {
        "embedding_model": embedding_model,
        "outside_term_count": len(outside_terms),
        "mapped_term_count": mapped_term_count,
        "mapping_rate": mapping_rate,
        "similarity_threshold": similarity_threshold,
    }


def build_report(
    run_data: Sequence[Dict[str, object]],
    all_tags: Sequence[str],
    model_id: str,
    use_tag_descriptions: bool,
    descriptions_path: Optional[Path],
    prediction_key: str,
    embedding_model: str,
    embedding_mapping_summary: Optional[Dict[str, object]],
) -> Dict[str, object]:
    total_queries = len(run_data)
    parse_success = sum(1 for row in run_data if row.get("parse_success"))
    parse_success_rate = parse_success / total_queries if total_queries else 0.0

    pred_sets: List[set[str]] = []
    required_sets: List[set[str]] = []
    required_query_count = 0
    required_cover_hits = 0

    blocked_query_count = 0
    blocked_clean_hits = 0

    raw_outside_count = 0
    raw_pred_tag_count = 0

    pred_tag_count_sum = 0

    for row in run_data:
        pred_tags = row.get(prediction_key) or row.get("pred_tags") or []
        required_tags = row.get("required_tags") or []
        blocked_tags = row.get("blocked_tags") or []
        outside = row.get("outside_tags_raw") or []

        pred_set = set(pred_tags)
        required_set = set(required_tags)
        blocked_set = set(blocked_tags)

        if required_set:
            required_query_count += 1
            required_cover_hits += int(required_set.issubset(pred_set))
            pred_sets.append(pred_set)
            required_sets.append(required_set)

        if blocked_set:
            blocked_query_count += 1
            blocked_clean_hits += int(pred_set.isdisjoint(blocked_set))

        raw_outside_count += len(outside)
        raw_pred_tag_count += len(
            [str(t).strip() for t in (row.get("raw_extracted_tags") or []) if str(t).strip()]
        )
        pred_tag_count_sum += len(pred_tags)

    required_micro = micro_prf(pred_sets, required_sets) if pred_sets else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    required_macro = macro_prf(pred_sets, required_sets, all_tags) if pred_sets else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    required_em = exact_match_rate(pred_sets, required_sets) if pred_sets else 0.0

    outside_rate = raw_outside_count / raw_pred_tag_count if raw_pred_tag_count else 0.0

    return {
        "total_queries": total_queries,
        "parse_success": parse_success,
        "parse_success_rate": parse_success_rate,
        "required_query_count": required_query_count,
        "required_exact_cover_hits": required_cover_hits,
        "required_exact_cover_rate": required_cover_hits / required_query_count if required_query_count else 0.0,
        "required_micro_precision": float(required_micro.get("precision", 0.0)),
        "required_micro_recall": float(required_micro.get("recall", 0.0)),
        "required_micro_f1": float(required_micro.get("f1", 0.0)),
        "required_macro_precision": float(required_macro.get("precision", 0.0)),
        "required_macro_recall": float(required_macro.get("recall", 0.0)),
        "required_macro_f1": float(required_macro.get("f1", 0.0)),
        "required_exact_match_rate": required_em,
        "blocked_query_count": blocked_query_count,
        "blocked_clean_hits": blocked_clean_hits,
        "blocked_clean_rate": blocked_clean_hits / blocked_query_count if blocked_query_count else 0.0,
        "raw_pred_tag_count": raw_pred_tag_count,
        "raw_outside_taxonomy_tag_count": raw_outside_count,
        "raw_outside_taxonomy_rate": outside_rate,
        "avg_pred_tag_count": (pred_tag_count_sum / total_queries) if total_queries else 0.0,
        "extraction_model_id": model_id,
        "embedding_model_id": embedding_model,
        "use_tag_descriptions": use_tag_descriptions,
        "descriptions_path": str(descriptions_path) if descriptions_path else "",
        "prediction_key": prediction_key,
    }

    if embedding_mapping_summary:
        report_payload["embedding_mapping_summary"] = embedding_mapping_summary

    return report_payload


def build_comparison(no_desc_report: dict, with_desc_report: dict) -> dict:
    metric_keys = [
        "parse_success_rate",
        "required_exact_cover_rate",
        "blocked_clean_rate",
        "required_micro_f1",
        "required_macro_f1",
        "required_exact_match_rate",
        "raw_outside_taxonomy_rate",
        "avg_pred_tag_count",
    ]

    metrics = {}
    for key in metric_keys:
        no_desc = float(no_desc_report.get(key, 0.0))
        with_desc = float(with_desc_report.get(key, 0.0))
        metrics[key] = {
            "no_tag_description": no_desc,
            "with_tag_description": with_desc,
            "delta_with_minus_no": with_desc - no_desc,
        }

    return {
        "total_queries": int(with_desc_report.get("total_queries", 0)),
        "extraction_model_id": with_desc_report.get("extraction_model_id", ""),
        "embedding_model_id": with_desc_report.get("embedding_model_id", ""),
        "prediction_key": with_desc_report.get("prediction_key", "pred_tags"),
        "metrics": metrics,
    }


def make_summary_markdown(
    comparison: dict,
    no_desc_report: dict,
    with_desc_report: dict,
) -> str:
    lines = [
        "# Tag Extraction A/B Report",
        "",
        "## Experiment Setup",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| total_queries | {comparison.get('total_queries', 0)} |",
        f"| extraction_model | {comparison.get('extraction_model_id', '')} |",
        f"| embedding_model | {comparison.get('embedding_model_id', '') or 'N/A'} |",
        f"| prediction_key | {comparison.get('prediction_key', 'pred_tags')} |",
        f"| no_desc_prediction_key | {no_desc_report.get('prediction_key', 'pred_tags')} |",
        f"| with_desc_prediction_key | {with_desc_report.get('prediction_key', 'pred_tags')} |",
        "",
        "## Metrics",
        "",
        "| Metric | No Description | With Description | Delta |",
        "| --- | --- | --- | --- |",
    ]

    for metric, payload in comparison.get("metrics", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    metric,
                    f"{float(payload.get('no_tag_description', 0.0)):.4f}",
                    f"{float(payload.get('with_tag_description', 0.0)):.4f}",
                    f"{float(payload.get('delta_with_minus_no', 0.0)):+.4f}",
                ]
            )
            + " |"
        )

    no_mapping = no_desc_report.get("embedding_mapping_summary") or {}
    yes_mapping = with_desc_report.get("embedding_mapping_summary") or {}
    if no_mapping or yes_mapping:
        lines.extend(
            [
                "",
                "## Embedding Mapping Stats",
                "",
                "| Condition | outside_terms | mapped_terms | mapping_rate | threshold |",
                "| --- | --- | --- | --- | --- |",
                "| no_description | "
                + f"{int(no_mapping.get('outside_term_count', 0))} | "
                + f"{int(no_mapping.get('mapped_term_count', 0))} | "
                + f"{float(no_mapping.get('mapping_rate', 0.0)):.4f} | "
                + f"{float(no_mapping.get('similarity_threshold', 0.0)):.2f} |",
                "| with_description | "
                + f"{int(yes_mapping.get('outside_term_count', 0))} | "
                + f"{int(yes_mapping.get('mapped_term_count', 0))} | "
                + f"{float(yes_mapping.get('mapping_rate', 0.0)):.4f} | "
                + f"{float(yes_mapping.get('similarity_threshold', 0.0)):.2f} |",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini tag extraction A/B: without tag descriptions vs with tag descriptions"
    )
    parser.add_argument("--queries-path", type=Path, default=None)
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))
    parser.add_argument("--descriptions-path", type=Path, default=Path("data/tag_descriptions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/tag_description_ab"))
    parser.add_argument("--model-id", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--embedding-model", type=str, default="")
    parser.add_argument("--outside-map-threshold", type=float, default=0.55)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = PROJECT_ROOT

    queries_path = resolve_queries_path(repo_root, args.queries_path)
    tags_path = args.tags_path if args.tags_path.is_absolute() else (repo_root / args.tags_path).resolve()
    descriptions_path = (
        args.descriptions_path
        if args.descriptions_path.is_absolute()
        else (repo_root / args.descriptions_path).resolve()
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir).resolve()

    if not queries_path.exists():
        raise FileNotFoundError(f"queries file not found: {queries_path}")
    if not tags_path.exists():
        raise FileNotFoundError(f"tags file not found: {tags_path}")
    if not descriptions_path.exists():
        raise FileNotFoundError(f"descriptions file not found: {descriptions_path}")

    queries = load_json(queries_path)
    if not isinstance(queries, list):
        raise ValueError("queries file must be a JSON array")

    all_tags = load_json(tags_path)
    if not isinstance(all_tags, list) or not all(isinstance(tag, str) for tag in all_tags):
        raise ValueError("tags file must be a JSON array of strings")

    description_items, missing_desc, extra_desc = load_description_items(descriptions_path, all_tags)
    if missing_desc:
        print(f"[ab] warning: {len(missing_desc)} tags missing descriptions.")
    if extra_desc:
        print(f"[ab] warning: {len(extra_desc)} extra description keys not in taxonomy.")

    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_set = set(all_tags)

    no_desc_data = run_condition(
        run_name="gemini_no_tag_desc",
        queries=queries,
        all_tags=all_tags,
        allowed_set=allowed_set,
        model_id=args.model_id,
        max_samples=args.max_samples,
        use_tag_descriptions=False,
        tag_descriptions=None,
    )

    no_desc_mapping_summary = None
    if args.embedding_model:
        no_desc_data, no_desc_mapping_summary = map_outside_tags_with_embedding(
            run_data=no_desc_data,
            all_tags=all_tags,
            embedding_model=args.embedding_model,
            similarity_threshold=args.outside_map_threshold,
            batch_size=args.embedding_batch_size,
        )

    prediction_key = "mapped_pred_tags" if args.embedding_model else "pred_tags"

    no_desc_report = build_report(
        run_data=no_desc_data,
        all_tags=all_tags,
        model_id=args.model_id,
        use_tag_descriptions=False,
        descriptions_path=None,
        prediction_key=prediction_key,
        embedding_model=args.embedding_model,
        embedding_mapping_summary=no_desc_mapping_summary,
    )

    with_desc_data = run_condition(
        run_name="gemini_with_tag_desc",
        queries=queries,
        all_tags=all_tags,
        allowed_set=allowed_set,
        model_id=args.model_id,
        max_samples=args.max_samples,
        use_tag_descriptions=True,
        tag_descriptions=description_items,
    )

    with_desc_mapping_summary = None
    if args.embedding_model:
        with_desc_data, with_desc_mapping_summary = map_outside_tags_with_embedding(
            run_data=with_desc_data,
            all_tags=all_tags,
            embedding_model=args.embedding_model,
            similarity_threshold=args.outside_map_threshold,
            batch_size=args.embedding_batch_size,
        )

    with_desc_report = build_report(
        run_data=with_desc_data,
        all_tags=all_tags,
        model_id=args.model_id,
        use_tag_descriptions=True,
        descriptions_path=descriptions_path,
        prediction_key=prediction_key,
        embedding_model=args.embedding_model,
        embedding_mapping_summary=with_desc_mapping_summary,
    )

    comparison = build_comparison(no_desc_report, with_desc_report)
    summary_md = make_summary_markdown(
        comparison=comparison,
        no_desc_report=no_desc_report,
        with_desc_report=with_desc_report,
    )

    no_desc_path = output_dir / "gemini_no_tag_desc.json"
    no_desc_report_path = output_dir / "gemini_no_tag_desc_report.json"
    with_desc_path = output_dir / "gemini_with_tag_desc.json"
    with_desc_report_path = output_dir / "gemini_with_tag_desc_report.json"
    comparison_path = output_dir / "gemini_tag_desc_comparison.json"
    summary_path = output_dir / "summary.md"

    no_desc_path.write_text(json.dumps(no_desc_data, ensure_ascii=False, indent=2), encoding="utf-8")
    no_desc_report_path.write_text(json.dumps(no_desc_report, ensure_ascii=False, indent=2), encoding="utf-8")
    with_desc_path.write_text(json.dumps(with_desc_data, ensure_ascii=False, indent=2), encoding="utf-8")
    with_desc_report_path.write_text(json.dumps(with_desc_report, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(summary_md, encoding="utf-8")

    print("[ab] done")
    print(f"  no_desc_report: {no_desc_report_path}")
    print(f"  with_desc_report: {with_desc_report_path}")
    print(f"  comparison: {comparison_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
