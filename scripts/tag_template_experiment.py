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


HEADER_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
TITLE_RE = re.compile(r"^(?P<label>.+?)(?:\s*[（(](?P<alias>.+?)[)）])?\s*$")
DEFAULT_EXPERIMENT_ROOT = Path("data/experiments/tag_template")
DEFAULT_SOURCE = DEFAULT_EXPERIMENT_ROOT / "datasets" / "tag_template_eval_dataset.json"
DEFAULT_CANDIDATE_TAGS_SOURCE = Path("data/all_tags.json")

DEFAULT_TEMPLATES = [
    {"name": "raw_label", "template": "{label}"},
    {"name": "novel_tag_cn", "template": "小說標籤：{label}"},
    {"name": "light_novel_tag", "template": "這本輕小說的標籤是{label}"},
    {"name": "novel_style", "template": "這是一部帶有{label}元素的小說"},
    {"name": "novel_theme", "template": "這部小說的核心題材是{label}"},
    {"name": "novel_genre", "template": "這部作品的類型偏向{label}"},
    {"name": "novel_trait", "template": "這部小說最鮮明的特色是{label}"},
    {"name": "novel_contains", "template": "這部小說包含明顯的{label}要素"},
    {"name": "novel_focus", "template": "這部小說的主要看點之一是{label}"},
    {"name": "novel_setting", "template": "這部小說的世界觀與設定帶有{label}色彩"},
    {"name": "novel_subject", "template": "從題材上看，這部小說屬於{label}"},
    {"name": "novel_element", "template": "這部小說具有鮮明的{label}元素"},
    {"name": "novel_core_element", "template": "這部作品的核心元素之一是{label}"},
    {"name": "novel_feature", "template": "這部小說展現出明確的{label}特徵"},
    {"name": "novel_main_axis", "template": "這部小說的主軸之一圍繞著{label}"},
    {"name": "novel_content_direction", "template": "這部作品的內容走向偏向{label}"},
    {"name": "novel_material", "template": "這部小說採用了大量的{label}題材"},
    {"name": "novel_color", "template": "這部作品整體帶有濃厚的{label}色彩"},
    {"name": "novel_property", "template": "這部小說具備明顯的{label}屬性"},
    {"name": "novel_impression", "template": "這部小說給人的主要印象之一是{label}"},
]

EXPERIMENT_CONFIG = {
    "source": DEFAULT_SOURCE,
    "candidate_tags_source": DEFAULT_CANDIDATE_TAGS_SOURCE,
    "output_dir": DEFAULT_EXPERIMENT_ROOT / "runs",
    "templates_file": None,
    "query_template": "tag: {label}",
    "use_symmetric_templates": True,
    "include_canonical_label_queries": False,
    "max_items": None,
    "query_task_type": "RETRIEVAL_QUERY",
    "candidate_task_type": "RETRIEVAL_DOCUMENT",
    "model": "gemini-embedding-001",
    "batch_size": 64,
    "preview": False,
}

RETIRED_MODELS = {
    "text-embedding-004": "2026-01-14",
    "models/text-embedding-004": "2026-01-14",
}


def normalize_model_name(model: str) -> str:
    normalized = model.strip()
    if normalized == "text-embedding-004":
        return "models/text-embedding-004"
    return normalized


def sanitize_path_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "default"


def get_model_run_name(model: str) -> str:
    normalized = model.strip()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    return sanitize_path_component(normalized)


def resolve_runtime_config(config: Dict[str, object]) -> Dict[str, object]:
    resolved = dict(config)
    default_model = str(config["model"])
    model = normalize_model_name(os.getenv("TAG_TEMPLATE_MODEL", default_model))
    retired_at = RETIRED_MODELS.get(model)
    if retired_at:
        raise ValueError(
            f"Embedding model '{model}' was retired on {retired_at}. "
            "Use a currently supported Gemini embedding model instead."
        )

    output_dir_override = os.getenv("TAG_TEMPLATE_OUTPUT_DIR")
    base_output_dir = Path(output_dir_override) if output_dir_override else Path(config["output_dir"])
    output_dir = base_output_dir / get_model_run_name(model)

    resolved["model"] = model
    resolved["output_dir"] = output_dir
    return resolved


@dataclass
class LabelRecord:
    label: str
    alias: str | None
    group_id: int
    positives: List[str]


@dataclass
class MappingExample:
    query_text: str
    target_label: str
    group_id: int
    source_type: str


class GeminiEmbedder:
    def __init__(
        self,
        model: str = "gemini-embedding-001",
        task_type: str = "RETRIEVAL_QUERY",
        batch_size: int = 64,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "google-genai is required for embedding evaluation. "
                "Install dependencies first or run with preview mode."
            ) from exc

        self.model = model
        self.task_type = task_type
        self.batch_size = batch_size
        self._genai = genai
        self._types = types
        from src.core.api_utils import get_current_api_key

        self._api_utils = __import__("src.core.api_utils", fromlist=["placeholder"])
        self.client = genai.Client(api_key=get_current_api_key())

    def _rotate_key(self) -> None:
        new_key = self._api_utils.get_api_key_rotator().on_rate_limit_error()
        self.client = self._genai.Client(api_key=new_key)

    def embed_many(self, texts: Sequence[str]) -> Dict[str, List[float]]:
        unique_texts = list(dict.fromkeys(texts))
        vectors: Dict[str, List[float]] = {}
        for index in range(0, len(unique_texts), self.batch_size):
            batch = unique_texts[index : index + self.batch_size]
            vectors.update(self._embed_batch_with_retry(batch))
        return vectors

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
                    print(
                        "[tag-template] embedding batch size mismatch: "
                        f"requested {len(texts)}, received {len(embeddings)}; "
                        "falling back to per-text requests."
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
                if not self._api_utils._is_retryable(exc):
                    raise
                print(
                    f"[tag-template] embedding batch failed on attempt {attempt}: {exc}"
                )
                self._rotate_key()
                time.sleep(2 * attempt)


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip())


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def split_terms(raw_text: str) -> List[str]:
    terms = []
    for chunk in re.split(r"[、,，/]", raw_text):
        term = normalize_term(chunk)
        if term:
            terms.append(term)
    return terms


def parse_label_header(raw_header: str) -> tuple[str, str | None]:
    match = TITLE_RE.match(normalize_term(raw_header))
    if not match:
        return normalize_term(raw_header), None
    label = normalize_term(match.group("label"))
    alias = normalize_term(match.group("alias") or "")
    return label, alias or None


def dedupe_positives(label: str, alias: str | None, positives: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = {label}
    for term in list(positives) + ([alias] if alias else []):
        normalized = normalize_term(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def parse_tmp_source(source_path: Path) -> List[tuple[int, str, str | None, List[str]]]:
    raw_text = source_path.read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    parsed: List[tuple[int, str, str | None, List[str]]] = []
    cursor = 0
    group_id = 1
    previous_index = 0
    while cursor < len(lines):
        header_match = HEADER_RE.match(lines[cursor])
        if not header_match:
            cursor += 1
            continue

        item_index = int(header_match.group(1))
        if previous_index and item_index <= previous_index:
            group_id += 1
        previous_index = item_index

        label, alias = parse_label_header(header_match.group(2))
        positives_line = lines[cursor + 1] if cursor + 1 < len(lines) else ""
        positives = split_terms(positives_line)
        cursor += 2

        parsed.append((group_id, label, alias, dedupe_positives(label, alias, positives)))

    if not parsed:
        raise ValueError(f"No label records parsed from {source_path}")
    return parsed


def parse_json_source(source_path: Path) -> List[tuple[int, str, str | None, List[str]]]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    parsed: List[tuple[int, str, str | None, List[str]]] = []

    if isinstance(payload, dict) and "groups" in payload:
        for group_index, group in enumerate(payload["groups"], start=1):
            group_id = int(group.get("group_id", group_index))
            for item in group.get("labels", []):
                label = normalize_term(str(item["label"]))
                alias = normalize_term(str(item.get("alias", "") or "")) or None
                positives = [normalize_term(str(term)) for term in item.get("positives", [])]
                parsed.append((group_id, label, alias, dedupe_positives(label, alias, positives)))
    elif isinstance(payload, list):
        for item in payload:
            label = normalize_term(str(item["label"]))
            alias = normalize_term(str(item.get("alias", "") or "")) or None
            group_id = int(item.get("group_id", 1))
            positives = [normalize_term(str(term)) for term in item.get("positives", [])]
            parsed.append((group_id, label, alias, dedupe_positives(label, alias, positives)))
    else:
        raise ValueError(f"Unsupported dataset format in {source_path}")

    if not parsed:
        raise ValueError(f"No label records parsed from {source_path}")
    return parsed


def load_source_records(source_path: Path) -> List[tuple[int, str, str | None, List[str]]]:
    if source_path.suffix.lower() == ".json":
        return parse_json_source(source_path)
    return parse_tmp_source(source_path)


def build_dataset(
    parsed_records: Sequence[tuple[int, str, str | None, List[str]]],
    max_items: int | None,
) -> List[LabelRecord]:
    limited_records = list(parsed_records[:max_items] if max_items else parsed_records)
    return [
        LabelRecord(label=label, alias=alias, group_id=group_id, positives=positives)
        for group_id, label, alias, positives in limited_records
    ]


def load_templates(templates_file: Path | None) -> List[Dict[str, str]]:
    if not templates_file:
        return list(DEFAULT_TEMPLATES)

    payload = json.loads(templates_file.read_text(encoding="utf-8"))
    templates: List[Dict[str, str]] = []
    if isinstance(payload, list):
        for index, item in enumerate(payload, start=1):
            if isinstance(item, str):
                templates.append({"name": f"template_{index}", "template": item})
            elif isinstance(item, dict) and "template" in item:
                name = item.get("name") or f"template_{index}"
                templates.append({"name": str(name), "template": str(item["template"])})
    if not templates:
        raise ValueError(f"No templates loaded from {templates_file}")
    return templates


def load_candidate_tags(candidate_tags_source: Path, dataset: Sequence[LabelRecord]) -> List[str]:
    try:
        raw_text = candidate_tags_source.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = candidate_tags_source.read_text(encoding="utf-16")

    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected candidate tag format in {candidate_tags_source}")

    candidates: List[str] = []
    seen = set()
    for tag in payload:
        normalized = normalize_term(str(tag))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)

    for record in dataset:
        if record.label not in seen:
            candidates.append(record.label)
            seen.add(record.label)

    if not candidates:
        raise ValueError(f"No candidate tags loaded from {candidate_tags_source}")
    return candidates


def build_mapping_examples(
    dataset: Sequence[LabelRecord],
    include_canonical_label_queries: bool,
) -> List[MappingExample]:
    examples: List[MappingExample] = []
    for record in dataset:
        for term in record.positives:
            examples.append(
                MappingExample(
                    query_text=term,
                    target_label=record.label,
                    group_id=record.group_id,
                    source_type="positive",
                )
            )
        if include_canonical_label_queries:
            examples.append(
                MappingExample(
                    query_text=record.label,
                    target_label=record.label,
                    group_id=record.group_id,
                    source_type="canonical",
                )
            )
    if not examples:
        raise ValueError("No mapping examples were built from the evaluation dataset.")
    return examples


def render_text(template_text: str, label: str) -> str:
    return template_text.format(label=label)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    dot_product = sum(x * y for x, y in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(x * x for x in vector_a))
    norm_b = math.sqrt(sum(y * y for y in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_cache_dir(output_dir: Path) -> Path:
    return output_dir / "embedding_cache"


def get_render_cache_path(
    output_dir: Path,
    cache_name: str,
    template_text: str,
    model: str,
    task_type: str,
    role: str,
) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", cache_name).strip("_") or "cache"
    cache_key = f"{role}|{cache_name}|{template_text}|{model}|{task_type}"
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    return get_cache_dir(output_dir) / f"{role}_{safe_name}_{digest}.json"


def load_render_cache(cache_path: Path) -> Dict[str, List[float]]:
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    vectors = payload.get("vectors", {})
    return {
        str(text): [float(value) for value in vector]
        for text, vector in vectors.items()
    }


def save_render_cache(
    cache_path: Path,
    cache_name: str,
    template_text: str,
    role: str,
    model: str,
    task_type: str,
    vectors: Dict[str, List[float]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": cache_name,
        "role": role,
        "template": template_text,
        "model": model,
        "task_type": task_type,
        "vector_count": len(vectors),
        "vectors": vectors,
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_rendered_vectors(
    rendered_texts: Sequence[str],
    output_dir: Path,
    cache_name: str,
    template_text: str,
    role: str,
    model: str,
    task_type: str,
    batch_size: int,
) -> Dict[str, List[float]]:
    deduped_texts = list(dict.fromkeys(rendered_texts))
    cache_path = get_render_cache_path(
        output_dir=output_dir,
        cache_name=cache_name,
        template_text=template_text,
        model=model,
        task_type=task_type,
        role=role,
    )
    cached_vectors = load_render_cache(cache_path)
    missing_texts = [text for text in deduped_texts if text not in cached_vectors]

    if missing_texts:
        print(
            f"[tag-template] {role}:{cache_name}: cache hit {len(cached_vectors)}/{len(deduped_texts)}, "
            f"embedding {len(missing_texts)} missing texts."
        )
        embedder = GeminiEmbedder(model=model, task_type=task_type, batch_size=batch_size)
        cached_vectors.update(embedder.embed_many(missing_texts))
        save_render_cache(
            cache_path=cache_path,
            cache_name=cache_name,
            template_text=template_text,
            role=role,
            model=model,
            task_type=task_type,
            vectors=cached_vectors,
        )
        print(f"[tag-template] {role}:{cache_name}: cache saved to {cache_path}")
    else:
        print(
            f"[tag-template] {role}:{cache_name}: cache hit {len(deduped_texts)}/{len(deduped_texts)}, no API call needed."
        )

    return {text: cached_vectors[text] for text in deduped_texts}


def compute_f1_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    label_set = sorted(set(true_labels) | set(pred_labels))
    if not label_set:
        return {"micro_f1": 0.0, "macro_f1": 0.0}

    correct = sum(1 for true_label, pred_label in zip(true_labels, pred_labels) if true_label == pred_label)
    total = len(true_labels)
    micro_f1 = correct / total if total else 0.0

    per_label_f1: List[float] = []
    for label in label_set:
        tp = sum(1 for true_label, pred_label in zip(true_labels, pred_labels) if true_label == label and pred_label == label)
        fp = sum(1 for true_label, pred_label in zip(true_labels, pred_labels) if true_label != label and pred_label == label)
        fn = sum(1 for true_label, pred_label in zip(true_labels, pred_labels) if true_label == label and pred_label != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            per_label_f1.append(0.0)
        else:
            per_label_f1.append(2 * precision * recall / (precision + recall))

    return {
        "micro_f1": micro_f1,
        "macro_f1": safe_mean(per_label_f1),
    }


def evaluate_template(
    template: Dict[str, str],
    candidate_labels: Sequence[str],
    candidate_vectors: Dict[str, List[float]],
    query_vectors: Dict[str, List[float]],
    query_template: str,
    examples: Sequence[MappingExample],
) -> Dict[str, object]:
    candidate_render_map = {
        label: render_text(template["template"], label)
        for label in candidate_labels
    }

    example_results: List[Dict[str, object]] = []
    top1_predictions: List[str] = []
    true_labels: List[str] = []

    for example in examples:
        query_rendered = render_text(query_template, example.query_text)
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
        top5_labels = ranked_labels[:5]

        top1_predictions.append(top1_label)
        true_labels.append(example.target_label)

        example_results.append(
            {
                "query_text": example.query_text,
                "target_label": example.target_label,
                "group_id": example.group_id,
                "source_type": example.source_type,
                "top1_prediction": top1_label,
                "top1_hit": top1_label == example.target_label,
                "top5_hit": example.target_label in top5_labels,
                "rank": target_rank,
                "reciprocal_rank": 1.0 / target_rank,
                "top5_predictions": [
                    {"label": label, "score": score}
                    for label, score in ranked[:5]
                ],
            }
        )

    f1_metrics = compute_f1_metrics(true_labels, top1_predictions)

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
                "top5_accuracy": safe_mean(1.0 if result["top5_hit"] else 0.0 for result in label_results),
                "mean_rank": safe_mean(float(result["rank"]) for result in label_results),
                "mrr": safe_mean(float(result["reciprocal_rank"]) for result in label_results),
                "top_confusions": [
                    {"label": wrong_label, "count": count}
                    for wrong_label, count in wrong_predictions.most_common(3)
                ],
            }
        )

    per_label_results.sort(
        key=lambda item: (
            -float(item["top1_accuracy"]),
            -float(item["mrr"]),
            float(item["mean_rank"]),
            str(item["label"]),
        )
    )

    return {
        "name": template["name"],
        "template": template["template"],
        "candidate_count": len(candidate_labels),
        "query_count": len(example_results),
        "top1_accuracy": safe_mean(1.0 if result["top1_hit"] else 0.0 for result in example_results),
        "top5_accuracy": safe_mean(1.0 if result["top5_hit"] else 0.0 for result in example_results),
        "mrr": safe_mean(float(result["reciprocal_rank"]) for result in example_results),
        "micro_f1": f1_metrics["micro_f1"],
        "macro_f1": f1_metrics["macro_f1"],
        "per_label": per_label_results,
        "best_labels": per_label_results[:5],
        "worst_labels": list(
            sorted(
                per_label_results,
                key=lambda item: (
                    float(item["top1_accuracy"]),
                    float(item["mrr"]),
                    -float(item["mean_rank"]),
                    str(item["label"]),
                ),
            )[:5]
        ),
        "examples": example_results,
    }


def make_report_markdown(
    dataset: Sequence[LabelRecord],
    examples: Sequence[MappingExample],
    candidate_labels: Sequence[str],
    templates: Sequence[Dict[str, str]],
    results: Sequence[Dict[str, object]],
    model: str,
    query_task_type: str,
    candidate_task_type: str,
    query_template: str,
    use_symmetric_templates: bool,
) -> str:
    lines = [
        "# Tag Template Mapping Experiment",
        "",
        f"- evaluation labels: {len(dataset)}",
        f"- mapping queries: {len(examples)}",
        f"- candidate canonical tags: {len(candidate_labels)}",
        f"- templates evaluated: {len(templates)}",
        f"- embedding model: `{model}`",
        f"- query task type: `{query_task_type}`",
        f"- candidate task type: `{candidate_task_type}`",
        f"- fixed query template: `{query_template}`",
        f"- symmetric templates: `{use_symmetric_templates}`",
        "",
        "## Ranking",
        "",
        "| Rank | Name | Template | Top-1 | Top-5 | MRR | Macro-F1 | Micro-F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    baseline_top1 = None
    for item in results:
        if item["name"] == "raw_label":
            baseline_top1 = float(item["top1_accuracy"])
            break

    for index, item in enumerate(results, start=1):
        gain = ""
        if baseline_top1 is not None:
            gain = f" ({float(item['top1_accuracy']) - baseline_top1:+.4f} vs raw)"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    str(item["name"]),
                    f"`{item['template']}`",
                    f"{float(item['top1_accuracy']):.4f}{gain}",
                    f"{float(item['top5_accuracy']):.4f}",
                    f"{float(item['mrr']):.4f}",
                    f"{float(item['macro_f1']):.4f}",
                    f"{float(item['micro_f1']):.4f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Diagnostics", ""])

    for item in results[:3]:
        lines.append(f"### {item['name']} - `{item['template']}`")
        lines.append("")
        lines.append("Best labels:")
        for label in item["best_labels"]:
            lines.append(
                f"- {label['label']}: top1={float(label['top1_accuracy']):.4f}, top5={float(label['top5_accuracy']):.4f}, mrr={float(label['mrr']):.4f}"
            )
        lines.append("")
        lines.append("Worst labels:")
        for label in item["worst_labels"]:
            confusion_text = ", ".join(
                f"{entry['label']}({entry['count']})" for entry in label["top_confusions"]
            ) or "none"
            lines.append(
                f"- {label['label']}: top1={float(label['top1_accuracy']):.4f}, top5={float(label['top5_accuracy']):.4f}, mrr={float(label['mrr']):.4f}, confusions={confusion_text}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def preview_setup(
    dataset: Sequence[LabelRecord],
    examples: Sequence[MappingExample],
    candidate_labels: Sequence[str],
    templates: Sequence[Dict[str, str]],
    query_template: str,
    use_symmetric_templates: bool,
) -> None:
    print(f"Loaded {len(dataset)} evaluation labels and {len(examples)} mapping queries.")
    print(f"Candidate canonical tags: {len(candidate_labels)}")
    print(f"Templates: {len(templates)}")
    print(f"Fixed query template: {query_template}")
    print(f"Symmetric templates: {use_symmetric_templates}")
    for example in examples[:5]:
        payload = {
            "query_text": example.query_text,
            "rendered_query": render_text(query_template, example.query_text),
            "target_label": example.target_label,
            "source_type": example.source_type,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    config = resolve_runtime_config(EXPERIMENT_CONFIG)
    source = Path(config["source"])
    candidate_tags_source = Path(config["candidate_tags_source"])
    output_dir = Path(config["output_dir"])
    templates_file = Path(config["templates_file"]) if config["templates_file"] else None
    query_template = str(config["query_template"])
    include_canonical_label_queries = bool(config["include_canonical_label_queries"])
    max_items = int(config["max_items"]) if config["max_items"] is not None else None
    query_task_type = str(config["query_task_type"])
    candidate_task_type = str(config["candidate_task_type"])
    model = str(config["model"])
    batch_size = int(config["batch_size"])
    preview = bool(config["preview"])

    templates = load_templates(templates_file)
    parsed_records = load_source_records(source)
    dataset = build_dataset(parsed_records=parsed_records, max_items=max_items)
    candidate_labels = load_candidate_tags(candidate_tags_source, dataset)
    examples = build_mapping_examples(
        dataset=dataset,
        include_canonical_label_queries=include_canonical_label_queries,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_payload = {
        "source": str(source),
        "candidate_tags_source": str(candidate_tags_source),
        "label_count": len(dataset),
        "query_count": len(examples),
        "candidate_count": len(candidate_labels),
        "query_template": query_template,
        "include_canonical_label_queries": include_canonical_label_queries,
        "labels": [asdict(item) for item in dataset],
        "examples": [asdict(item) for item in examples],
        "candidate_labels": list(candidate_labels),
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if preview:
        preview_setup(dataset, examples, candidate_labels, templates, query_template, bool(config.get("use_symmetric_templates")))
        print(f"Dataset exported to {output_dir / 'dataset.json'}")
        return

    query_rendered_texts = [render_text(query_template, example.query_text) for example in examples]
    query_vectors = get_rendered_vectors(
        rendered_texts=query_rendered_texts,
        output_dir=output_dir,
        cache_name="mapping_queries",
        template_text=query_template,
        role="query",
        model=model,
        task_type=query_task_type,
        batch_size=batch_size,
    )

    results: List[Dict[str, object]] = []
    for template in templates:
        # Determine query template for this run
        current_query_template = query_template
        if config.get("use_symmetric_templates"):
            current_query_template = template["template"]

        # Re-embed queries if symmetric or if it's the first time
        current_query_vectors = query_vectors
        if config.get("use_symmetric_templates"):
            query_rendered_texts = [render_text(current_query_template, example.query_text) for example in examples]
            current_query_vectors = get_rendered_vectors(
                rendered_texts=query_rendered_texts,
                output_dir=output_dir,
                cache_name=f"queries_{template['name']}",
                template_text=current_query_template,
                role="query",
                model=model,
                task_type=query_task_type,
                batch_size=batch_size,
            )

        candidate_rendered_texts = [render_text(template["template"], label) for label in candidate_labels]
        candidate_vectors = get_rendered_vectors(
            rendered_texts=candidate_rendered_texts,
            output_dir=output_dir,
            cache_name=template["name"],
            template_text=template["template"],
            role="candidate",
            model=model,
            task_type=candidate_task_type,
            batch_size=batch_size,
        )
        results.append(
            evaluate_template(
                template=template,
                candidate_labels=candidate_labels,
                candidate_vectors=candidate_vectors,
                query_vectors=current_query_vectors,
                query_template=current_query_template,
                examples=examples,
            )
        )

    results.sort(
        key=lambda item: (
            float(item["top1_accuracy"]),
            float(item["mrr"]),
            float(item["top5_accuracy"]),
            float(item["macro_f1"]),
        ),
        reverse=True,
    )

    report_payload = {
        "source": str(source),
        "candidate_tags_source": str(candidate_tags_source),
        "model": model,
        "query_task_type": query_task_type,
        "candidate_task_type": candidate_task_type,
        "query_template": query_template,
        "include_canonical_label_queries": include_canonical_label_queries,
        "templates": templates,
        "dataset": dataset_payload,
        "results": results,
    }
    report_json_path = output_dir / "results.json"
    report_md_path = output_dir / "summary.md"
    report_json_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md_path.write_text(
        make_report_markdown(
            dataset=dataset,
            examples=examples,
            candidate_labels=candidate_labels,
            templates=templates,
            results=results,
            model=model,
            query_task_type=query_task_type,
            candidate_task_type=candidate_task_type,
            query_template=query_template,
            use_symmetric_templates=bool(config.get("use_symmetric_templates")),
        ),
        encoding="utf-8",
    )

    top = results[0]
    print(
        f"Best template: {top['name']} -> {top['template']} "
        f"(top1={float(top['top1_accuracy']):.4f}, top5={float(top['top5_accuracy']):.4f}, "
        f"mrr={float(top['mrr']):.4f}, macro_f1={float(top['macro_f1']):.4f})"
    )
    print(f"Results saved to {report_json_path}")
    print(f"Markdown summary saved to {report_md_path}")


if __name__ == "__main__":
    main()
