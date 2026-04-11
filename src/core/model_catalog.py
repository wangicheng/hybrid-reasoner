from __future__ import annotations

import re
from typing import Iterable, List, Tuple


DEFAULT_MODEL_ID = "gemma-3-27b-it"

MODEL_OPTIONS: List[Tuple[str, str]] = [
    ("gemma-3-27b-it", "Gemma 3 27B"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
    ("gemini-3-flash-preview", "Gemini 3 Flash Preview"),
]

FALLBACK_MODELS = [model_id for model_id, _label in MODEL_OPTIONS]
JUDGE_MODELS = list(FALLBACK_MODELS)


def normalize_model_id(model_id: str | None) -> str:
    value = (model_id or "").strip()
    return value or DEFAULT_MODEL_ID


def build_model_candidates(
    selected_model: str | None,
    fallback_models: Iterable[str] | None = None,
) -> List[str]:
    fallback = list(fallback_models or FALLBACK_MODELS)
    normalized = normalize_model_id(selected_model)

    if normalized in fallback:
        return [normalized] + [model for model in fallback if model != normalized]

    return [normalized] + fallback


def sanitize_model_tag(model_id: str | None) -> str:
    normalized = normalize_model_id(model_id).lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
