from __future__ import annotations

import re
from typing import Iterable, List, Tuple


# The model used by default for parsing and generation when none is specified.
DEFAULT_MODEL_ID = "gemma-4-31b-it"

AVAILABLE_MODELS: List[Tuple[str, str]] = [
    ("gemma-4-31b-it", "Gemma 4 31B"),
    ("gemini-1.5-flash", "Gemini 1.5 Flash"),
    ("gemini-3-flash-preview", "Gemini 3 Flash Preview"),
]

MODEL_PRIORITY_ORDER = [model_id for model_id, _label in AVAILABLE_MODELS]
JUDGE_MODELS = list(MODEL_PRIORITY_ORDER)


def normalize_model_id(model_id: str | None) -> str:
    value = (model_id or "").strip()
    return value or DEFAULT_MODEL_ID


def build_model_priority(
    selected_model: str | None,
    candidate_models: Iterable[str] | None = None,
) -> List[str]:
    model_order = list(candidate_models or MODEL_PRIORITY_ORDER)
    normalized = normalize_model_id(selected_model)

    if normalized in model_order:
        return [normalized] + [model for model in model_order if model != normalized]

    return [normalized] + model_order


def sanitize_model_tag(model_id: str | None) -> str:
    normalized = normalize_model_id(model_id).lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
