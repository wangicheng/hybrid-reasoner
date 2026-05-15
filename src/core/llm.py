import json
import functools
import os
import time
import re
import threading
import ast
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from google import genai
from google.genai import types
from pydantic import BaseModel as PydanticBaseModel
from src.models.schemas import QueryParseResult, ScoringCriteria, ScoringParameters, TagIntent
from src.core.api_utils import is_rate_limit_error

# ??????謘?(???????制??謅???????剜???蹇?????
DEFAULT_PARSER_MODEL = "gemma-4-31b-it"
PARSER_VARIANT_SEMANTIC_SECTIONS_V3_TAGLITE = "semantic_sections_v3_taglite"
DEFAULT_PARSER_VARIANT = PARSER_VARIANT_SEMANTIC_SECTIONS_V3_TAGLITE


def _load_llm_timeout_seconds() -> Optional[float]:
    raw_value = str(os.getenv("HYBRID_REASONER_LLM_TIMEOUT_SECONDS", "")).strip()
    if not raw_value:
        return 180.0

    if raw_value.lower() in {"0", "none", "off", "disable", "disabled"}:
        return None

    try:
        timeout_seconds = float(raw_value)
    except ValueError:
        return 180.0
    return timeout_seconds if timeout_seconds > 0 else None


LLM_REQUEST_TIMEOUT_SECONDS: Optional[float] = _load_llm_timeout_seconds()
LLM_RETRY_DELAY_SECONDS = 10.0

DEBUG_LLM_OUTPUT = True
_T = TypeVar("_T")


class ParserBranchError(RuntimeError):
    def __init__(
        self,
        task_label: str,
        message: str,
        branch_metadata: Dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.task_label = task_label
        self.branch_metadata = branch_metadata
        self.parser_metadata: Dict[str, Any] = {}


def _parser_mode_for_variant() -> str:
    return "semantic_sections_v3_taglite_three_call"


def _augment_exception_with_call_metadata(exc: Exception, call_metadata: Dict[str, Any]) -> None:
    existing = getattr(exc, "llm_call_metadata", None)
    if isinstance(existing, dict):
        existing.update(call_metadata)
    else:
        setattr(exc, "llm_call_metadata", call_metadata)


def _call_with_timeout(
    func: Callable[[], _T],
    timeout_seconds: Optional[float],
    label: str,
) -> _T:
    if timeout_seconds is None or timeout_seconds <= 0:
        return func()

    result: Dict[str, _T] = {}
    error: Dict[str, Exception] = {}
    completed = threading.Event()

    def _runner() -> None:
        try:
            result["value"] = func()
        except Exception as exc:  # pragma: no cover - exercised through callers
            error["value"] = exc
        finally:
            completed.set()

    thread = threading.Thread(target=_runner, name=f"llm-timeout-{label}", daemon=True)
    thread.start()
    if not completed.wait(timeout_seconds):
        raise TimeoutError(f"{label} timed out after {timeout_seconds:.1f}s")
    if "value" in error:
        raise error["value"]
    return result["value"]


def _is_null_like_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "null", "none", "nil", "n/a", "na", "unknown", "unspecified"}


def _extract_text_from_response(response: Any) -> str:
    text_chunks: List[str] = []

    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        text_chunks.append(response_text)

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                text_chunks.append(part_text)

    return "\n".join(chunk for chunk in text_chunks if chunk).strip()


def _parse_json_object_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidates = []
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())
    candidates.append(text)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1].strip())

    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(normalized)
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _repair_structured_payload_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    candidate_keys = [
        "target_status_candidate",
        "author_name_candidate",
        "words_min_candidate",
        "words_max_candidate",
    ]

    repaired: Dict[str, Any] = {}
    for candidate_key in candidate_keys:
        pattern = re.compile(
            rf'"?{re.escape(candidate_key)}"?\s*:\s*(null|\{{[^{{}}]*\}})',
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            continue

        candidate_text = match.group(1).strip()
        if candidate_text.lower() == "null":
            repaired[candidate_key] = {
                "value": "" if "words_" not in candidate_key else 0,
                "evidence": "",
                "is_explicit": False,
            }
            continue

        parsed_candidate = _parse_json_object_from_text(candidate_text)
        if isinstance(parsed_candidate, dict):
            repaired[candidate_key] = parsed_candidate

    return repaired or None


def _parse_marked_section_output(
    raw_text: str,
    expected_sections: List[str],
) -> Dict[str, str]:
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Marked section output was empty")

    section_pattern = re.compile(r"^\[(?P<name>[a-z_]+)\]\s*$", re.MULTILINE)
    matches = list(section_pattern.finditer(text))
    if not matches:
        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Marked section output missing section headers; raw_text={preview}")

    sections: Dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group("name")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    missing = [name for name in expected_sections if name not in sections]
    if missing:
        raise ValueError(f"Marked section output missing sections: {', '.join(missing)}")

    return sections


def _parse_semantic_sections_text(raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sections = _parse_marked_section_output(
        raw_text,
        expected_sections=[
            "semantic_query_text",
            "positive_concepts",
            "negative_concepts",
        ],
    )

    def _parse_list_section(section_text: str) -> List[str]:
        items: List[str] = []
        for raw_line in str(section_text or "").splitlines():
            cleaned = re.sub(r"^\s*[-*•]\s*", "", raw_line).strip()
            if not cleaned:
                continue
            if "," in cleaned and not any(token in cleaned for token in ("、", "，")):
                parts = [part.strip() for part in cleaned.split(",")]
                items.extend(part for part in parts if part)
                continue
            items.append(cleaned)
        return _dedupe_terms(items)

    parsed = {
        "semantic_query_text": str(sections["semantic_query_text"]).strip(),
        "positive_concepts": _parse_list_section(sections["positive_concepts"]),
        "negative_concepts": _parse_list_section(sections["negative_concepts"]),
    }
    return parsed, {
        "parse_source": "marked_sections",
        "recovered_from_raw_text": False,
    }


def _parse_semantic_sections_v3_text(raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sections = _parse_marked_section_output(
        raw_text,
        expected_sections=[
            "semantic_query_text",
            "intent_summary",
            "positive_concepts",
            "negative_concepts",
        ],
    )

    parsed, _ = _parse_semantic_sections_text(
        "\n".join(
            [
                "[semantic_query_text]",
                str(sections["semantic_query_text"]).strip(),
                "[positive_concepts]",
                str(sections["positive_concepts"]).strip(),
                "[negative_concepts]",
                str(sections["negative_concepts"]).strip(),
            ]
        )
    )
    parsed["intent_summary"] = str(sections["intent_summary"]).strip()
    return parsed, {
        "parse_source": "marked_sections_v3",
        "recovered_from_raw_text": False,
    }


def _coerce_response_to_json_object(
    response: Any,
    task_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, PydanticBaseModel):
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump(), {
                "parse_source": "sdk_parsed_pydantic",
                "recovered_from_raw_text": False,
            }
        return parsed.dict(), {
            "parse_source": "sdk_parsed_pydantic",
            "recovered_from_raw_text": False,
        }
    if isinstance(parsed, dict):
        return parsed, {
            "parse_source": "sdk_parsed_dict",
            "recovered_from_raw_text": False,
        }
    if parsed is not None and hasattr(parsed, "model_dump"):
        return parsed.model_dump(), {
            "parse_source": "sdk_parsed_model_dump",
            "recovered_from_raw_text": False,
        }
    if parsed is not None and hasattr(parsed, "dict"):
        return parsed.dict(), {
            "parse_source": "sdk_parsed_dict_method",
            "recovered_from_raw_text": False,
        }

    raw_text = _extract_text_from_response(response)
    reparsed = _parse_json_object_from_text(raw_text)
    if reparsed is not None:
        print(f"[llm:{task_label}] recovered JSON from raw text because parsed payload was missing.")
        return reparsed, {
            "parse_source": "raw_text_json",
            "recovered_from_raw_text": True,
        }

    if task_label == "structured":
        repaired = _repair_structured_payload_from_text(raw_text)
        if repaired is not None:
            print(f"[llm:{task_label}] repaired structured payload from raw text fragments.")
            return repaired, {
                "parse_source": "structured_text_repair",
                "recovered_from_raw_text": True,
            }

    raw_preview = raw_text[:200].replace("\n", "\\n") if raw_text else "(empty)"
    raise ValueError(
        f"Structured output missing parsed payload for {task_label}; raw_text={raw_preview}"
    )


def _coerce_response_to_text(response: Any, task_label: str) -> Tuple[str, Dict[str, Any]]:
    raw_text = _extract_text_from_response(response)
    if raw_text:
        return raw_text, {
            "parse_source": "raw_text",
            "recovered_from_raw_text": False,
        }

    raise ValueError(f"Text output missing raw text for {task_label}")
def _query_contains_evidence(user_query: str, evidence: Any) -> bool:
    query = str(user_query or "").strip()
    snippet = str(evidence or "").strip()
    if not query or not snippet:
        return False
    return snippet in query




def _dedupe_terms(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        normalized_value = str(value).replace(" ", "").strip()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        deduped.append(str(value).strip())
    return deduped


def _compose_retrieval_search_terms(
    user_query: str,
    semantic_query_text: str,
    positive_terms: List[str],
    max_positive_terms: int = 6,
) -> str:
    tag_like_terms = _dedupe_terms([str(term).strip() for term in positive_terms])[:max_positive_terms]
    base_text = str(semantic_query_text or "").strip()

    segments: List[str] = []
    segments.extend(tag_like_terms)
    if base_text:
        segments.append(base_text)

    if segments:
        return " ".join(segments).strip()

    return str(user_query or "").strip() or user_query


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        values: List[str] = []
        for item in value:
            values.extend(_coerce_string_list(item))
        return values
    if isinstance(value, dict):
        values: List[str] = []
        for candidate_key in ("text", "value", "name", "query_text", "keyword"):
            if candidate_key in value:
                values.extend(_coerce_string_list(value[candidate_key]))
        return values
    text = str(value).strip()
    return [text] if text else []


def _normalize_semantic_understanding(parsed: Any, user_query: str) -> Dict[str, Any]:
    normalized = parsed if isinstance(parsed, dict) else {}

    semantic_query_text = str(
        normalized.get("semantic_query_text")
        or normalized.get("search_terms")
        or user_query
    ).strip() or user_query
    intent_summary = str(
        normalized.get("intent_summary")
        or normalized.get("summary")
        or ""
    ).strip()

    positive_concepts = _dedupe_terms(
        _coerce_string_list(
            normalized.get("positive_concepts")
            or normalized.get("keyword_candidates")
            or normalized.get("generated_keywords")
        )
    )
    negative_concepts = _dedupe_terms(
        _coerce_string_list(
            normalized.get("negative_concepts")
            or normalized.get("excluded_keywords")
            or normalized.get("negative_keywords")
        )
    )
    ambiguities = _dedupe_terms(
        _coerce_string_list(normalized.get("ambiguities"))
    )

    return {
        "semantic_query_text": semantic_query_text,
        "intent_summary": intent_summary,
        "positive_concepts": positive_concepts,
        "negative_concepts": negative_concepts,
        "ambiguities": ambiguities,
    }


def _build_tag_projection_context(semantic_understanding: Dict[str, Any]) -> str:
    semantic_query_text = str(semantic_understanding.get("semantic_query_text") or "").strip()
    intent_summary = str(semantic_understanding.get("intent_summary") or "").strip()
    positive_concepts = [
        str(term).strip()
        for term in semantic_understanding.get("positive_concepts", [])
        if str(term).strip()
    ]
    negative_concepts = [
        str(term).strip()
        for term in semantic_understanding.get("negative_concepts", [])
        if str(term).strip()
    ]
    ambiguities = [
        str(term).strip()
        for term in semantic_understanding.get("ambiguities", [])
        if str(term).strip()
    ]

    lines = [
        "SEMANTIC UNDERSTANDING OUTPUT:",
        f"- semantic_query_text: {semantic_query_text or '(empty)'}",
        f"- intent_summary: {intent_summary or '(empty)'}",
        f"- positive_concepts: {', '.join(positive_concepts) if positive_concepts else '(empty)'}",
        f"- negative_concepts: {', '.join(negative_concepts) if negative_concepts else '(empty)'}",
        f"- ambiguities: {', '.join(ambiguities) if ambiguities else '(empty)'}",
    ]
    return "\n".join(lines)


def _build_tag_projection_compact_context(semantic_understanding: Dict[str, Any]) -> str:
    semantic_query_text = str(semantic_understanding.get("semantic_query_text") or "").strip()
    intent_summary = str(semantic_understanding.get("intent_summary") or "").strip()
    positive_concepts = [
        str(term).strip()
        for term in semantic_understanding.get("positive_concepts", [])
        if str(term).strip()
    ][:3]
    negative_concepts = [
        str(term).strip()
        for term in semantic_understanding.get("negative_concepts", [])
        if str(term).strip()
    ][:6]

    lines = [
        "SEMANTIC UNDERSTANDING OUTPUT (COMPACT):",
        f"- semantic_query_text: {semantic_query_text or '(empty)'}",
        f"- intent_summary: {intent_summary or '(empty)'}",
        f"- strongest_positive_concepts: {', '.join(positive_concepts) if positive_concepts else '(empty)'}",
        f"- strongest_negative_concepts: {', '.join(negative_concepts) if negative_concepts else '(empty)'}",
    ]
    return "\n".join(lines)


def _normalize_tag_projection(parsed: Any) -> Dict[str, Any]:
    normalized = parsed if isinstance(parsed, dict) else {}
    positive_terms = _dedupe_terms(
        _coerce_string_list(
            normalized.get("positive_terms")
            or normalized.get("generated_keywords")
            or normalized.get("positive_tags")
        )
    )
    negative_terms = _dedupe_terms(
        _coerce_string_list(
            normalized.get("negative_terms")
            or normalized.get("excluded_keywords")
            or normalized.get("negative_tags")
        )
    )

    return {
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
    }


def _build_tag_intent_from_projection(
    user_query: str,
    semantic_understanding: Dict[str, Any],
    tag_projection: Dict[str, Any],
) -> TagIntent:
    positive_terms = list(tag_projection.get("positive_terms") or [])
    negative_terms = list(tag_projection.get("negative_terms") or [])

    if not positive_terms:
        positive_terms = list(semantic_understanding.get("positive_concepts") or [])
    if not negative_terms:
        negative_terms = list(semantic_understanding.get("negative_concepts") or [])

    deduped_positive_terms = _dedupe_terms([str(term).strip() for term in positive_terms])
    deduped_negative_terms = _dedupe_terms([str(term).strip() for term in negative_terms])
    search_terms = _compose_retrieval_search_terms(
        user_query=user_query,
        semantic_query_text=str(semantic_understanding.get("semantic_query_text") or ""),
        positive_terms=deduped_positive_terms,
    )

    return TagIntent(
        search_terms=search_terms,
        positive_terms=deduped_positive_terms,
        negative_terms=deduped_negative_terms,
    )


def _build_structured_context_from_semantic_understanding(
    semantic_understanding: Dict[str, Any],
    tag_projection: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        _build_tag_projection_context(semantic_understanding),
        "Use this understanding only as supporting context for hard constraints.",
        "Do not invent completion status, author, or word-count filters unless the query directly states them.",
    ]

    if tag_projection is not None:
        positive_terms = [
            str(term).strip()
            for term in tag_projection.get("positive_terms", [])
            if str(term).strip()
        ]
        negative_terms = [
            str(term).strip()
            for term in tag_projection.get("negative_terms", [])
            if str(term).strip()
        ]
        lines.extend(
            [
                "TAG PROJECTION OUTPUT:",
                f"- positive_terms: {', '.join(positive_terms) if positive_terms else '(empty)'}",
                f"- negative_terms: {', '.join(negative_terms) if negative_terms else '(empty)'}",
            ]
        )

    return "\n".join(lines)




def _normalize_structured_draft(parsed: Any) -> Dict[str, Any]:
    normalized = parsed if isinstance(parsed, dict) else {}

    def _normalize_string_candidate(name: str) -> Dict[str, Any]:
        candidate = normalized.get(name)
        if not isinstance(candidate, dict):
            return {}
        value = str(candidate.get("value") or "").strip()
        evidence = str(candidate.get("evidence") or "").strip()
        is_explicit = bool(candidate.get("is_explicit"))
        if not value:
            return {}
        if not is_explicit and (_is_null_like_text(value) or _is_null_like_text(evidence)):
            return {}
        if name == "target_status_candidate" and value.lower() not in {"completed", "ongoing"}:
            return {}
        return {"value": value, "evidence": evidence, "is_explicit": is_explicit}

    def _normalize_number_candidate(name: str) -> Dict[str, Any]:
        candidate = normalized.get(name)
        if not isinstance(candidate, dict):
            return {}
        evidence = str(candidate.get("evidence") or "").strip()
        is_explicit = bool(candidate.get("is_explicit"))
        try:
            numeric_value = float(candidate.get("value"))
        except (TypeError, ValueError):
            return {}
        if not is_explicit and (_is_null_like_text(evidence) or numeric_value == 0.0):
            return {}
        return {"value": numeric_value, "evidence": evidence, "is_explicit": is_explicit}

    return {
        "target_status_candidate": _normalize_string_candidate("target_status_candidate"),
        "author_name_candidate": _normalize_string_candidate("author_name_candidate"),
        "words_min_candidate": _normalize_number_candidate("words_min_candidate"),
        "words_max_candidate": _normalize_number_candidate("words_max_candidate"),
    }


def _structured_candidates_to_slots(structured_candidates: Dict[str, Any]) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}

    status_candidate = structured_candidates.get("target_status_candidate", {})
    status_value = str(status_candidate.get("value") or "").strip().lower()
    if status_value in {"completed", "ongoing"}:
        slots["target_status"] = status_value

    author_candidate = structured_candidates.get("author_name_candidate", {})
    author_value = str(author_candidate.get("value") or "").strip()
    if author_value:
        slots["author_name"] = author_value

    words_min_candidate = structured_candidates.get("words_min_candidate", {})
    words_min = words_min_candidate.get("value")
    if words_min is not None:
        slots["words_min"] = words_min

    words_max_candidate = structured_candidates.get("words_max_candidate", {})
    words_max = words_max_candidate.get("value")
    if words_max is not None:
        slots["words_max"] = words_max

    if "words_min" in slots or "words_max" in slots:
        slots["field"] = "words_total"

    return slots


def _apply_structured_draft_guards(
    user_query: str,
    structured_slots: Dict[str, Any],
    structured_draft: Dict[str, Any],
) -> Dict[str, Any]:
    guarded: Dict[str, Any] = {}

    status_candidate = structured_draft.get("target_status_candidate", {})
    status_value = str(structured_slots.get("target_status") or "").strip().lower()
    if (
        status_value in {"completed", "ongoing"}
        and status_candidate.get("is_explicit")
        and _query_contains_evidence(user_query, status_candidate.get("evidence"))
    ):
        guarded["target_status"] = status_value

    author_candidate = structured_draft.get("author_name_candidate", {})
    author_value = str(structured_slots.get("author_name") or "").strip()
    if (
        author_value
        and author_candidate.get("is_explicit")
        and _query_contains_evidence(user_query, author_candidate.get("evidence"))
        and author_value == str(author_candidate.get("value") or "").strip()
    ):
        guarded["author_name"] = author_value

    words_min_candidate = structured_draft.get("words_min_candidate", {})
    words_min = structured_slots.get("words_min")
    if (
        words_min is not None
        and words_min_candidate.get("is_explicit")
        and _query_contains_evidence(user_query, words_min_candidate.get("evidence"))
    ):
        guarded["words_min"] = words_min

    words_max_candidate = structured_draft.get("words_max_candidate", {})
    words_max = structured_slots.get("words_max")
    if (
        words_max is not None
        and words_max_candidate.get("is_explicit")
        and _query_contains_evidence(user_query, words_max_candidate.get("evidence"))
    ):
        guarded["words_max"] = words_max

    if "words_min" in guarded or "words_max" in guarded:
        guarded["field"] = "words_total"

    return guarded

def _merge_structured_slots(structured_slots: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    target_status = structured_slots.get("target_status")
    author_name = structured_slots.get("author_name")
    words_min = structured_slots.get("words_min")
    words_max = structured_slots.get("words_max")

    if target_status is not None:
        merged["target_status"] = target_status
    if author_name is not None:
        merged["author_name"] = author_name
    if words_min is not None:
        merged["words_min"] = words_min
    if words_max is not None:
        merged["words_max"] = words_max
    if words_min is not None or words_max is not None:
        merged["field"] = "words_total"

    return merged



def _criteria_to_key(criteria: Any) -> Tuple[str, bool, str]:
    if isinstance(criteria, PydanticBaseModel):
        if hasattr(criteria, "model_dump"):
            payload = criteria.model_dump()
        else:
            payload = criteria.dict()
    elif hasattr(criteria, "model_dump"):
        payload = criteria.model_dump()
    elif hasattr(criteria, "dict"):
        payload = criteria.dict()
    else:
        payload = dict(criteria)

    return (
        str(payload.get("name", "")),
        bool(payload.get("is_negative", False)),
        json.dumps(payload.get("parameters", {}), ensure_ascii=False, sort_keys=True),
    )


def _merge_query_parse_results(
    user_query: str,
    tag_intent: TagIntent,
    structured_slots: Dict[str, Any],
    total_latency_ms: float,
    branch_metrics: Dict[str, Dict[str, Any]],
) -> QueryParseResult:
    semantic_query_text = str(tag_intent.search_terms or user_query).strip() or user_query
    merged_criteria = [
        ScoringCriteria(
            name="semantic_similarity",
            is_negative=False,
            parameters=ScoringParameters(query_text=semantic_query_text),
        )
    ]
    seen_criteria = set()
    for criteria in merged_criteria:
        seen_criteria.add(_criteria_to_key(criteria))

    for positive_term in tag_intent.positive_terms[:6]:
        positive_text = str(positive_term).strip()
        if not positive_text:
            continue
        criteria = ScoringCriteria(
            name="semantic_similarity",
            is_negative=False,
            parameters=ScoringParameters(query_text=positive_text),
        )
        key = _criteria_to_key(criteria)
        if key not in seen_criteria:
            seen_criteria.add(key)
            merged_criteria.append(criteria)

    for excluded_keyword in tag_intent.negative_terms:
        excluded_text = str(excluded_keyword).strip()
        if not excluded_text:
            continue
        criteria = ScoringCriteria(
            name="semantic_similarity",
            is_negative=True,
            parameters=ScoringParameters(query_text=excluded_text),
        )
        key = _criteria_to_key(criteria)
        if key not in seen_criteria:
            seen_criteria.add(key)
            merged_criteria.append(criteria)

    target_status = str(structured_slots.get("target_status") or "").strip()
    if target_status:
        criteria = ScoringCriteria(
            name="status_check",
            is_negative=False,
            parameters=ScoringParameters(target_status=target_status),
        )
        key = _criteria_to_key(criteria)
        if key not in seen_criteria:
            seen_criteria.add(key)
            merged_criteria.append(criteria)

    author_name = str(structured_slots.get("author_name") or "").strip()
    if author_name:
        criteria = ScoringCriteria(
            name="author_match",
            is_negative=False,
            parameters=ScoringParameters(author_name=author_name),
        )
        key = _criteria_to_key(criteria)
        if key not in seen_criteria:
            seen_criteria.add(key)
            merged_criteria.append(criteria)

    words_min = structured_slots.get("words_min")
    words_max = structured_slots.get("words_max")
    if words_min is not None or words_max is not None:
        criteria = ScoringCriteria(
            name="numeric_range",
            is_negative=False,
            parameters=ScoringParameters(
                field="words_total",
                min_val=words_min,
                max_val=words_max,
            ),
        )
        key = _criteria_to_key(criteria)
        if key not in seen_criteria:
            seen_criteria.add(key)
            merged_criteria.append(criteria)

    metadata = {
        "latency_ms": round(total_latency_ms, 2),
        "branches": branch_metrics,
        "parser_mode": _parser_mode_for_variant(),
        "parser_variant": DEFAULT_PARSER_VARIANT,
        "task_split": "semantic_understanding_tag_projection_structured",
        "total_request_count": sum(
            int(branch.get("request_count", 0) or 0)
            for branch in branch_metrics.values()
        ),
        "total_retry_count": sum(
            int(branch.get("retry_count", 0) or 0)
            for branch in branch_metrics.values()
        ),
        "tag_intent": {
            "positive_count": len(tag_intent.positive_terms),
            "negative_count": len(tag_intent.negative_terms),
        },
    }

    return QueryParseResult(
        original_query=user_query,
        search_terms=semantic_query_text,
        generated_keywords=list(tag_intent.positive_terms),
        tag_intent=tag_intent,
        hypothetical_intro="",
        criteria=merged_criteria,
        parse_metadata=metadata,
    )


def _build_parallel_context(
    tag_list: Optional[Tuple[str, ...]] = None,
    reference_book_context: Optional[str] = None,
    response_contract: str = "json",
) -> str:
    output_instruction = (
        "Return marked sections only. Do not use JSON."
        if response_contract == "sections"
        else "Return strict JSON only. Use snake_case keys. Do not wrap the JSON in markdown."
    )
    sections = [
        "You are one branch of a web novel query parser.",
        output_instruction,
        "The final system merges outputs from multiple branches, so focus only on your assigned subtask.",
        "Use Traditional Chinese for generated retrieval terms and hypothetical intros unless the query explicitly uses another title language.",
    ]

    if tag_list:
        sections.append(
            "AVAILABLE TAGS:\n"
            "Prefer exact tag names from the list when the user mentions genres, tropes, vibes, or themes.\n"
            f"{', '.join(tag_list)}"
        )

    if reference_book_context:
        sections.append(
            "RELATED BOOKS:\n"
            "Use these only as soft context for interpreting the query. They are not hard retrieval constraints.\n"
            f"{reference_book_context}"
        )

    return "\n\n".join(sections)


def _generate_json_from_contents(
    *,
    contents: str,
    task_label: str,
    system_instruction: str,
    response_schema: Optional[Dict[str, Any]],
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.2,
    enforce_rate_limit: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.core.api_utils import (
        _is_retryable,
        get_api_key_rotator,
        get_current_api_key,
        get_rate_limiter,
    )

    selected_model = str(model_id or DEFAULT_PARSER_MODEL).strip() or DEFAULT_PARSER_MODEL
    use_response_schema = True
    last_exception = None

    print(f"[llm:{task_label}] trying model: {selected_model}")

    rotator = get_api_key_rotator()
    if not getattr(rotator, "api_keys", None):
        raise RuntimeError(f"[llm:{task_label}] no API keys configured")

    request_count = 0
    total_api_keys = len(rotator.api_keys)

    while True:
        api_key = rotator.acquire()
        try:
            key_index = rotator.api_keys.index(api_key) + 1
        except ValueError:
            key_index = 0
        client = genai.Client(api_key=api_key)

        try:
            if enforce_rate_limit:
                print(f"[llm:{task_label}] waiting for rate limiter before request.")
                get_rate_limiter().wait(api_key)

            config_args = {
                "system_instruction": system_instruction,
                "temperature": sampling_temperature,
                "top_p": 0.95,
            }
            if use_response_schema:
                config_args["response_mime_type"] = "application/json"
                config_args["response_schema"] = response_schema

            request_count += 1
            print(
                f"[llm:{task_label}] sending request "
                f"(request {request_count}, key {key_index}/{total_api_keys})..."
            )
            started_at = time.perf_counter()
            response = _call_with_timeout(
                lambda: client.models.generate_content(
                    model=selected_model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_args),
                ),
                timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
                label=f"{task_label}:{selected_model}",
            )
            latency = time.perf_counter() - started_at
            print(f"[llm:{task_label}] response received in {latency:.1f}s.")
            parsed_payload, parse_metadata = _coerce_response_to_json_object(response, task_label)
            return parsed_payload, {
                "request_count": request_count,
                "retry_count": max(0, request_count - 1),
                "first_attempt_success": request_count == 1,
                "used_response_schema": use_response_schema,
                "parse_source": parse_metadata.get("parse_source", "unknown"),
                "recovered_from_raw_text": bool(parse_metadata.get("recovered_from_raw_text")),
                "model_id": selected_model,
                "last_retry_error": str(last_exception) if last_exception else "",
            }
        except Exception as exc:
            last_exception = exc
            print(f"[llm:{task_label}] request failed on key {key_index}/{total_api_keys}: {exc}")
            call_metadata = {
                "request_count": request_count,
                "retry_count": max(0, request_count - 1),
                "first_attempt_success": False,
                "used_response_schema": use_response_schema,
                "parse_source": "failed",
                "recovered_from_raw_text": False,
                "model_id": selected_model,
                "last_retry_error": str(exc),
            }
            _augment_exception_with_call_metadata(exc, call_metadata)
            if _is_retryable(exc):
                # Put the key to sleep so it is temporarily removed from rotation
                rotator.sleep_key(api_key, LLM_RETRY_DELAY_SECONDS)
                
                print(
                    f"[llm:{task_label}] retryable error; sleeping "
                    f"{LLM_RETRY_DELAY_SECONDS:.1f}s before next attempt."
                )
                time.sleep(LLM_RETRY_DELAY_SECONDS)
                continue
            raise
        finally:
            rotator.release(api_key)


def _generate_text_from_contents(
    *,
    contents: str,
    task_label: str,
    system_instruction: str,
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.2,
    enforce_rate_limit: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    from src.core.api_utils import (
        _is_retryable,
        get_api_key_rotator,
        get_current_api_key,
        get_rate_limiter,
    )

    selected_model = str(model_id or DEFAULT_PARSER_MODEL).strip() or DEFAULT_PARSER_MODEL
    last_exception = None

    print(f"[llm:{task_label}] trying model: {selected_model}")

    rotator = get_api_key_rotator()
    if not getattr(rotator, "api_keys", None):
        raise RuntimeError(f"[llm:{task_label}] no API keys configured")

    request_count = 0
    total_api_keys = len(rotator.api_keys)

    while True:
        api_key = rotator.acquire()
        try:
            key_index = rotator.api_keys.index(api_key) + 1
        except ValueError:
            key_index = 0
        client = genai.Client(api_key=api_key)

        try:
            if enforce_rate_limit:
                print(f"[llm:{task_label}] waiting for rate limiter before request.")
                get_rate_limiter().wait(api_key)

            request_count += 1
            print(
                f"[llm:{task_label}] sending request "
                f"(request {request_count}, key {key_index}/{total_api_keys})..."
            )
            started_at = time.perf_counter()
            response = _call_with_timeout(
                lambda: client.models.generate_content(
                    model=selected_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=sampling_temperature,
                        top_p=0.95,
                    ),
                ),
                timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
                label=f"{task_label}:{selected_model}",
            )
            latency = time.perf_counter() - started_at
            print(f"[llm:{task_label}] response received in {latency:.1f}s.")
            raw_text, parse_metadata = _coerce_response_to_text(response, task_label)
            return raw_text, {
                "request_count": request_count,
                "retry_count": max(0, request_count - 1),
                "first_attempt_success": request_count == 1,
                "used_response_schema": False,
                "parse_source": parse_metadata.get("parse_source", "raw_text"),
                "recovered_from_raw_text": False,
                "model_id": selected_model,
                "last_retry_error": str(last_exception) if last_exception else "",
            }
        except Exception as exc:
            last_exception = exc
            print(f"[llm:{task_label}] request failed on key {key_index}/{total_api_keys}: {exc}")
            call_metadata = {
                "request_count": request_count,
                "retry_count": max(0, request_count - 1),
                "first_attempt_success": False,
                "used_response_schema": False,
                "parse_source": "failed",
                "recovered_from_raw_text": False,
                "model_id": selected_model,
                "last_retry_error": str(exc),
            }
            _augment_exception_with_call_metadata(exc, call_metadata)
            if _is_retryable(exc):
                # Put the key to sleep so it is temporarily removed from rotation
                rotator.sleep_key(api_key, LLM_RETRY_DELAY_SECONDS)
                
                print(
                    f"[llm:{task_label}] retryable error; sleeping "
                    f"{LLM_RETRY_DELAY_SECONDS:.1f}s before next attempt."
                )
                time.sleep(LLM_RETRY_DELAY_SECONDS)
                continue
            raise
        finally:
            rotator.release(api_key)


def _generate_json_task(
    *,
    user_query: str,
    task_label: str,
    system_instruction: str,
    response_schema: Optional[Dict[str, Any]],
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.2,
    enforce_rate_limit: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _generate_json_from_contents(
        contents=f"User Query: {user_query}",
        task_label=task_label,
        system_instruction=system_instruction,
        response_schema=response_schema,
        model_id=model_id,
        sampling_temperature=sampling_temperature,
        enforce_rate_limit=enforce_rate_limit,
    )



def _parse_query_parallel_ctx_v2(
    user_query: str,
    model_id: Optional[str] = None,
    tag_list: Optional[Tuple[str, ...]] = None,
    reference_book_context: Optional[str] = None,
    sampling_temperature: float = 0.2,
    use_schema_constraint: bool = True,
) -> QueryParseResult:
    shared_context = _build_parallel_context(
        tag_list=tag_list,
        reference_book_context=reference_book_context,
        response_contract="json",
    )
    semantic_sections_context = _build_parallel_context(
        tag_list=tag_list,
        reference_book_context=reference_book_context,
        response_contract="sections",
    )

    semantic_understanding_schema = {
        "type": "object",
        "properties": {
            "semantic_query_text": {"type": "string"},
            "intent_summary": {"type": "string"},
            "positive_concepts": {
                "type": "array",
                "items": {"type": "string"},
            },
            "negative_concepts": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "semantic_query_text",
            "intent_summary",
            "positive_concepts",
            "negative_concepts",
        ],
    }

    semantic_sections_v3_instruction = f"""
{semantic_sections_context}

You are the semantic understanding pass for retrieval text.

Return exactly four marked sections in this exact order:
[semantic_query_text]
<one compact retrieval line>
[intent_summary]
<one short sentence capturing the user's semantic intent>
[positive_concepts]
- <concept 1>
- <concept 2>
[negative_concepts]
- <concept 1>
- <concept 2>

Rules:
- Do not return JSON.
- Do not add any sections other than the four listed above.
- `semantic_query_text` must be a compact retrieval query, not a long explanation.
- Prefer 8-16 words or a few short noun/adjective phrases.
- Compress aggressively. Omit explanation, repetition, and flavor text that does not help retrieval.
- `intent_summary` must be short: one sentence, ideally under 35 Chinese characters or about 20 words.
- Preserve literal domain anchors when the user gives them, such as named subgenres, activity domains, or trope labels. Do not replace them with broader neighboring concepts.
- If the user clearly requests multiple required facets, keep all of them alive in `semantic_query_text`. Do not collapse them into a single broad theme.
- Make it tag-heavy only when the tags are explicit or strongly supported by the user query itself.
- `positive_concepts` and `negative_concepts` should be short concept phrases, not full sentences.
- `positive_concepts` should contain at most 6 items.
- `negative_concepts` should contain at most 8 items.
- Include only explicit or high-confidence retrieval anchors in `positive_concepts`.
- Put directly rejected ideas into `negative_concepts`.
- CRITICAL: `negative_concepts` must capture EVERY genre, type, tag, or element the user explicitly rejects or says they dislike. Pay close attention to these Chinese patterns:
  - 「不要」「不喜歡」「不想看」「不接受」「不要有」「不要太」「沒興趣」「對....沒興趣」 = explicit rejection
  - 「别」「不要....這種」「排除」「非....不可」 = explicit rejection
  - 「氾濫」「太過氾濫」「看膩了」「不可以」「不行」 = explicit rejection
  For example: 「對武俠/奇幻沒興趣」 → 武俠 and 奇幻 are negative_concepts
  For example: 「不要有奇幻元素」 → 奇幻 is a negative_concept
  For example: 「後宮戰鬥有點膩了」 → 後宮 and 戰鬥 are negative_concepts
- If the user says a concept is optional, acceptable-but-not-required, or "not necessary", do not place it in `positive_concepts` unless it is also clearly core to the request.
- Use related books only as soft calibration. Do not import extra traits from example titles unless the user explicitly asks for those traits.
- Do not let example works override the user's literal constraints.
- Do not broaden exact concepts into nearby but looser substitutes.
- Do not output hard constraints such as completion status, author, or word count here.
- Do not put optional concepts into `negative_concepts` unless the user explicitly rejects them.
""".strip()

    # ── Constrained Decoding: build tag enum from whitelist ──
    # When the caller provides a tag_list (the full set of legal tags from the DB),
    # we inject it as a JSON Schema `enum` constraint on the items of both
    # positive_terms and negative_terms.  The Gemini Structured Outputs backend
    # uses this to mask illegal tokens at decode time, so the model physically
    # cannot emit a tag that does not exist in the database.
    if tag_list and use_schema_constraint:
        tag_enum = list(tag_list)  # tuple → list for JSON serialisation
        tag_item_schema = {"type": "string", "enum": tag_enum}
        print(f"[Parser] Constrained decoding ACTIVE: {len(tag_enum)} legal tags in enum")
    else:
        tag_item_schema = {"type": "string"}
        print(f"[Parser] Constrained decoding INACTIVE: use_schema_constraint={use_schema_constraint}, free-form strings allowed")

    tag_projection_schema = {
        "type": "object",
        "properties": {
            "positive_terms": {
                "type": "array",
                "items": tag_item_schema,
            },
            "negative_terms": {
                "type": "array",
                "items": tag_item_schema,
            },
        },
        "required": ["positive_terms", "negative_terms"],
    }

    string_candidate_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "evidence": {"type": "string"},
            "is_explicit": {"type": "boolean"},
        },
        "required": ["value", "evidence", "is_explicit"],
    }

    number_candidate_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "evidence": {"type": "string"},
            "is_explicit": {"type": "boolean"},
        },
        "required": ["value", "evidence", "is_explicit"],
    }

    structured_schema = {
        "type": "object",
        "properties": {
            "target_status_candidate": string_candidate_schema,
            "author_name_candidate": string_candidate_schema,
            "words_min_candidate": number_candidate_schema,
            "words_max_candidate": number_candidate_schema,
        },
        "required": [
            "target_status_candidate",
            "author_name_candidate",
            "words_min_candidate",
            "words_max_candidate",
        ],
    }

    tag_projection_lite_instruction = f"""
{shared_context}

You are the tag projection pass.

    Return JSON with:
    - positive_terms
    - negative_terms

    Rules:
    - The input includes the original query and a compact semantic understanding summary.
    - Project only the strongest retrieval anchors into short tag-like terms.
    - Prefer exact tag names from AVAILABLE TAGS whenever possible.
    - Be conservative for positive_terms. Omit weak, optional, or example-derived concepts.
    - `positive_terms` should contain 3-6 high-confidence terms only.
    - `negative_terms` should contain 0-8 explicit exclusions.
    - CRITICAL for `negative_terms`: 
      1. MUST capture ALL genres or elements explicitly rejected by the user.
      2. PROACTIVE EXPANSION: When a user rejects a concept (e.g., "不要奇幻"), you MUST also add all closely related tags from the list to `negative_terms` (e.g., add "魔法", "異世界"). Do not just stop at one tag.
      3. INFER GENRE CLASHES: If the user requests pure, peaceful, or wholesome genres (like "日常", "溫馨", "治癒", "純戀愛"), you MUST proactively add highly toxic/clashing tags like "NTR", "黑暗", "獵奇", "病嬌" to `negative_terms`, even if the user didn't explicitly mention them.
    - Each rejected concept should map to the closest AVAILABLE TAG name.
    - Return only these two keys. Do not emit helper fields, explanations, or notes.
""".strip()

    structured_instruction = f"""
{shared_context}

You are the structured constraints pass.

Return JSON with exactly these four keys:
- target_status_candidate
- author_name_candidate
- words_min_candidate
- words_max_candidate

Rules:
- Only identify hard constraints that are directly stated in the query.
- Each candidate must always include `value`, `evidence`, and `is_explicit`.
- `evidence` must be a short verbatim quote copied from the user query.
- Set `is_explicit` to true only when the quoted evidence directly supports the candidate value.
- Never infer hard constraints from examples, vibes, or semantic themes.
- Normalize completion status values to `completed` or `ongoing`.
- Use numeric word counts for `words_min_candidate` and `words_max_candidate`.
- If a structured constraint is absent or uncertain, keep the key but return an empty candidate:
  - string candidate: `{{"value":"","evidence":"","is_explicit":false}}`
  - numeric candidate: `{{"value":0,"evidence":"","is_explicit":false}}`
- Outside the explicit empty-candidate format above, never use placeholder values such as `none`, `null`, `unknown`, or fake evidence like `none`.
""".strip()

    def _run_schema_task(
        task_label: str,
        instruction: str,
        schema: Dict[str, Any],
        contents: Optional[str] = None,
        normalizer: Optional[Any] = None,
        use_marked_sections: bool = False,
    ) -> Tuple[Any, Dict[str, Any]]:
        started_at = time.perf_counter()
        try:
            if use_marked_sections:
                request_contents = contents if contents is not None else f"User Query: {user_query}"
                raw_text, call_metadata = _generate_text_from_contents(
                    contents=request_contents,
                    task_label=task_label,
                    system_instruction=instruction,
                    model_id=model_id,
                    sampling_temperature=sampling_temperature,
                    enforce_rate_limit=False,
                )
                raw_result, parse_metadata = _parse_semantic_sections_v3_text(raw_text)
                call_metadata["parse_source"] = parse_metadata.get("parse_source", "marked_sections")
                call_metadata["recovered_from_raw_text"] = bool(
                    parse_metadata.get("recovered_from_raw_text", False)
                )
            else:
                if contents is None:
                    raw_result, call_metadata = _generate_json_task(
                        user_query=user_query,
                        task_label=task_label,
                        system_instruction=instruction,
                        response_schema=schema,
                        model_id=model_id,
                        sampling_temperature=sampling_temperature,
                        enforce_rate_limit=False,
                    )
                else:
                    raw_result, call_metadata = _generate_json_from_contents(
                        contents=contents,
                        task_label=task_label,
                        system_instruction=instruction,
                        response_schema=schema,
                        model_id=model_id,
                        sampling_temperature=sampling_temperature,
                        enforce_rate_limit=False,
                    )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            call_metadata = getattr(exc, "llm_call_metadata", {})
            branch_metadata = {
                "success": False,
                "latency_ms": round(elapsed_ms, 2),
                "request_count": int(call_metadata.get("request_count", 0) or 0),
                "retry_count": int(call_metadata.get("retry_count", 0) or 0),
                "first_attempt_success": bool(call_metadata.get("first_attempt_success", False)),
                "used_response_schema": bool(call_metadata.get("used_response_schema", False)),
                "parse_source": str(call_metadata.get("parse_source", "failed")),
                "recovered_from_raw_text": bool(call_metadata.get("recovered_from_raw_text", False)),
                "model_id": str(call_metadata.get("model_id") or model_id or DEFAULT_PARSER_MODEL),
                "last_retry_error": str(call_metadata.get("last_retry_error", "")),
                "error": str(exc),
            }
            raise ParserBranchError(task_label, f"{task_label} failed: {exc}", branch_metadata) from exc

        if DEBUG_LLM_OUTPUT:
            print(f"[debug:{task_label}] raw={json.dumps(raw_result, ensure_ascii=False)}")
        result = normalizer(raw_result) if normalizer else raw_result
        if DEBUG_LLM_OUTPUT:
            print(f"[debug:{task_label}] normalized={json.dumps(result, ensure_ascii=False)}")
        latency_ms = (time.perf_counter() - started_at) * 1000
        branch_metadata = {
            "success": True,
            "latency_ms": round(latency_ms, 2),
            "request_count": int(call_metadata.get("request_count", 0) or 0),
            "retry_count": int(call_metadata.get("retry_count", 0) or 0),
            "first_attempt_success": bool(call_metadata.get("first_attempt_success", False)),
            "used_response_schema": bool(call_metadata.get("used_response_schema", False)),
            "parse_source": str(call_metadata.get("parse_source", "unknown")),
            "recovered_from_raw_text": bool(call_metadata.get("recovered_from_raw_text", False)),
            "model_id": str(call_metadata.get("model_id") or model_id or DEFAULT_PARSER_MODEL),
            "last_retry_error": str(call_metadata.get("last_retry_error", "")),
        }
        return result, branch_metadata

    started_at = time.perf_counter()
    branch_metrics: Dict[str, Dict[str, Any]] = {}

    def _attach_parser_metadata(exc: ParserBranchError) -> ParserBranchError:
        branch_metrics[exc.task_label] = exc.branch_metadata
        exc.parser_metadata = {
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "branches": branch_metrics,
            "parser_mode": _parser_mode_for_variant(),
            "parser_variant": DEFAULT_PARSER_VARIANT,
            "task_split": "semantic_understanding_tag_projection_structured",
            "failed_branch": exc.task_label,
            "total_request_count": sum(
                int(branch.get("request_count", 0) or 0)
                for branch in branch_metrics.values()
            ),
            "total_retry_count": sum(
                int(branch.get("retry_count", 0) or 0)
                for branch in branch_metrics.values()
            ),
        }
        return exc

    try:
        semantic_understanding, branch_metadata = _run_schema_task(
            "semantic_understanding",
            semantic_sections_v3_instruction,
            semantic_understanding_schema,
            normalizer=lambda raw: _normalize_semantic_understanding(raw, user_query),
            use_marked_sections=True,
        )
        branch_metrics["semantic_understanding"] = branch_metadata
    except ParserBranchError as exc:
        raise _attach_parser_metadata(exc)

    tag_projection_context = _build_tag_projection_compact_context(semantic_understanding)
    tag_projection_label = "Compact Semantic Understanding"
    tag_projection_contents = (
        f"Original Query:\n{user_query}\n\n"
        f"{tag_projection_label}:\n{tag_projection_context}"
    )
    try:
        tag_projection, branch_metadata = _run_schema_task(
            "tag_projection",
            tag_projection_lite_instruction,
            tag_projection_schema,
            contents=tag_projection_contents,
            normalizer=_normalize_tag_projection,
        )
        branch_metrics["tag_projection"] = branch_metadata
    except ParserBranchError as exc:
        raise _attach_parser_metadata(exc)

    tag_intent = _build_tag_intent_from_projection(
        user_query=user_query,
        semantic_understanding=semantic_understanding,
        tag_projection=tag_projection,
    )
    structured_context = _build_structured_context_from_semantic_understanding(
        semantic_understanding,
        tag_projection=tag_projection,
    )
    structured_contents = (
        f"Original Query:\n{user_query}\n\n"
        f"{structured_context}"
    )
    try:
        structured_candidates, branch_metadata = _run_schema_task(
            "structured",
            structured_instruction,
            structured_schema,
            contents=structured_contents,
            normalizer=_normalize_structured_draft,
        )
        branch_metrics["structured"] = branch_metadata
    except ParserBranchError as exc:
        raise _attach_parser_metadata(exc)

    structured_slots = _apply_structured_draft_guards(
        user_query,
        _structured_candidates_to_slots(structured_candidates),
        structured_candidates,
    )
    structured_slots = _merge_structured_slots(structured_slots)

    total_latency_ms = (time.perf_counter() - started_at) * 1000
    if DEBUG_LLM_OUTPUT:
        print(f"[debug:structured] merged_slots={json.dumps(structured_slots, ensure_ascii=False)}")

    merged_result = _merge_query_parse_results(
        user_query=user_query,
        tag_intent=tag_intent,
        structured_slots=structured_slots,
        total_latency_ms=total_latency_ms,
        branch_metrics=branch_metrics,
    )
    return merged_result


@functools.lru_cache(maxsize=1000)
def parse_query(
    user_query: str,
    model_id: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    tag_list: Optional[Tuple[str, ...]] = None,
    reference_book_context: Optional[str] = None,
    sampling_temperature: float = 0.2,
    use_schema_constraint: bool = True,
) -> QueryParseResult:
    _ = cache_namespace
    return _parse_query_parallel_ctx_v2(
        user_query,
        model_id=model_id,
        tag_list=tag_list,
        reference_book_context=reference_book_context,
        sampling_temperature=sampling_temperature,
        use_schema_constraint=use_schema_constraint,
    )


# ═══════════════════════════════════════════════════════════════
# LLM-as-Router: Let the LLM decide the fusion strategy
# ═══════════════════════════════════════════════════════════════

_LLM_ROUTER_SYSTEM_PROMPT = """\
You are a query intent classifier for a novel recommendation system.
Your task is to analyze the user's book recommendation request and choose the most suitable retrieval strategy.

You have two retrieval tools to choose from:

## Tool A: Weighted (Strict Tag-Based Retriever)
Best for queries with **explicit, concrete hard constraints**, such as:
  - Specifying completion status (e.g. "must be completed", "要完結的")
  - Specifying an author name (e.g. "books by author XX")
  - Specifying word count range (e.g. "at least 1 million words")
  - Having many specific genre/tag requirements (e.g. "isekai + harem + battle")
  - Having explicit exclusion conditions (e.g. "no NTR", "不要異世界")
Strength: Precise matching of hard filters. Best for "find books matching these exact criteria" queries.

## Tool B: RRF (Fuzzy Semantic Retriever)
Best for queries describing **atmosphere, feelings, or vague preferences**, such as:
  - "I want something that gets my blood pumping"
  - "Something like Mushoku Tensei with a vast world"
  - "Recommend novels that will make me cry"
  - Long descriptions of personal reading tastes and feelings
  - Using a specific work as a reference to find "similar style" books
Strength: Strong semantic understanding. Best for "recommend books with this kind of vibe" queries.

## Decision Rules
1. If the query contains **any hard filtering conditions** (completion status, author, word count limits), choose weighted.
2. If the query primarily describes **atmosphere, style, or feelings** with no hard filters, choose rrf.
3. If the query mixes both, determine which aspect dominates. If hard constraints are the core need, choose weighted; if they are just mentioned in passing, choose rrf.
4. Use confidence to express your certainty (0.0~1.0). If unsure, set to 0.5.
"""

_LLM_ROUTER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "strategy": {
            "type": "STRING",
            "description": "The chosen retrieval strategy. Must be 'weighted' or 'rrf'.",
            "enum": ["weighted", "rrf"],
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence level of the routing decision, from 0.0 to 1.0.",
        },
        "reasoning": {
            "type": "STRING",
            "description": "Brief reasoning for the decision.",
        },
    },
    "required": ["strategy", "confidence", "reasoning"],
}


@functools.lru_cache(maxsize=500)
def route_query_with_llm(
    user_query: str,
    model_id: Optional[str] = None,
    cache_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """Use the LLM to classify the query intent and decide the fusion strategy.

    Returns a dict with keys:
        strategy: 'weighted' or 'rrf'
        confidence: float 0.0-1.0
        reasoning: str
        metadata: dict with LLM call metrics
    """
    _ = cache_namespace  # included for lru_cache key diversity

    contents = (
        f"## User's Book Recommendation Request\n\n"
        f"{user_query}\n\n"
        f"## Your Task\n\n"
        f"Analyze the request above and decide whether to use "
        f"'weighted' (strict tag retriever) or 'rrf' (fuzzy semantic retriever)."
    )

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            parsed, metadata = _generate_json_from_contents(
                contents=contents,
                task_label=f"routing_attempt_{attempt}",
                system_instruction=_LLM_ROUTER_SYSTEM_PROMPT,
                response_schema=_LLM_ROUTER_RESPONSE_SCHEMA,
                model_id=model_id,
                sampling_temperature=0.1,
                enforce_rate_limit=True,
            )

            strategy = str(parsed.get("strategy", "weighted")).strip().lower()
            if strategy not in ("weighted", "rrf"):
                print(f"[Router:LLM] invalid strategy '{strategy}', falling back to weighted")
                strategy = "weighted"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("reasoning", "")).strip()

            print(
                f"[Router:LLM] decision={strategy} confidence={confidence:.2f} "
                f"reason={reasoning[:80]}"
            )

            return {
                "strategy": strategy,
                "confidence": confidence,
                "reasoning": reasoning,
                "metadata": metadata,
                "fallback": False,
            }
        except Exception as exc:
            last_error = exc
            print(f"[Router:LLM] LLM routing attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                time.sleep(1.5)

    print(f"[Router:LLM] LLM routing failed after {max_retries} attempts. Falling back to weighted.")
    return {
        "strategy": "weighted",
        "confidence": 0.0,
        "reasoning": f"LLM routing failed after {max_retries} attempts: {last_error}",
        "metadata": {"error": str(last_error)},
        "fallback": True,
    }
