import json
import functools
import time
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from google import genai
from google.genai import types
from pydantic import BaseModel as PydanticBaseModel
from src.models.schemas import QueryParseResult, ScoringCriteria, ScoringParameters, TagIntent

# ??????謘?(???????制??謅???????剜???蹇?????
DEFAULT_PARSER_MODEL = "gemma-3-27b-it"
LLM_REQUEST_TIMEOUT_SECONDS = 45.0

DEBUG_LLM_OUTPUT = True
_T = TypeVar("_T")


def _call_with_timeout(func: Callable[[], _T], timeout_seconds: float, label: str) -> _T:
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
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _coerce_response_to_json_object(response: Any, task_label: str) -> Dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, PydanticBaseModel):
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        return parsed.dict()
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None and hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if parsed is not None and hasattr(parsed, "dict"):
        return parsed.dict()

    raw_text = _extract_text_from_response(response)
    reparsed = _parse_json_object_from_text(raw_text)
    if reparsed is not None:
        print(f"[llm:{task_label}] recovered JSON from raw text because parsed payload was missing.")
        return reparsed

    raw_preview = raw_text[:200].replace("\n", "\\n") if raw_text else "(empty)"
    raise ValueError(
        f"Structured output missing parsed payload for {task_label}; raw_text={raw_preview}"
    )
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
    hypothetical_intro: str,
    structured_slots: Dict[str, Any],
    total_latency_ms: float,
    branch_latencies_ms: Dict[str, float],
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
        "branches": {name: round(latency, 2) for name, latency in branch_latencies_ms.items()},
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
        hypothetical_intro=str(hypothetical_intro or "").strip(),
        criteria=merged_criteria,
        parse_metadata=metadata,
    )


def _build_parallel_context(
    tag_list: Optional[Tuple[str, ...]] = None,
    reference_book_context: Optional[str] = None,
) -> str:
    sections = [
        "You are one branch of a web novel query parser.",
        "Return strict JSON only. Use snake_case keys. Do not wrap the JSON in markdown.",
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
    response_schema: Dict[str, Any],
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.2,
    enforce_rate_limit: bool = True,
) -> Dict[str, Any]:
    from src.core.api_utils import (
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
        api_key = get_current_api_key()
        key_index = rotator.current_index + 1
        client = genai.Client(api_key=api_key)

        try:
            if enforce_rate_limit:
                print(f"[llm:{task_label}] waiting for rate limiter before request.")
                get_rate_limiter().wait(api_key)

            config_args = {
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "system_instruction": system_instruction,
                "temperature": sampling_temperature,
                "top_p": 0.95,
            }

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
            return _coerce_response_to_json_object(response, task_label)
        except Exception as exc:
            last_exception = exc
            print(f"[llm:{task_label}] request failed on key {key_index}/{total_api_keys}: {exc}")
            continue
        finally:
            if total_api_keys > 1:
                rotator.rotate()


def _generate_json_task(
    *,
    user_query: str,
    task_label: str,
    system_instruction: str,
    response_schema: Dict[str, Any],
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.2,
    enforce_rate_limit: bool = True,
) -> Dict[str, Any]:
    return _generate_json_from_contents(
        contents=f"User Query: {user_query}",
        task_label=task_label,
        system_instruction=system_instruction,
        response_schema=response_schema,
        model_id=model_id,
        sampling_temperature=sampling_temperature,
        enforce_rate_limit=enforce_rate_limit,
    )


def _generate_text_from_contents(
    *,
    contents: str,
    task_label: str,
    system_instruction: str,
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.4,
    enforce_rate_limit: bool = True,
) -> str:
    from src.core.api_utils import (
        get_api_key_rotator,
        get_current_api_key,
        get_rate_limiter,
    )

    selected_model = str(model_id or DEFAULT_PARSER_MODEL).strip() or DEFAULT_PARSER_MODEL

    print(f"[llm:{task_label}] trying model: {selected_model}")

    rotator = get_api_key_rotator()
    if not getattr(rotator, "api_keys", None):
        raise RuntimeError(f"[llm:{task_label}] no API keys configured")

    request_count = 0
    total_api_keys = len(rotator.api_keys)

    while True:
        api_key = get_current_api_key()
        key_index = rotator.current_index + 1
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

            raw_text = _extract_text_from_response(response).strip()
            if raw_text:
                return raw_text
            raise ValueError(f"{task_label} returned empty text")
        except Exception as exc:
            print(f"[llm:{task_label}] request failed on key {key_index}/{total_api_keys}: {exc}")
            continue
        finally:
            if total_api_keys > 1:
                rotator.rotate()


def _generate_hypothetical_intro(
    *,
    user_query: str,
    semantic_understanding: Dict[str, Any],
    tag_projection: Dict[str, Any],
    model_id: Optional[str] = None,
    sampling_temperature: float = 0.4,
) -> str:
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
    projected_tags = [
        str(term).strip()
        for term in tag_projection.get("positive_terms", [])
        if str(term).strip()
    ]

    system_instruction = """
You write HyDE passages for semantic retrieval against novel intros.

Return only the hypothetical intro text. Do not return JSON, markdown, lists, labels, or explanations.

Rules:
- Write a realistic novel intro or jacket summary in Traditional Chinese unless the query clearly uses another language.
- Keep it to 3-5 sentences.
- Focus on premise, conflict, tone, and the kind of emotional or thematic experience the reader wants.
- Make it resemble the wording found in real novel intros.
- Prefer positive description of desired qualities instead of listing exclusions.
- Avoid repetition, slogan-like phrasing, and bullet-list cadence.
""".strip()

    contents = (
        f"Original Query:\n{user_query}\n\n"
        f"Semantic Query Text:\n{semantic_query_text or '(empty)'}\n\n"
        f"Intent Summary:\n{intent_summary or '(empty)'}\n\n"
        f"Positive Concepts:\n{', '.join(positive_concepts) if positive_concepts else '(empty)'}\n\n"
        f"Negative Concepts:\n{', '.join(negative_concepts) if negative_concepts else '(empty)'}\n\n"
        f"Projected Tags:\n{', '.join(projected_tags) if projected_tags else '(empty)'}\n\n"
        "Write one hypothetical intro that would be highly similar to the target novels' real intros."
    )

    return _generate_text_from_contents(
        contents=contents,
        task_label="hyde_intro",
        system_instruction=system_instruction,
        model_id=model_id,
        sampling_temperature=sampling_temperature,
        enforce_rate_limit=False,
    ).strip()



def _parse_query_parallel_ctx_v2(
    user_query: str,
    model_id: Optional[str] = None,
    tag_list: Optional[Tuple[str, ...]] = None,
    reference_book_context: Optional[str] = None,
    sampling_temperature: float = 0.2,
) -> QueryParseResult:
    shared_context = _build_parallel_context(
        tag_list=tag_list,
        reference_book_context=reference_book_context,
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
            "ambiguities": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "semantic_query_text",
            "intent_summary",
            "positive_concepts",
            "negative_concepts",
            "ambiguities",
        ],
    }

    tag_projection_schema = {
        "type": "object",
        "properties": {
            "positive_terms": {
                "type": "array",
                "items": {"type": "string"},
            },
            "negative_terms": {
                "type": "array",
                "items": {"type": "string"},
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
        "required": [],
    }

    semantic_understanding_instruction = f"""
{shared_context}

You are the semantic understanding pass for retrieval text.

Return JSON with:
- semantic_query_text
- intent_summary
- positive_concepts
- negative_concepts
- ambiguities

    Rules:
    - `semantic_query_text` is the short retrieval text that will be embedded and compared against novel intros.
    - Write `semantic_query_text` as a compact retrieval query, not a long explanation.
    - Prefer 8-20 words or a few short noun/adjective phrases.
    - Make it tag-heavy when the query clearly implies genres, tropes, vibes, or themes.
    - Keep richer reasoning, nuance, and examples inside `intent_summary`, not `semantic_query_text`.
    - `positive_concepts` and `negative_concepts` should be short concept phrases, not full sentences.
    - Put directly rejected ideas into `negative_concepts`.
    - Use `ambiguities` for things that are unclear or could be interpreted in more than one way.
    - Do not output hard constraints such as completion status, author, or word count here.
""".strip()

    tag_projection_instruction = f"""
{shared_context}

You are the tag projection pass.

    Return JSON with:
    - positive_terms
    - negative_terms

    Rules:
    - The input includes the original query and the semantic understanding JSON.
    - Project semantic concepts into short tag-like terms.
    - Prefer exact tag names from AVAILABLE TAGS whenever possible.
    - `positive_terms` should contain enough tag-like concepts to drive retrieval and tag mapping.
    - Return 5-12 `positive_terms` when the query supports that many distinct concepts.
    - `negative_terms` should contain only concepts the user explicitly rejects or excludes.
    - Return 3-8 `negative_terms` when the query contains multiple explicit dislikes.
    - If a concept should stay only in semantic retrieval text and not be projected into tags, omit it.
    - Return only these two keys. Do not emit helper fields such as `rejected_terms`, `mapping_notes`, `intent_summary`, or explanations.
""".strip()

    structured_instruction = f"""
{shared_context}

You are the structured constraints pass.

Return JSON with any of:
- target_status_candidate
- author_name_candidate
- words_min_candidate
- words_max_candidate

Rules:
- Only identify hard constraints that are directly stated in the query.
- Each candidate must include `value`, `evidence`, and `is_explicit`.
- `evidence` must be a short verbatim quote copied from the user query.
- Set `is_explicit` to true only when the quoted evidence directly supports the candidate value.
- Never infer hard constraints from examples, vibes, or semantic themes.
- Normalize completion status values to `completed` or `ongoing`.
- Use numeric word counts for `words_min_candidate` and `words_max_candidate`.
- If a structured constraint is absent or uncertain, omit that candidate entirely.
- Never use placeholder values such as `none`, `null`, `unknown`, `0`, or fake evidence like `none`.
""".strip()

    def _run_schema_task(
        task_label: str,
        instruction: str,
        schema: Dict[str, Any],
        contents: Optional[str] = None,
        normalizer: Optional[Any] = None,
    ) -> Tuple[Any, float]:
        started_at = time.perf_counter()
        if contents is None:
            raw_result = _generate_json_task(
                user_query=user_query,
                task_label=task_label,
                system_instruction=instruction,
                response_schema=schema,
                model_id=model_id,
                sampling_temperature=sampling_temperature,
                enforce_rate_limit=False,
            )
        else:
            raw_result = _generate_json_from_contents(
                contents=contents,
                task_label=task_label,
                system_instruction=instruction,
                response_schema=schema,
                model_id=model_id,
                sampling_temperature=sampling_temperature,
                enforce_rate_limit=False,
            )
        if DEBUG_LLM_OUTPUT:
            print(f"[debug:{task_label}] raw={json.dumps(raw_result, ensure_ascii=False)}")
        result = normalizer(raw_result) if normalizer else raw_result
        if DEBUG_LLM_OUTPUT:
            print(f"[debug:{task_label}] normalized={json.dumps(result, ensure_ascii=False)}")
        return result, (time.perf_counter() - started_at) * 1000

    started_at = time.perf_counter()
    branch_latencies_ms: Dict[str, float] = {}

    semantic_understanding, latency_ms = _run_schema_task(
        "semantic_understanding",
        semantic_understanding_instruction,
        semantic_understanding_schema,
        normalizer=lambda raw: _normalize_semantic_understanding(raw, user_query),
    )
    branch_latencies_ms["semantic_understanding"] = latency_ms

    tag_projection_contents = (
        f"Original Query:\n{user_query}\n\n"
        f"Semantic Understanding JSON:\n{json.dumps(semantic_understanding, ensure_ascii=False)}"
    )
    tag_projection, latency_ms = _run_schema_task(
        "tag_projection",
        tag_projection_instruction,
        tag_projection_schema,
        contents=tag_projection_contents,
        normalizer=_normalize_tag_projection,
    )
    branch_latencies_ms["tag_projection"] = latency_ms

    tag_intent = _build_tag_intent_from_projection(
        user_query=user_query,
        semantic_understanding=semantic_understanding,
        tag_projection=tag_projection,
    )
    started_hyde_at = time.perf_counter()
    hypothetical_intro = _generate_hypothetical_intro(
        user_query=user_query,
        semantic_understanding=semantic_understanding,
        tag_projection=tag_projection,
        model_id=model_id,
        sampling_temperature=sampling_temperature,
    )
    branch_latencies_ms["hyde_intro"] = (time.perf_counter() - started_hyde_at) * 1000

    structured_context = _build_structured_context_from_semantic_understanding(
        semantic_understanding,
        tag_projection=tag_projection,
    )
    structured_contents = (
        f"Original Query:\n{user_query}\n\n"
        f"{structured_context}"
    )
    structured_candidates, latency_ms = _run_schema_task(
        "structured",
        structured_instruction,
        structured_schema,
        contents=structured_contents,
        normalizer=_normalize_structured_draft,
    )
    branch_latencies_ms["structured"] = latency_ms

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
        hypothetical_intro=hypothetical_intro,
        structured_slots=structured_slots,
        total_latency_ms=total_latency_ms,
        branch_latencies_ms=branch_latencies_ms,
    )
    return merged_result.model_copy(
        update={
            "parse_metadata": {
                **merged_result.parse_metadata,
                "parser_mode": "response_schema_three_call_plus_hyde",
                "task_split": "semantic_understanding_tag_projection_hyde_structured",
                "hyde_enabled": True,
            }
        }
    )


@functools.lru_cache(maxsize=1000)
def parse_query(
    user_query: str,
    model_id: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    tag_list: Optional[Tuple[str, ...]] = None,
    reference_book_context: Optional[str] = None,
    sampling_temperature: float = 0.2,
) -> QueryParseResult:
    _ = cache_namespace
    return _parse_query_parallel_ctx_v2(
        user_query,
        model_id=model_id,
        tag_list=tag_list,
        reference_book_context=reference_book_context,
        sampling_temperature=sampling_temperature,
    )


