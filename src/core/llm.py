import os
import json
import functools
import time
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from src.models.schemas import QueryParseResult, ScoringCriteria, ScoringParameters
from src.core.api_utils import retry_on_rate_limit, _is_retryable

# ?舐璅∪?皜 (靘??摨????嗅?璅∪?憭望??????
FALLBACK_MODELS = ["gemma-3-27b-it", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]

DEBUG_LLM_OUTPUT = True

def _normalize_llm_output(parsed: Any, user_query: str) -> Dict[str, Any]:
    """
    Normalizes LLM output (dict or list) into a structure matching QueryParseResult.
    Handles snake_case conversion, missing fields, and flat lists.
    """
    import re
    
    def to_snake_case(name: str) -> str:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def recursive_normalize_keys(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                new_k = to_snake_case(k)
                new_obj[new_k] = recursive_normalize_keys(v)
            return new_obj
        elif isinstance(obj, list):
            return [recursive_normalize_keys(item) for item in obj]
        else:
            return obj

    def to_string(value: Any, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                text = to_string(item, "")
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        if isinstance(value, dict):
            for key in ("text", "value", "content", "query_text", "description", "title", "name"):
                if key in value:
                    text = to_string(value.get(key), "")
                    if text:
                        return text
            return default
        return str(value).strip()

    def to_string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if "," in text:
                candidates = [segment.strip() for segment in text.split(",")]
            elif "\n" in text:
                candidates = [segment.strip("-* \t") for segment in text.splitlines()]
            else:
                candidates = [text]
            return [candidate for candidate in candidates if candidate]
        if isinstance(value, list):
            flattened: List[str] = []
            for item in value:
                flattened.extend(to_string_list(item))
            return flattened
        if isinstance(value, dict):
            return [text for text in (to_string(value, ""),) if text]
        text = str(value).strip()
        return [text] if text else []

    def dedupe_strings(values: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for value in values:
            normalized = value.replace(" ", "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(value.strip())
        return deduped

    # 1. Normalize keys to snake_case
    parsed = recursive_normalize_keys(parsed)

    final_result = {
        "original_query": user_query,
        "search_terms": user_query, # Default
        "generated_keywords": [],
        "hypothetical_intro": "",
        "criteria": [],
        "parse_metadata": {},
    }

    # 2. Handle Dict vs List
    if isinstance(parsed, dict):
        # Merge into defaults
        if "original_query" in parsed:
            final_result["original_query"] = to_string(parsed["original_query"], user_query) or user_query
        if "search_terms" in parsed:
            final_result["search_terms"] = to_string(parsed["search_terms"], user_query) or user_query
        if "generated_keywords" in parsed:
            final_result["generated_keywords"] = dedupe_strings(to_string_list(parsed["generated_keywords"]))
        if "hypothetical_intro" in parsed:
            final_result["hypothetical_intro"] = to_string(parsed["hypothetical_intro"], "")
        if "parse_metadata" in parsed and isinstance(parsed["parse_metadata"], dict):
            final_result["parse_metadata"] = parsed["parse_metadata"]
        
        # Handle criteria/scoring_criteria alias
        if "criteria" in parsed:
            final_result["criteria"] = parsed["criteria"]
        elif "scoring_criteria" in parsed:
            final_result["criteria"] = parsed["scoring_criteria"]
            
    elif isinstance(parsed, list):
        # Assume list of criteria
        final_result["criteria"] = parsed

    if not final_result["search_terms"]:
        final_result["search_terms"] = user_query

    if not final_result["generated_keywords"] and final_result["search_terms"]:
        generated_from_search_terms = dedupe_strings(to_string_list(final_result["search_terms"]))
        if len(generated_from_search_terms) > 1:
            final_result["generated_keywords"] = generated_from_search_terms[:10]

    if not final_result["hypothetical_intro"] and isinstance(parsed, dict):
        for fallback_key in ("intro", "summary", "blurb", "description"):
            if fallback_key in parsed:
                final_result["hypothetical_intro"] = to_string(parsed[fallback_key], "")
                if final_result["hypothetical_intro"]:
                    break

    # 3. Validate/Fix Criteria List
    valid_criteria = []
    criteria_source = final_result["criteria"]
    if isinstance(criteria_source, dict):
        criteria_source = [criteria_source]
    elif isinstance(criteria_source, str):
        criteria_source = []

    if isinstance(criteria_source, list):
        for item in criteria_source:
            if not isinstance(item, dict): continue
            
            # Fix Name/Function/Type aliases
            if "function" in item and "name" not in item:
                item["name"] = item.pop("function")
            if "type" in item and "name" not in item:
                item["name"] = item.pop("type")
            if "name" not in item: continue # Skip if no name

            # Fix Parameters
            if "parameters" not in item or not isinstance(item.get("parameters"), dict):
                item["parameters"] = {}
            
            # Move top-level params to parameters dict
            known_params = [
                "field", "keyword", "min_val", "max_val", "target_status", 
                "query_text", "author_name", "ranking_direction", "normalize_max"
            ]
            for param in known_params:
                if param in item:
                    if param not in item["parameters"]:
                        item["parameters"][param] = item.pop(param)
                    else:
                        item.pop(param) # Duplicate
            
            # Remove non-schema keys Gemma might add (e.g. "weight", "type")
            for extra_key in ["weight", "type", "function"]:
                item.pop(extra_key, None)
            
            # Fix Is Negative
            if "is_negative" not in item:
                item["is_negative"] = False
            else:
                item["is_negative"] = bool(item["is_negative"])

            valid_criteria.append(item)
    
    final_result["criteria"] = valid_criteria
    final_result["generated_keywords"] = dedupe_strings(
        [keyword for keyword in final_result["generated_keywords"] if keyword]
    )
    return final_result


def _normalize_structured_slots(parsed: Any) -> Dict[str, Any]:
    import re
    null_like_strings = {"", "null", "none", "nil", "n/a", "na", "未指定", "無"}

    def to_snake_case(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def recursive_normalize_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {to_snake_case(k): recursive_normalize_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [recursive_normalize_keys(item) for item in obj]
        return obj

    def to_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in null_like_strings:
                return None
            return text or None
        if isinstance(value, list):
            parts = [to_string(item) for item in value]
            parts = [part for part in parts if part]
            return " ".join(parts).strip() or None
        if isinstance(value, dict):
            for key in ("value", "text", "content", "name"):
                if key in value:
                    return to_string(value.get(key))
            return None
        return str(value).strip() or None

    def to_number(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "")
        if text.lower() in null_like_strings:
            return None
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    normalized = recursive_normalize_keys(parsed if isinstance(parsed, dict) else {})

    status = to_string(normalized.get("target_status"))
    author = to_string(normalized.get("author_name"))
    words_min = to_number(normalized.get("words_min"))
    words_max = to_number(normalized.get("words_max"))

    if words_min is None:
        words_min = to_number(normalized.get("min_val"))
    if words_max is None:
        words_max = to_number(normalized.get("max_val"))

    # Treat all-zero placeholders as "unspecified" rather than a real <= 0 range.
    if words_min == 0 and words_max == 0:
        words_min = None
        words_max = None

    field = to_string(normalized.get("field"))
    if field and field != "words_total":
        field = None

    result: Dict[str, Any] = {}
    if status is not None:
        result["target_status"] = status
    if author is not None:
        result["author_name"] = author
    if words_min is not None:
        result["words_min"] = words_min
    if words_max is not None:
        result["words_max"] = words_max
    if (words_min is not None or words_max is not None) and (field or "words_total") == "words_total":
        result["field"] = "words_total"

    return result


def _normalize_semantic_slots(parsed: Any, user_query: str) -> Dict[str, Any]:
    normalized = _normalize_llm_output(parsed, user_query)

    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            values: List[str] = []
            for item in value:
                values.extend(_to_list(item))
            return values
        if isinstance(value, dict):
            values: List[str] = []
            for candidate_key in ("text", "value", "name", "query_text"):
                if candidate_key in value:
                    values.extend(_to_list(value[candidate_key]))
            return values
        text = str(value).strip()
        return [text] if text else []

    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            normalized_value = value.replace(" ", "").strip()
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            deduped.append(value.strip())
        return deduped

    raw_excluded: Any = None
    raw_negative_criteria: List[str] = []
    if isinstance(parsed, dict):
        raw_excluded = (
            parsed.get("excluded_keywords")
            or parsed.get("negative_keywords")
            or parsed.get("excluded_tags")
            or parsed.get("negative_tags")
        )
        criteria_source = parsed.get("criteria") or parsed.get("scoring_criteria") or []
        if isinstance(criteria_source, dict):
            criteria_source = [criteria_source]
        if isinstance(criteria_source, list):
            for item in criteria_source:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name", "")).strip() != "semantic_similarity":
                    continue
                if not bool(item.get("is_negative", False)):
                    continue
                parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
                query_text = parameters.get("query_text") or item.get("query_text")
                raw_negative_criteria.extend(_to_list(query_text))

    return {
        "search_terms": normalized.get("search_terms") or user_query,
        "generated_keywords": _dedupe(_to_list(normalized.get("generated_keywords"))),
        "excluded_keywords": _dedupe(_to_list(raw_excluded) + raw_negative_criteria),
    }


def _extract_structured_slots_from_query(user_query: str) -> Dict[str, Any]:
    import re

    query = str(user_query or "").strip()
    lowered = query.lower()

    status = None
    completed_markers = ["完結", "已完結", "完本", "已完本", "completed", "finished"]
    ongoing_markers = ["連載", "連載中", "更新中", "ongoing", "serializing"]

    if any(marker in query for marker in completed_markers):
        status = "completed"
    elif any(marker in query for marker in ongoing_markers) or any(marker in lowered for marker in ongoing_markers):
        status = "ongoing"

    author_name = None
    author_patterns = [
        r"(?:作者|author)\s*[:：]?\s*([^\s，,。！？!?]+)",
        r"([^\s，,。！？!?]+)\s*寫的",
        r"([^\s，,。！？!?]+)\s*的書",
    ]
    for pattern in author_patterns:
        match = re.search(pattern, query)
        if match:
            author_name = match.group(1).strip()
            if author_name:
                break

    def parse_count_token(token: str) -> Optional[float]:
        token = str(token or "").strip().replace(",", "")
        if not token:
            return None
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(萬|万|w|W|k|K)?", token)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2)
        if unit in ("萬", "万", "w", "W"):
            value *= 10000
        elif unit in ("k", "K"):
            value *= 1000
        return value

    words_min = None
    words_max = None

    range_patterns = [
        r"(\d+(?:\.\d+)?\s*(?:萬|万|w|W|k|K)?)\s*(?:字|萬字|万字)?\s*(?:到|至|~|-|－|—)\s*(\d+(?:\.\d+)?\s*(?:萬|万|w|W|k|K)?)\s*(?:字|萬字|万字)?",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, query)
        if match:
            words_min = parse_count_token(match.group(1))
            words_max = parse_count_token(match.group(2))
            break

    if words_min is None and words_max is None:
        lower_match = re.search(
            r"(?:至少|最少|不少於|不低於|>=|大於等於)?\s*(\d+(?:\.\d+)?\s*(?:萬|万|w|W|k|K)?)\s*(?:字|萬字|万字)?\s*(?:以上|起|up)?",
            query,
        )
        if lower_match:
            words_min = parse_count_token(lower_match.group(1))

    if words_min is None and words_max is None:
        upper_match = re.search(
            r"(?:最多|至多|不超過|<=|小於等於)?\s*(\d+(?:\.\d+)?\s*(?:萬|万|w|W|k|K)?)\s*(?:字|萬字|万字)?\s*(?:以下|內|down)?",
            query,
        )
        if upper_match:
            words_max = parse_count_token(upper_match.group(1))

    result: Dict[str, Any] = {}
    if status is not None:
        result["target_status"] = status
    if author_name is not None:
        result["author_name"] = author_name
    if words_min is not None:
        result["words_min"] = words_min
    if words_max is not None:
        result["words_max"] = words_max
    if words_min is not None or words_max is not None:
        result["field"] = "words_total"

    return result


def _merge_structured_slots(llm_slots: Dict[str, Any], rule_slots: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    target_status = llm_slots.get("target_status") or rule_slots.get("target_status")
    author_name = llm_slots.get("author_name") or rule_slots.get("author_name")
    words_min = llm_slots.get("words_min")
    words_max = llm_slots.get("words_max")

    if words_min is None:
        words_min = rule_slots.get("words_min")
    if words_max is None:
        words_max = rule_slots.get("words_max")

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


def _validate_structured_slots_against_query(
    user_query: str,
    structured_slots: Dict[str, Any],
) -> Dict[str, Any]:
    import re

    query = str(user_query or "")
    lowered = query.lower()
    validated = dict(structured_slots)

    status_value = str(validated.get("target_status") or "").strip().lower()
    if status_value == "completed":
        completed_markers = ["completed", "complete", "finished", "完結", "完结", "完本", "已完結", "已完结"]
        if not any(marker.lower() in lowered or marker in query for marker in completed_markers):
            validated.pop("target_status", None)
    elif status_value == "ongoing":
        ongoing_markers = ["ongoing", "serializing", "serialized", "updating", "連載", "连载", "更新中", "未完"]
        if not any(marker.lower() in lowered or marker in query for marker in ongoing_markers):
            validated.pop("target_status", None)
    elif "target_status" in validated:
        validated.pop("target_status", None)

    author_name = str(validated.get("author_name") or "").strip()
    if author_name:
        escaped_author = re.escape(author_name)
        explicit_author_patterns = [
            rf"作者\s*[:：]\s*{escaped_author}",
            rf"作者\s+{escaped_author}",
            rf"{escaped_author}\s*(?:作者|作家|著)",
            rf"author\s*[:：]?\s*{escaped_author}",
        ]
        looks_like_descriptive_phrase = any(token in author_name for token in ["小說", "文字", "情節", "感情", "風格", "作品"])
        if (
            len(author_name) > 20
            or looks_like_descriptive_phrase
            or not any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in explicit_author_patterns)
        ):
            validated.pop("author_name", None)

    if "words_min" in validated or "words_max" in validated:
        has_word_count_cue = bool(
            re.search(r"\d+(?:\.\d+)?\s*(?:萬字|万字|字|w|W|k|K|words?)", query)
        )
        if not has_word_count_cue:
            validated.pop("words_min", None)
            validated.pop("words_max", None)
            validated.pop("field", None)

    if "words_min" not in validated and "words_max" not in validated:
        validated.pop("field", None)

    return validated


def _query_contains_evidence(user_query: str, evidence: Any) -> bool:
    query = str(user_query or "").strip()
    snippet = str(evidence or "").strip()
    if not query or not snippet:
        return False
    return snippet in query


def _normalize_semantic_draft(parsed: Any) -> Dict[str, Any]:
    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            normalized_value = value.replace(" ", "").strip()
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            deduped.append(value.strip())
        return deduped

    normalized = parsed if isinstance(parsed, dict) else {}
    search_terms_candidate = str(normalized.get("search_terms_candidate") or "").strip()
    keyword_candidates = normalized.get("keyword_candidates") if isinstance(normalized.get("keyword_candidates"), list) else []
    excluded_candidates = normalized.get("excluded_candidates") if isinstance(normalized.get("excluded_candidates"), list) else []

    normalized_excluded = []
    for item in excluded_candidates:
        if not isinstance(item, dict):
            continue
        keyword = str(item.get("keyword") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        is_explicit = bool(item.get("is_explicit"))
        if keyword:
            normalized_excluded.append(
                {
                    "keyword": keyword,
                    "evidence": evidence,
                    "is_explicit": is_explicit,
                }
            )

    return {
        "search_terms_candidate": search_terms_candidate,
        "keyword_candidates": _dedupe([str(item).strip() for item in keyword_candidates if str(item).strip()]),
        "excluded_candidates": normalized_excluded,
    }


def _apply_semantic_draft_guards(
    user_query: str,
    semantic_slots: Dict[str, Any],
    semantic_draft: Dict[str, Any],
) -> Dict[str, Any]:
    guarded = dict(semantic_slots)

    explicit_exclusions = {
        str(item.get("keyword")).replace(" ", "").strip()
        for item in semantic_draft.get("excluded_candidates", [])
        if item.get("is_explicit") and _query_contains_evidence(user_query, item.get("evidence"))
    }
    if explicit_exclusions:
        guarded["excluded_keywords"] = [
            keyword
            for keyword in guarded.get("excluded_keywords", [])
            if str(keyword).replace(" ", "").strip() in explicit_exclusions
        ]
    else:
        guarded["excluded_keywords"] = []

    allowed_keywords = {
        str(keyword).replace(" ", "").strip()
        for keyword in semantic_draft.get("keyword_candidates", [])
    }
    if allowed_keywords:
        guarded["generated_keywords"] = [
            keyword
            for keyword in guarded.get("generated_keywords", [])
            if str(keyword).replace(" ", "").strip() in allowed_keywords
        ]

    draft_search_terms = str(semantic_draft.get("search_terms_candidate") or "").strip()
    if draft_search_terms and not guarded.get("search_terms"):
        guarded["search_terms"] = draft_search_terms

    return guarded


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
        return {"value": numeric_value, "evidence": evidence, "is_explicit": is_explicit}

    return {
        "target_status_candidate": _normalize_string_candidate("target_status_candidate"),
        "author_name_candidate": _normalize_string_candidate("author_name_candidate"),
        "words_min_candidate": _normalize_number_candidate("words_min_candidate"),
        "words_max_candidate": _normalize_number_candidate("words_max_candidate"),
    }


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


def _build_semantic_to_structured_context_from_slots(semantic_slots: Dict[str, Any]) -> str:
    search_terms = str(semantic_slots.get("search_terms") or "").strip()
    keywords = [str(keyword).strip() for keyword in semantic_slots.get("generated_keywords", []) if str(keyword).strip()]
    excluded_keywords = [
        str(keyword).strip()
        for keyword in semantic_slots.get("excluded_keywords", [])
        if str(keyword).strip()
    ]

    lines = [
        "SEMANTIC BRANCH OUTPUT:",
        f"- search_terms: {search_terms or '(empty)'}",
        f"- generated_keywords: {', '.join(keywords) if keywords else '(empty)'}",
        f"- excluded_keywords: {', '.join(excluded_keywords) if excluded_keywords else '(empty)'}",
        "Use this semantic summary as additional context when deciding whether the user also expressed hard constraints.",
        "Do not invent constraints just because a semantic theme suggests them.",
    ]
    return "\n".join(lines)


def _build_semantic_fallback_result(
    user_query: str,
    started_at: float,
    fallback_reason: Optional[str] = None,
) -> QueryParseResult:
    fallback_metadata = {
        "fallback_used": "pure_semantic",
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    if fallback_reason:
        fallback_metadata["fallback_reason"] = fallback_reason

    return QueryParseResult(
        original_query=user_query,
        search_terms=user_query,
        generated_keywords=[],
        hypothetical_intro="",
        criteria=[
            ScoringCriteria(
                name="semantic_similarity",
                is_negative=False,
                parameters=ScoringParameters(query_text=user_query),
            )
        ],
        parse_metadata=fallback_metadata,
    )


def _build_model_candidates(selected_model: Optional[str]) -> List[str]:
    if selected_model:
        return [selected_model]
    return list(FALLBACK_MODELS)


def _criteria_to_key(criteria: Any) -> Tuple[str, bool, str]:
    if hasattr(criteria, "model_dump"):
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
    semantic_slots: Dict[str, Any],
    structured_slots: Dict[str, Any],
    total_latency_ms: float,
    branch_latencies_ms: Dict[str, float],
    fallback_used: Optional[str] = None,
) -> QueryParseResult:
    semantic_query_text = str(semantic_slots.get("search_terms") or user_query).strip() or user_query
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

    for excluded_keyword in semantic_slots.get("excluded_keywords", []):
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

    merged_keywords = []
    seen_keywords = set()
    for keyword in semantic_slots.get("generated_keywords", []):
        normalized = str(keyword).replace(" ", "").strip()
        if not normalized or normalized in seen_keywords:
            continue
        seen_keywords.add(normalized)
        merged_keywords.append(str(keyword).strip())

    metadata = {
        "latency_ms": round(total_latency_ms, 2),
        "branches": {name: round(latency, 2) for name, latency in branch_latencies_ms.items()},
    }
    if fallback_used:
        metadata["fallback_used"] = fallback_used

    return QueryParseResult(
        original_query=user_query,
        search_terms=semantic_query_text,
        generated_keywords=merged_keywords,
        hypothetical_intro="",
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
    from src.core.api_utils import get_api_key_rotator, get_current_api_key, get_rate_limiter

    models_to_try = _build_model_candidates(model_id or FALLBACK_MODELS[0])
    last_exception = None

    for candidate_model in models_to_try:
        print(f"[llm:{task_label}] trying model: {candidate_model}")

        rotator = get_api_key_rotator()
        api_key_attempts = 0
        max_api_key_attempts = len(rotator.api_keys)

        while api_key_attempts < max_api_key_attempts:
            try:
                api_key = get_current_api_key()
                client = genai.Client(api_key=api_key)
                max_retries = 1

                for attempt in range(max_retries + 1):
                    try:
                        if enforce_rate_limit:
                            get_rate_limiter().wait()

                        config_args = {
                            "response_mime_type": "application/json",
                            "response_schema": response_schema,
                            "system_instruction": system_instruction,
                            "temperature": sampling_temperature,
                            "top_p": 0.95,
                        }

                        response = client.models.generate_content(
                            model=candidate_model,
                            contents=contents,
                            config=types.GenerateContentConfig(**config_args),
                        )

                        parsed = getattr(response, "parsed", None)
                        if parsed is None:
                            raise ValueError(
                                f"Structured output missing parsed payload for {task_label}"
                            )
                        if isinstance(parsed, dict):
                            return parsed
                        raise ValueError(f"Expected JSON object from {task_label}, got {type(parsed).__name__}")
                    except Exception as exc:
                        last_exception = exc
                        if not _is_retryable(exc) or attempt == max_retries:
                            raise
                        print(f"[llm:{task_label}] retryable error: {exc}. Retrying in 3.0s...")
                        time.sleep(3.0)
            except Exception as exc:
                error_str = str(exc)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

                if is_rate_limit and api_key_attempts < max_api_key_attempts - 1:
                    print(f"[llm:{task_label}] current API key hit a rate limit, rotating key...")
                    rotator.on_rate_limit_error()
                    api_key_attempts += 1
                    time.sleep(2.0)
                    continue

                last_exception = exc
                print(f"[llm:{task_label}] model failed: {exc}")
                break

    if last_exception:
        raise last_exception

    raise RuntimeError(f"[llm:{task_label}] failed without a concrete exception")


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


def _parse_query_parallel_ctx(
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

    semantic_schema = {
        "type": "object",
        "properties": {
            "search_terms": {"type": "string"},
            "generated_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "excluded_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["search_terms", "generated_keywords", "excluded_keywords"],
    }

    structured_schema = {
        "type": "object",
        "properties": {
            "target_status": {"type": "string"},
            "author_name": {"type": "string"},
            "words_min": {"type": "number"},
            "words_max": {"type": "number"},
        },
        "required": [],
    }

    semantic_instruction = f"""
{shared_context}

Your branch owns only semantic intent parsing.

    Return JSON with:
    - search_terms
    - generated_keywords
    - excluded_keywords

    Rules:
    - Focus on tags, genres, themes, tropes, vibes, plot concepts, and explicit semantic exclusions.
    - Prefer exact tags from AVAILABLE TAGS in both `search_terms` and `generated_keywords` whenever possible.
    - Keep `search_terms` short, retrieval-oriented, and tag-heavy.
    - Generate 5-10 concise keywords when possible.
    - Put explicit negative tags, blocked tropes, and "銝?/?踹?/?" concepts into `excluded_keywords`.
    - `excluded_keywords` should contain short tag-like terms, not full sentences.
    - If there are no explicit exclusions, return an empty array for `excluded_keywords`.
    - Do NOT emit `criteria`, `status_check`, `author_match`, or `numeric_range`.
""".strip()

    base_structured_instruction = f"""
{shared_context}

Your branch owns only structured constraints.

    Return JSON with:
    - target_status
    - author_name
    - words_min
    - words_max

    Rules:
    - Extract only hard-constraint slots.
    - Set `target_status` only when the query explicitly asks for completion status, such as `completed`, `finished`, `ongoing`, `serializing`, `完結`, `完本`, or `連載`.
    - Never infer `target_status` from genre, tone, trope, popularity, recommendation examples, or the semantic branch output.
    - Set `author_name` only when the query explicitly mentions an author with cues like `作者`, `author`, or `作家`.
    - Set `words_min` and/or `words_max` only when the query explicitly mentions a word-count constraint with units like `字`, `萬字`, `w`, `k`, or `words`.
    - Ignore list numbering like `1.` / `2.` and any other non-word-count numbers.
    - If a slot is not specified, omit that key entirely. Prefer omission over placeholder values like `"null"` or `0`.
    - Do NOT generate `criteria`, `semantic_similarity`, `search_terms`, `generated_keywords`, or `hypothetical_intro`.
""".strip()

    def _run_branch(branch_name: str, instruction: str, schema: Dict[str, Any]) -> Tuple[str, Any, float]:
        started_at = time.perf_counter()
        raw_result = _generate_json_task(
            user_query=user_query,
            task_label=branch_name,
            system_instruction=instruction,
            response_schema=schema,
            model_id=model_id,
            sampling_temperature=sampling_temperature,
            enforce_rate_limit=False,
        )
        if DEBUG_LLM_OUTPUT:
            print(f"[debug:{branch_name}] raw={json.dumps(raw_result, ensure_ascii=False)}")
        if branch_name == "semantic":
            result = _normalize_semantic_slots(raw_result, user_query)
            if DEBUG_LLM_OUTPUT:
                print(
                    f"[debug:{branch_name}] normalized="
                    f"{json.dumps(result, ensure_ascii=False)}"
                )
        else:
            result = _normalize_structured_slots(raw_result)
            if DEBUG_LLM_OUTPUT:
                print(f"[debug:{branch_name}] normalized={json.dumps(result, ensure_ascii=False)}")
        latency_ms = (time.perf_counter() - started_at) * 1000
        return branch_name, result, latency_ms

    started_at = time.perf_counter()
    branch_results: Dict[str, Any] = {}
    branch_latencies_ms: Dict[str, float] = {}

    try:
        branch_name, result, latency_ms = _run_branch("semantic", semantic_instruction, semantic_schema)
        branch_results[branch_name] = result
        branch_latencies_ms[branch_name] = latency_ms

        semantic_context = _build_semantic_to_structured_context_from_slots(branch_results["semantic"])
        structured_instruction = f"{base_structured_instruction}\n\n{semantic_context}"
        branch_name, result, latency_ms = _run_branch("structured", structured_instruction, structured_schema)
        branch_results[branch_name] = result
        branch_latencies_ms[branch_name] = latency_ms

        total_latency_ms = (time.perf_counter() - started_at) * 1000
        merged_structured_slots = _merge_structured_slots(
            llm_slots=branch_results["structured"],
            rule_slots=_extract_structured_slots_from_query(user_query),
        )
        merged_structured_slots = _validate_structured_slots_against_query(
            user_query,
            merged_structured_slots,
        )
        if DEBUG_LLM_OUTPUT:
            print(f"[debug:structured] merged_slots={json.dumps(merged_structured_slots, ensure_ascii=False)}")
        merged_result = _merge_query_parse_results(
            user_query=user_query,
            semantic_slots=branch_results["semantic"],
            structured_slots=merged_structured_slots,
            total_latency_ms=total_latency_ms,
            branch_latencies_ms=branch_latencies_ms,
        )
        return merged_result.model_copy(
            update={
                "parse_metadata": {
                    **merged_result.parse_metadata,
                }
            }
        )
    except Exception as exc:
        print(f"[llm:parallel_ctx] parser failed, falling back to pure semantic parse: {exc}")
        return _build_semantic_fallback_result(
            user_query=user_query,
            started_at=started_at,
            fallback_reason=str(exc),
        )

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

    semantic_schema = {
        "type": "object",
        "properties": {
            "search_terms": {"type": "string"},
            "generated_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "excluded_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["search_terms", "generated_keywords", "excluded_keywords"],
    }

    exclusion_candidate_schema = {
        "type": "object",
        "properties": {
            "keyword": {"type": "string"},
            "evidence": {"type": "string"},
            "is_explicit": {"type": "boolean"},
        },
        "required": ["keyword", "evidence", "is_explicit"],
    }

    semantic_draft_schema = {
        "type": "object",
        "properties": {
            "search_terms_candidate": {"type": "string"},
            "keyword_candidates": {
                "type": "array",
                "items": {"type": "string"},
            },
            "excluded_candidates": {
                "type": "array",
                "items": exclusion_candidate_schema,
            },
        },
        "required": ["search_terms_candidate", "keyword_candidates", "excluded_candidates"],
    }

    structured_schema = {
        "type": "object",
        "properties": {
            "target_status": {"type": "string"},
            "author_name": {"type": "string"},
            "words_min": {"type": "number"},
            "words_max": {"type": "number"},
        },
        "required": [],
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

    structured_draft_schema = {
        "type": "object",
        "properties": {
            "target_status_candidate": string_candidate_schema,
            "author_name_candidate": string_candidate_schema,
            "words_min_candidate": number_candidate_schema,
            "words_max_candidate": number_candidate_schema,
        },
        "required": [],
    }

    semantic_draft_instruction = f"""
{shared_context}

You are the semantic understanding pass.

Return JSON with:
- search_terms_candidate
- keyword_candidates
- excluded_candidates

Rules:
- Focus on genres, themes, vibes, plot concepts, and retrieval-friendly tags.
- Prefer exact tags from AVAILABLE TAGS whenever possible.
- Keep `search_terms_candidate` short and retrieval-oriented.
- Put 5-10 concise positive retrieval ideas into `keyword_candidates` when possible.
- Use `excluded_candidates` only for directly rejected concepts.
- Each excluded candidate must contain a short `keyword`, a short verbatim `evidence` quote copied from the user query, and `is_explicit`.
- Set `is_explicit` to true only when the query directly rejects that concept.
- Do not infer trope labels from personality descriptions or soft preferences.
- If a concept is only implied or uncertain, do not place it in `excluded_candidates`.
- Do NOT output structured constraints here.
""".strip()

    semantic_instruction = f"""
{shared_context}

You are the semantic serialization pass.

Return JSON with:
- search_terms
- generated_keywords
- excluded_keywords

Rules:
- The input includes the original query and a semantic draft JSON.
- Base your output on that draft instead of inventing new concepts.
- `generated_keywords` should be chosen from `keyword_candidates`.
- `excluded_keywords` may contain only directly rejected concepts from `excluded_candidates`.
- If no exclusions are directly supported, return an empty array for `excluded_keywords`.
- Keep `search_terms` short, retrieval-oriented, and tag-heavy.
""".strip()

    structured_draft_instruction = f"""
{shared_context}

You are the structured understanding pass.

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
""".strip()

    structured_instruction = f"""
{shared_context}

You are the structured serialization pass.

Return JSON with:
- target_status
- author_name
- words_min
- words_max

Rules:
- The input includes the original query, semantic context, and structured draft JSON.
- Emit only fields that are directly supported by explicit draft candidates.
- Do not invent missing fields.
- `target_status` must be either `completed` or `ongoing` when present.
- Omit uncertain fields instead of guessing.
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

    try:
        semantic_draft, latency_ms = _run_schema_task(
            "semantic_draft",
            semantic_draft_instruction,
            semantic_draft_schema,
            normalizer=_normalize_semantic_draft,
        )
        branch_latencies_ms["semantic_draft"] = latency_ms

        semantic_contents = (
            f"Original Query:\n{user_query}\n\n"
            f"Semantic Draft JSON:\n{json.dumps(semantic_draft, ensure_ascii=False)}"
        )
        semantic_slots, latency_ms = _run_schema_task(
            "semantic",
            semantic_instruction,
            semantic_schema,
            contents=semantic_contents,
            normalizer=lambda raw: _apply_semantic_draft_guards(
                user_query,
                _normalize_semantic_slots(raw, user_query),
                semantic_draft,
            ),
        )
        branch_latencies_ms["semantic"] = latency_ms

        semantic_context = _build_semantic_to_structured_context_from_slots(semantic_slots)
        structured_draft_contents = (
            f"Original Query:\n{user_query}\n\n"
            f"{semantic_context}"
        )
        structured_draft, latency_ms = _run_schema_task(
            "structured_draft",
            structured_draft_instruction,
            structured_draft_schema,
            contents=structured_draft_contents,
            normalizer=_normalize_structured_draft,
        )
        branch_latencies_ms["structured_draft"] = latency_ms

        structured_contents = (
            f"Original Query:\n{user_query}\n\n"
            f"{semantic_context}\n\n"
            f"Structured Draft JSON:\n{json.dumps(structured_draft, ensure_ascii=False)}"
        )
        structured_slots, latency_ms = _run_schema_task(
            "structured",
            structured_instruction,
            structured_schema,
            contents=structured_contents,
            normalizer=lambda raw: _apply_structured_draft_guards(
                user_query,
                _normalize_structured_slots(raw),
                structured_draft,
            ),
        )
        branch_latencies_ms["structured"] = latency_ms

        total_latency_ms = (time.perf_counter() - started_at) * 1000
        if DEBUG_LLM_OUTPUT:
            print(f"[debug:structured] merged_slots={json.dumps(structured_slots, ensure_ascii=False)}")

        merged_result = _merge_query_parse_results(
            user_query=user_query,
            semantic_slots=semantic_slots,
            structured_slots=structured_slots,
            total_latency_ms=total_latency_ms,
            branch_latencies_ms=branch_latencies_ms,
        )
        return merged_result.model_copy(
            update={
                "parse_metadata": {
                    **merged_result.parse_metadata,
                    "parser_mode": "response_schema_two_stage",
                }
            }
        )
    except Exception as exc:
        print(f"[llm:parallel_ctx_v2] parser failed, falling back to pure semantic parse: {exc}")
        return _build_semantic_fallback_result(
            user_query=user_query,
            started_at=started_at,
            fallback_reason=str(exc),
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
