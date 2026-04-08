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

    field = to_string(normalized.get("field"))
    if field and field != "words_total":
        field = None

    return {
        "target_status": status,
        "author_name": author,
        "words_min": words_min,
        "words_max": words_max,
        "field": field or "words_total",
    }


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

    return {
        "target_status": status,
        "author_name": author_name,
        "words_min": words_min,
        "words_max": words_max,
        "field": "words_total",
    }


def _merge_structured_slots(llm_slots: Dict[str, Any], rule_slots: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "target_status": llm_slots.get("target_status") or rule_slots.get("target_status"),
        "author_name": llm_slots.get("author_name") or rule_slots.get("author_name"),
        "words_min": llm_slots.get("words_min"),
        "words_max": llm_slots.get("words_max"),
        "field": "words_total",
    }

    if merged["words_min"] is None:
        merged["words_min"] = rule_slots.get("words_min")
    if merged["words_max"] is None:
        merged["words_max"] = rule_slots.get("words_max")

    return merged


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
    if selected_model and selected_model in FALLBACK_MODELS:
        return [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
    if selected_model:
        return [selected_model] + FALLBACK_MODELS
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
    tag_context: Optional[str] = None,
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

    if tag_context:
        sections.append(
            "TAG DESCRIPTIONS:\n"
            "Use these descriptions as soft semantic grounding, but do not invent tags outside the allowed list.\n"
            f"{tag_context}"
        )

    if reference_book_context:
        sections.append(
            "RELATED BOOKS:\n"
            "Use these only as soft context for interpreting the query. They are not hard retrieval constraints.\n"
            f"{reference_book_context}"
        )

    return "\n\n".join(sections)


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
    from src.core.api_utils import get_api_key_rotator, get_current_api_key, get_rate_limiter

    models_to_try = _build_model_candidates(model_id or FALLBACK_MODELS[0])
    last_exception = None

    for candidate_model in models_to_try:
        print(f"[llm:{task_label}] trying model: {candidate_model}")

        is_gemma = "gemma" in candidate_model.lower()
        rotator = get_api_key_rotator()
        api_key_attempts = 0
        max_api_key_attempts = len(rotator.api_keys)

        while api_key_attempts < max_api_key_attempts:
            try:
                api_key = get_current_api_key()
                client = genai.Client(api_key=api_key)
                max_retries = 2 if is_gemma else 1

                for attempt in range(max_retries + 1):
                    try:
                        if enforce_rate_limit:
                            get_rate_limiter().wait()

                        final_prompt = f"User Query: {user_query}"
                        if is_gemma:
                            config_args = {
                                "temperature": sampling_temperature,
                                "top_p": 0.95,
                            }
                            final_contents = (
                                f"{system_instruction}\n\n"
                                f"Task Input:\n{final_prompt}\n\n"
                                "IMPORTANT FIELD TYPES:\n"
                                "- search_terms must be a single string, not an array.\n"
                                "- generated_keywords must be an array of strings.\n"
                                "- hypothetical_intro must be a single string.\n"
                                "- criteria must be an array of objects.\n\n"
                                "IMPORTANT: Output ONLY valid JSON (no markdown). Ensure keys are snake_case."
                            )
                        else:
                            config_args = {
                                "response_mime_type": "application/json",
                                "response_schema": response_schema,
                                "system_instruction": system_instruction,
                                "temperature": sampling_temperature,
                                "top_p": 0.95,
                            }
                            final_contents = final_prompt

                        response = client.models.generate_content(
                            model=candidate_model,
                            contents=final_contents,
                            config=types.GenerateContentConfig(**config_args),
                        )

                        if not response.text:
                            raise ValueError("Empty response from LLM")

                        raw_text = response.text.strip()
                        raw_text = raw_text.removeprefix("```json").removeprefix("```").strip()
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3].strip()

                        parsed = json.loads(raw_text)
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


def _parse_query_parallel_ctx(
    user_query: str,
    model_id: Optional[str] = None,
    tag_list: Optional[Tuple[str, ...]] = None,
    tag_context: Optional[str] = None,
    reference_book_context: Optional[str] = None,
    sampling_temperature: float = 0.2,
) -> QueryParseResult:
    shared_context = _build_parallel_context(
        tag_list=tag_list,
        tag_context=tag_context,
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
    - If the user explicitly wants completed / ongoing, set `target_status` to `completed` or `ongoing`.
    - If the user explicitly names an author, set `author_name`.
    - If the user gives a strict word-count requirement, set `words_min` and/or `words_max` using actual word counts.
    - If a slot is not specified, omit it or return null.
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

@functools.lru_cache(maxsize=1000)
def parse_query(
    user_query: str,
    model_id: Optional[str] = None,
    cache_namespace: Optional[str] = None,
    tag_list: Optional[Tuple[str, ...]] = None,
    tag_context: Optional[str] = None,
    reference_book_context: Optional[str] = None,
    sampling_temperature: float = 0.2,
) -> QueryParseResult:
    _ = cache_namespace
    return _parse_query_parallel_ctx(
        user_query,
        model_id=model_id,
        tag_list=tag_list,
        tag_context=tag_context,
        reference_book_context=reference_book_context,
        sampling_temperature=sampling_temperature,
    )
