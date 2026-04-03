import json
from typing import Any, Dict, List, Tuple


DEFAULT_BOOK_SCORE = 3.0


def normalize_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        text = raw_tags.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return normalize_tags(parsed)
    if isinstance(raw_tags, dict) and "data" in raw_tags:
        raw_tags = raw_tags["data"]
    if not isinstance(raw_tags, (list, tuple, set)):
        return []

    normalized: List[str] = []
    for tag in raw_tags:
        value = str(tag).strip()
        if value:
            normalized.append(value)
    return normalized


def tag_matches(required_tag: str, candidate_tag: str) -> bool:
    required = str(required_tag).strip()
    candidate = str(candidate_tag).strip()
    if not required or not candidate:
        return False
    return required == candidate or required in candidate or candidate in required


def apply_hard_filters(golden_rules: Dict[str, Any], book_item: Dict[str, Any]) -> bool:
    min_words = golden_rules.get("min_words")
    max_words = golden_rules.get("max_words")
    words_total = book_item.get("words_total", 0)

    if min_words is not None and words_total < min_words:
        return False
    if max_words is not None and words_total > max_words:
        return False

    req_status = golden_rules.get("required_status")
    if req_status:
        status = str(book_item.get("publish_status", "")).lower()
        if req_status == "completed" and status not in ["completed", "已完結", "完結"]:
            return False
        if req_status == "ongoing" and status not in ["ongoing", "連載中", "連載"]:
            return False

    must_be_animated = golden_rules.get("must_be_animated")
    if must_be_animated is not None:
        if bool(book_item.get("is_animated")) != bool(must_be_animated):
            return False

    blocked_tags = normalize_tags(golden_rules.get("blocked_tags") or [])
    if blocked_tags:
        book_tags = normalize_tags(book_item.get("tags", []))
        for bt in blocked_tags:
            if any(tag_matches(bt, book_tag) for book_tag in book_tags):
                return False

    return True


def score_required_tags(
    golden_rules: Dict[str, Any],
    book_item: Dict[str, Any],
    max_score: float = DEFAULT_BOOK_SCORE,
) -> Tuple[float, int, int, List[Tuple[str, str]]]:
    required_tags = normalize_tags(golden_rules.get("required_tags") or [])
    if not required_tags:
        return max_score, 0, 0, []

    book_tags = normalize_tags(book_item.get("tags", []))
    if not book_tags:
        return 0.0, 0, len(required_tags), []

    remaining_tags = list(book_tags)
    matched_pairs: List[Tuple[str, str]] = []

    for required_tag in required_tags:
        match_index = next(
            (
                index
                for index, candidate_tag in enumerate(remaining_tags)
                if tag_matches(required_tag, candidate_tag)
            ),
            None,
        )
        if match_index is None:
            continue
        matched_tag = remaining_tags.pop(match_index)
        matched_pairs.append((required_tag, matched_tag))

    required_count = len(required_tags)
    match_count = len(matched_pairs)
    if match_count == 0:
        return 0.0, 0, required_count, []

    candidate_count = len(book_tags)
    denominator = min(required_count, candidate_count)
    if denominator <= 0:
        return 0.0, match_count, required_count, matched_pairs

    progress = match_count / denominator
    return max_score * progress, match_count, required_count, matched_pairs
