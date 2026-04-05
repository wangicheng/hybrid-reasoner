"""Query synonym normalization using data/tags/tags_switch.

This module is intentionally standalone and only depends on the Python standard
library. It is designed to be called after book-title matching and before LLM
query parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


_HEADING_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")


@dataclass(frozen=True)
class _ConceptEntry:
    canonical: str
    synonyms: Tuple[str, ...]
    antonyms: Tuple[str, ...]


def _default_tags_switch_path() -> Path:
    # src/core/query_synonym_normalizer.py -> project root -> data/tags/tags_switch
    return Path(__file__).resolve().parents[2] / "data" / "tags" / "tags_switch"


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _split_terms(raw: str) -> List[str]:
    text = str(raw or "")
    for symbol in ["、", "，", ",", "；", ";", "。", "\n", "\t"]:
        text = text.replace(symbol, ",")
    terms: List[str] = []
    seen: Set[str] = set()
    for part in text.split(","):
        token = part.strip().strip("\"'()[]{}")
        if not token:
            continue
        normalized = _normalize_term(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token)
    return terms


def _parse_tags_switch(content: str) -> List[_ConceptEntry]:
    entries: List[_ConceptEntry] = []
    canonical: Optional[str] = None
    synonyms_raw = ""
    antonyms_raw = ""

    def flush_current() -> None:
        nonlocal canonical, synonyms_raw, antonyms_raw
        if not canonical:
            return

        synonym_terms = _split_terms(synonyms_raw)
        antonym_terms = _split_terms(antonyms_raw)

        if canonical not in synonym_terms:
            synonym_terms.insert(0, canonical)

        entries.append(
            _ConceptEntry(
                canonical=canonical,
                synonyms=tuple(synonym_terms),
                antonyms=tuple(antonym_terms),
            )
        )

        canonical = None
        synonyms_raw = ""
        antonyms_raw = ""

    for line in content.splitlines():
        raw = line.strip()
        if not raw:
            continue

        heading_match = _HEADING_RE.match(raw)
        if heading_match:
            flush_current()
            canonical = heading_match.group(1).strip()
            continue

        if raw.startswith("同義詞："):
            synonyms_raw = f"{synonyms_raw} {raw.split('：', 1)[1].strip()}".strip()
            continue

        if raw.startswith("反義詞："):
            antonyms_raw = f"{antonyms_raw} {raw.split('：', 1)[1].strip()}".strip()
            continue

        # Allow multiline continuation for very long term lists.
        if canonical and synonyms_raw and not antonyms_raw:
            synonyms_raw = f"{synonyms_raw} {raw}".strip()
        elif canonical and antonyms_raw:
            antonyms_raw = f"{antonyms_raw} {raw}".strip()

    flush_current()
    return entries


@lru_cache(maxsize=2)
def _load_entries(path_value: str) -> Tuple[_ConceptEntry, ...]:
    path = Path(path_value)
    if not path.exists():
        return tuple()

    text: str
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-16")

    return tuple(_parse_tags_switch(text))


def _is_ascii_word(term: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'\-\.\s]*", term))


def _canonical_hint_text(term: str) -> str:
    value = str(term or "").strip()
    if _is_ascii_word(value):
        return value.lower()
    return value


def _iter_matches(text: str, alias: str) -> Iterable[Tuple[int, int, str]]:
    if _is_ascii_word(alias):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            yield m.start(), m.end(), m.group(0)
        return

    start = 0
    while True:
        index = text.find(alias, start)
        if index < 0:
            return
        yield index, index + len(alias), alias
        start = index + len(alias)


def _already_has_hint(text: str, end: int, target: str) -> bool:
    hint = _canonical_hint_text(target)
    rest = text[end: end + len(hint) + 4]
    compact = re.sub(r"\s+", "", rest)
    return compact.startswith(f"({hint})") or compact.startswith(f"（{hint}）")


def normalize_query_after_book_lookup(
    user_query: str,
    tags_switch_path: Optional[str] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """Normalize user query by appending canonical hints in parentheses.

    This function uses literal term occurrence only. It intentionally avoids
    semantic/context judgment and simply annotates matched terms.

    Returns:
        (normalized_query, annotations)
        annotations item schema:
            {
                "from": matched synonym,
                "to": canonical term used in hint
            }
    """
    query = str(user_query or "")
    if not query.strip():
        return query, []

    path = str(Path(tags_switch_path) if tags_switch_path else _default_tags_switch_path())
    entries = _load_entries(path)
    if not entries:
        return query, []

    alias_to_canonical: Dict[str, str] = {}
    alias_to_display: Dict[str, str] = {}
    canonical_norms: Set[str] = set()

    for entry in entries:
        canonical_norms.add(_normalize_term(entry.canonical))
        for synonym in entry.synonyms:
            synonym_norm = _normalize_term(synonym)
            if not synonym_norm or len(synonym_norm) < 2:
                continue
            if synonym_norm == _normalize_term(entry.canonical):
                continue
            # Keep first mapping by file order and annotate by literal match only.
            if synonym_norm not in alias_to_canonical:
                alias_to_canonical[synonym_norm] = entry.canonical
                alias_to_display[synonym_norm] = synonym

    if not alias_to_canonical:
        return query, []

    candidates: List[Tuple[int, int, str, str]] = []
    for alias_norm, canonical in alias_to_canonical.items():
        alias_display = alias_to_display[alias_norm]
        for start, end, matched_text in _iter_matches(query, alias_display):
            if _normalize_term(matched_text) == _normalize_term(canonical):
                continue

            if _normalize_term(canonical) not in canonical_norms:
                continue

            if _already_has_hint(query, end, canonical):
                continue

            candidates.append((start, end, matched_text, canonical))

    if not candidates:
        return query, []

    # Apply non-overlapping annotations, preferring earlier and longer matches.
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: List[Tuple[int, int, str, str]] = []
    occupied = [False] * len(query)
    for start, end, source, target in candidates:
        if any(occupied[i] for i in range(start, end)):
            continue
        selected.append((start, end, source, target))
        for i in range(start, end):
            occupied[i] = True

    if not selected:
        return query, []

    selected.sort(key=lambda item: item[1], reverse=True)
    rewritten = query
    annotations: List[Dict[str, str]] = []
    for start, end, source, target in selected:
        hint = _canonical_hint_text(target)
        rewritten = f"{rewritten[:end]}({hint}){rewritten[end:]}"
        annotations.append({"from": source, "to": hint})

    annotations.reverse()
    return rewritten, annotations
