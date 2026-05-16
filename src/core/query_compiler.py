"""
Query Compiler — Phase 1 of the HyST Architecture Upgrade.

Pure-function module that splits LLM-parsed intent into two groups:

  * **Hard Filters** (delegated to Qdrant server-side pre-filtering):
      - must_not:  negative/blocked tags
      - must:      completion status, word-count range

  * **Soft Factors** (kept in Python memory for scoring):
      - positive_tags:  desired tags for Tag Match Ratio bonus
      - tag_terms_list: deduplicated terms for tag-mapping weights

This module is intentionally side-effect-free: it accepts parsed data and
returns a structured ``CompiledQuery``.  The actual Qdrant filter DSL
construction is delegated to ``VectorStore.build_metadata_filter()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HardFilters:
    """Constraints enforced at the database (Qdrant) level."""

    negative_tag_terms: List[str] = field(default_factory=list)
    status_filter: Optional[str] = None
    author_filter: Optional[str] = None
    words_min: Optional[int] = None
    words_max: Optional[int] = None

    def to_constraint_dict(self) -> Dict[str, Any]:
        """Produce the dict expected by ``VectorStore.build_metadata_filter()``."""
        return {
            "negative_tag_terms": list(self.negative_tag_terms),
            "status_filter": self.status_filter,
            "author_filter": self.author_filter,
            "words_min": self.words_min,
            "words_max": self.words_max,
        }


@dataclass(frozen=True)
class SoftFactors:
    """Scoring factors evaluated in Python memory."""

    positive_tags: List[str] = field(default_factory=list)
    tag_terms_list: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompiledQuery:
    """The output of the Query Compiler.

    Consumers use ``hard_filters`` to build Qdrant pre-filter DSL and
    ``soft_factors`` for in-memory soft scoring.
    """

    hard_filters: HardFilters = field(default_factory=HardFilters)
    soft_factors: SoftFactors = field(default_factory=SoftFactors)


# ─────────────────────────────────────────────────────────────────────
# Compiler logic
# ─────────────────────────────────────────────────────────────────────

def _normalize_status(status_value: str) -> Optional[str]:
    """Normalise raw status strings to 'completed' | 'ongoing' | None."""
    raw_value = str(status_value or "").strip()
    lowered = raw_value.lower()

    completed_keywords = ["complet", "finish", "ended", "done", "完結", "已完結"]
    ongoing_keywords = ["ongoing", "serializ", "running", "active", "連載", "連載中"]

    if any(kw in lowered or kw in raw_value for kw in completed_keywords):
        return "completed"
    if any(kw in lowered or kw in raw_value for kw in ongoing_keywords):
        return "ongoing"
    return None


def _dedupe_terms(terms: List[str]) -> List[str]:
    """Remove duplicate terms (space-insensitive)."""
    seen: set[str] = set()
    deduped: List[str] = []
    for term in terms:
        normalised = term.replace(" ", "").strip()
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        deduped.append(normalised)
    return deduped


def compile_query(
    parse_result: Any,
    negative_tag_terms: List[str],
) -> CompiledQuery:
    """Compile an LLM parse result into Hard Filters + Soft Factors.

    Parameters
    ----------
    parse_result : QueryParseResult
        Structured output from ``parse_query()``.
    negative_tag_terms : list[str]
        Pre-resolved negative tag terms (already deduplicated).

    Returns
    -------
    CompiledQuery
        Immutable compiled query with hard_filters and soft_factors.
    """

    # ── 1. Extract hard constraints from criteria ──
    status_filter: Optional[str] = None
    author_filter: Optional[str] = None
    words_min: Optional[int] = None
    words_max: Optional[int] = None

    for criteria in parse_result.criteria:
        params = _criteria_params(criteria)

        if criteria.name == "status_check":
            status_filter = _normalize_status(params.get("target_status", ""))

        elif criteria.name == "author_match":
            val = params.get("author_name", "").strip()
            if val:
                author_filter = val

        elif criteria.name == "numeric_range" and params.get("field") == "words_total":
            if params.get("min_val") is not None:
                words_min = int(params["min_val"])
            if params.get("max_val") is not None:
                words_max = int(params["max_val"])

    hard_filters = HardFilters(
        negative_tag_terms=list(negative_tag_terms),
        status_filter=status_filter,
        author_filter=author_filter,
        words_min=words_min,
        words_max=words_max,
    )

    # ── 2. Extract soft factors ──
    positive_tags = list(parse_result.tag_intent.positive_terms) if hasattr(parse_result, "tag_intent") else []
    fuzzy_positive_tags = list(parse_result.tag_intent.fuzzy_positive_terms) if hasattr(parse_result, "tag_intent") else []
    
    # We store the combined exact + fuzzy terms in tag_terms_list
    tag_keywords = positive_tags + fuzzy_positive_tags
    if not tag_keywords:
        tag_keywords = list(parse_result.generated_keywords)
        
    tag_terms_list = _dedupe_terms(tag_keywords)

    soft_factors = SoftFactors(
        positive_tags=positive_tags,
        tag_terms_list=tag_terms_list,
    )

    return CompiledQuery(hard_filters=hard_filters, soft_factors=soft_factors)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _criteria_params(criteria: Any) -> Dict[str, Any]:
    """Extract parameters dict from a ScoringCriteria object."""
    params = getattr(criteria, "parameters", {})
    if hasattr(params, "model_dump"):
        return params.model_dump()
    if hasattr(params, "dict"):
        return params.dict()
    return dict(params)
