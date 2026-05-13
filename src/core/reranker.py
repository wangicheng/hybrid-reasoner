"""
Permutation Self-Consistency (PermSC) Reranker.

Based on "Found in the Middle: Permutation Self-Consistency
Improves Listwise Ranking in Large Language Models" (Tang et al., 2024).

This module provides a production-ready reranker that mitigates the
"Lost in the Middle" position bias problem in LLM listwise ranking by
generating multiple permuted rankings and aggregating them via
time-bounded Kemeny-Young approximation.
"""

import asyncio
import json
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from src.core.api_utils import _is_retryable, get_api_key_rotator, get_rate_limiter


class PermSCReranker:
    """
    Listwise reranker using Permutation Self-Consistency (PermSC).

    The reranker sends the same candidate list in N different orderings to
    the LLM, collects per-permutation rankings, and fuses them via
    Kemeny-Young aggregation to produce a position-bias-resilient
    final ranking.

    Parameters
    ----------
    model_id : str
        Gemini / Gemma model identifier for the reranking LLM.
    n_permutations : int
        Number of permuted orderings to evaluate (default: 5).
        More permutations = more robust but slower / higher cost.
    top_k : int
        Number of top candidates each permutation should select (default: 10).
    """

    def __init__(
        self,
        model_id: str = "gemma-4-31b-it",
        n_permutations: int = 5,
        top_k: int = 10,
        kemeny_time_budget_seconds: float = 10.0,
    ) -> None:
        self.model_id = model_id
        self.n_permutations = n_permutations
        self.top_k = top_k
        self.kemeny_time_budget_seconds = max(0.1, float(kemeny_time_budget_seconds))
        self.rotator = get_api_key_rotator()
        self.rate_limiter = get_rate_limiter()

    def _rotate_api_key(self) -> None:
        new_key = self.rotator.on_rate_limit_error()
        print(f"  [Reranker] API key rotated. Current index: {self.rotator.current_index}")

    def _pick_api_key(self, index: int) -> str:
        key_pool = self.rotator.api_keys
        if not key_pool:
            return self.rotator.get_current_key()
        return key_pool[index % len(key_pool)]

    async def _get_single_ranking(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        api_key: str,
    ) -> List[str]:
        """Ask the LLM to rank candidates for one permutation order."""
        candidates_text = ""
        for i, c in enumerate(candidates):
            tags_preview = ", ".join(str(t) for t in c.get("tags", [])[:3])
            intro_preview = str(c.get("intro", ""))[:100]
            candidates_text += f"[{i}] {c['name']} (Tags: {tags_preview}) - {intro_preview}...\n"

        prompt = f"""\
You are an expert web novel recommender. The user is looking for novels based on the following query:
User Query: {query}

Below is a list of {len(candidates)} candidate novels. Each novel has an ID [number], Title, Tags, and a short Intro.
Please select the top {self.top_k} most relevant novels from this list and rank them from best (1) to worst ({self.top_k}).

Candidate Novels:
{candidates_text}

Output exactly a JSON object containing a list of the IDs of the top {self.top_k} books you selected, in ranked order.
Format:
{{
  "top_{self.top_k}_ids": [id1, id2, ..., id{self.top_k}]
}}
Do not output anything else.
"""
        attempt = 0
        max_attempts = 5
        key_pool = list(self.rotator.api_keys or [])
        if not key_pool:
            key_pool = [api_key]
        key_index = key_pool.index(api_key) if api_key in key_pool else 0
        while attempt < max_attempts:
            try:
                active_key = key_pool[key_index]
                self.rate_limiter.wait(active_key)
                client = genai.Client(api_key=active_key)
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )

                if not response.text:
                    raise ValueError("Empty response from reranking LLM")

                text = response.text.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text.rsplit("\n", 1)[0]
                text = text.strip()

                parsed = json.loads(text)
                top_ids = parsed.get(f"top_{self.top_k}_ids", [])

                # Convert list index → book_id (top_k)
                ranked_book_ids = []
                for idx in top_ids:
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        ranked_book_ids.append(candidates[idx]["book_id"])

                # Expand to full-length ranking by appending the remaining
                # candidates in the current permutation order.
                ranked_set = set(ranked_book_ids)
                remaining = [
                    candidate["book_id"]
                    for candidate in candidates
                    if candidate["book_id"] not in ranked_set
                ]
                return ranked_book_ids + remaining

            except Exception as exc:
                attempt += 1
                if not _is_retryable(exc):
                    print(f"  [Reranker] Non-retryable error: {exc}")
                    break

                error_text = str(exc)
                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    if len(key_pool) > 1:
                        key_index = (key_index + 1) % len(key_pool)
                    else:
                        self._rotate_api_key()
                await asyncio.sleep(2.0)

        return []

    def _collect_ranked_ids(self, rankings: List[List[str]]) -> List[str]:
        ranked_ids: List[str] = []
        seen = set()
        for ranking in rankings:
            for book_id in ranking:
                if book_id in seen:
                    continue
                seen.add(book_id)
                ranked_ids.append(book_id)
        return ranked_ids

    def _build_pairwise_preferences(
        self,
        candidate_ids: List[str],
        rankings: List[List[str]],
    ) -> Dict[str, Dict[str, int]]:
        pref: Dict[str, Dict[str, int]] = {
            book_id: defaultdict(int) for book_id in candidate_ids
        }
        for ranking in rankings:
            for i, higher in enumerate(ranking):
                if higher not in pref:
                    continue
                for lower in ranking[i + 1 :]:
                    if lower not in pref:
                        continue
                    pref[higher][lower] += 1
        return pref

    def _initial_order(
        self,
        candidate_ids: List[str],
        pref: Dict[str, Dict[str, int]],
    ) -> List[str]:
        def net_wins(book_id: str) -> int:
            score = 0
            for other_id in candidate_ids:
                if other_id == book_id:
                    continue
                score += pref[book_id].get(other_id, 0)
                score -= pref[other_id].get(book_id, 0)
            return score

        return sorted(candidate_ids, key=net_wins, reverse=True)

    def _approximate_kemeny_order(
        self,
        rankings: List[List[str]],
    ) -> List[str]:
        candidate_ids = self._collect_ranked_ids(rankings)
        if len(candidate_ids) <= 1:
            return candidate_ids

        pref = self._build_pairwise_preferences(candidate_ids, rankings)
        order = self._initial_order(candidate_ids, pref)

        started_at = time.monotonic()
        deadline = started_at + self.kemeny_time_budget_seconds

        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            for i in range(len(order) - 1):
                if time.monotonic() >= deadline:
                    break
                left = order[i]
                right = order[i + 1]
                delta = pref[right].get(left, 0) - pref[left].get(right, 0)
                if delta > 0:
                    order[i], order[i + 1] = right, left
                    improved = True

        return order

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using PermSC Borda Count aggregation.

        Each candidate dict must contain at least:
          - ``book_id``  (str)
          - ``name``     (str)
          - ``tags``     (List[str])
          - ``intro``    (str)

        Returns the full candidate list sorted by Borda score (descending).
        """
        if not candidates:
            return candidates

        tasks = []
        for i in range(self.n_permutations):
            shuffled = list(candidates)
            random.shuffle(shuffled)
            api_key = self._pick_api_key(i)
            tasks.append(self._get_single_ranking(query, shuffled, api_key))

        rankings = await asyncio.gather(*tasks)
        rankings = [ranking for ranking in rankings if ranking]
        if not rankings:
            return candidates

        kemeny_order = self._approximate_kemeny_order(rankings)
        kemeny_set = set(kemeny_order)

        id_to_candidate = {c["book_id"]: c for c in candidates}
        reranked: List[Dict[str, Any]] = []
        for rank, book_id in enumerate(kemeny_order, start=1):
            candidate = id_to_candidate.get(book_id)
            if candidate is None:
                continue
            candidate["kemeny_rank"] = rank
            reranked.append(candidate)

        remaining = [
            c for c in sorted(candidates, key=lambda x: x.get("original_rank", 0))
            if c.get("book_id") not in kemeny_set
        ]
        reranked.extend(remaining)
        return reranked
