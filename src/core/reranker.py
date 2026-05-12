"""
Permutation Self-Consistency (PermSC) Reranker.

Based on "Found in the Middle: Permutation Self-Consistency
Improves Listwise Ranking in Large Language Models" (Tang et al., 2024).

This module provides a production-ready reranker that mitigates the
"Lost in the Middle" position bias problem in LLM listwise ranking by
generating multiple permuted rankings and aggregating them via Borda Count.
"""

import asyncio
import json
import random
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
    Borda Count to produce a position-bias-resilient final ranking.

    Parameters
    ----------
    model_id : str
        Gemini / Gemma model identifier for the reranking LLM.
    n_permutations : int
        Number of permuted orderings to evaluate (default: 3).
        More permutations = more robust but slower / higher cost.
    top_k : int
        Number of top candidates each permutation should select (default: 10).
    """

    def __init__(
        self,
        model_id: str = "gemma-4-31b-it",
        n_permutations: int = 3,
        top_k: int = 10,
    ) -> None:
        self.model_id = model_id
        self.n_permutations = n_permutations
        self.top_k = top_k
        self.rotator = get_api_key_rotator()
        self.client = genai.Client(api_key=self.rotator.get_current_key())
        self.rate_limiter = get_rate_limiter()

    def _rotate_api_key(self) -> None:
        new_key = self.rotator.on_rate_limit_error()
        self.client = genai.Client(api_key=new_key)
        print(f"  [Reranker] API key rotated. Current index: {self.rotator.current_index}")

    async def _get_single_ranking(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
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
        while attempt < max_attempts:
            try:
                self.rate_limiter.wait()
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
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

                # Convert list index → book_id
                ranked_book_ids = []
                for idx in top_ids:
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        ranked_book_ids.append(candidates[idx]["book_id"])
                return ranked_book_ids

            except Exception as exc:
                attempt += 1
                if not _is_retryable(exc):
                    print(f"  [Reranker] Non-retryable error: {exc}")
                    break

                error_text = str(exc)
                # Rotate key on any retryable error (like 500 INTERNAL) to try another key/project context
                self._rotate_api_key()
                await asyncio.sleep(2.0)

        return []

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
            if i == 0:
                pass  # Original order
            elif i == 1:
                shuffled.reverse()  # Reverse order
            else:
                random.seed(42 + i)
                random.shuffle(shuffled)

            tasks.append(self._get_single_ranking(query, shuffled))

        rankings = await asyncio.gather(*tasks)

        # Borda Count Aggregation
        borda_scores: Dict[str, int] = defaultdict(int)
        for ranked_list in rankings:
            n_items = len(ranked_list)
            for rank_pos, book_id in enumerate(ranked_list):
                # 1st gets n_items points, 2nd gets n_items-1, etc.
                borda_scores[book_id] += (n_items - rank_pos)

        for c in candidates:
            c["borda_score"] = borda_scores.get(c["book_id"], 0)

        reranked = sorted(
            candidates,
            key=lambda x: (x.get("borda_score", 0), -x.get("original_rank", 0)),
            reverse=True,
        )
        return reranked
