"""
Baseline Experiment Runner
==========================
Generates experiment results for three baseline retrieval strategies
to compare against the full HybridEngine (LLM-parsed Hybrid RAG).

Baseline 1 — Pure BM25 (Lexical Baseline)
    Proves the necessity of semantic understanding.
    Uses only keyword-based BM25 retrieval; no embeddings, no LLM parsing.

Baseline 2 — Pure Dense (Naive Vector RAG)
    Proves the necessity of structural filtering.
    Uses only vector similarity search; no filters, no tag mapping, no LLM.

Baseline 3 — Naive Hybrid (BM25 + Vector + RRF)
    Proves the superiority of LLM-based intent decomposition.
    Combines BM25 + vector results with Reciprocal Rank Fusion (RRF),
    but does NOT use LLM to parse user intent or extract structured constraints.

Usage:
    python -m src.eval.generate_baseline_runs [--repeats N] [--experiment-dir DIR]
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.database import Database
from src.core.lexical_store import LexicalStore
from src.core.vector_store import VectorStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─── Helpers ──────────────────────────────────────────────────────────

def _normalize_tags(raw_tags: Any) -> List[str]:
    """Parse tags from various formats into a flat list of strings."""
    if isinstance(raw_tags, str):
        try:
            raw_tags = json.loads(raw_tags)
        except Exception:
            return []
    if isinstance(raw_tags, list):
        return [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    return []


def _item_to_result(item: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Convert a DB/payload item dict into the standard result format."""
    author_name = item.get("author") or item.get("user", {}).get("name", "")
    return {
        "book_id": str(item.get("id", "")).strip(),
        "title": item.get("name", ""),
        "author": author_name,
        "intro": item.get("intro", ""),
        "words_total": item.get("words_total", 0),
        "publish_status": item.get("publish_status", ""),
        "tags": _normalize_tags(item.get("tags", [])),
        "rank": rank,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline 1: Pure BM25 (Lexical Baseline)
# ═══════════════════════════════════════════════════════════════════════

class BM25Baseline:
    """Pure lexical retrieval using BM25 (Okapi).

    No vector search, no LLM parsing, no structural filtering.
    Simply tokenises the raw user query and ranks by BM25 score.
    """

    def __init__(self, db: Database, k: int = 10):
        self.db = db
        self.k = k
        self.lexical_store = LexicalStore(db)

    def search(self, query: str) -> Dict[str, Any]:
        bm25_results = self.lexical_store.search(query, limit=self.k)

        results = []
        for rank, res in enumerate(bm25_results):
            item = res["item"]
            if not str(item.get("id", "")).strip():
                continue
            results.append(_item_to_result(item, rank + 1))

        return {"results": results}


# ═══════════════════════════════════════════════════════════════════════
# Baseline 2: Pure Dense (Naive Vector RAG)
# ═══════════════════════════════════════════════════════════════════════

class PureDenseBaseline:
    """Pure vector (dense) retrieval — NO structural filtering.

    Embeds the raw user query and performs a top-K cosine similarity
    search against Qdrant.  No metadata filters are applied, so books
    violating hard constraints (status, word count, etc.) will appear.
    """

    def __init__(self, vs: VectorStore, db: Database, k: int = 10):
        self.vs = vs
        self.db = db
        self.k = k

    def search(self, query: str) -> Dict[str, Any]:
        # Direct vector search — NO query_filter
        vector_results, _ = self.vs.search(
            query_text=query,
            limit=self.k,
            query_filter=None,
            with_payload=True,
        )

        results = []
        for rank, hit in enumerate(vector_results):
            payload = hit.get("payload") or {}
            book_id = str(payload.get("id", "")).strip()
            if not book_id:
                continue

            # Enrich with DB metadata if payload is sparse
            if not payload.get("name") or not payload.get("tags"):
                db_item = self.db.get_item(book_id)
                if db_item:
                    payload = {**db_item, **payload}

            results.append(_item_to_result(payload, rank + 1))

        return {"results": results}


# ═══════════════════════════════════════════════════════════════════════
# Baseline 3: Naive Hybrid (BM25 + Vector + RRF)
# ═══════════════════════════════════════════════════════════════════════

class NaiveHybridBaseline:
    """Standard hybrid retrieval with Reciprocal Rank Fusion.

    Runs BM25 and vector search independently, then fuses the two
    ranked lists using RRF.  NO LLM intent parsing, NO structural
    filtering, NO tag mapping.  The fusion weight is implicit through
    the RRF formula: score = 1/(k + rank_bm25) + 1/(k + rank_vec).

    Parameters
    ----------
    rrf_k : int
        The RRF constant (default 60).
    """

    def __init__(
        self,
        vs: VectorStore,
        db: Database,
        k: int = 10,
        rrf_k: int = 60,
        retrieval_pool: int = 1000,
    ):
        self.vs = vs
        self.db = db
        self.k = k
        self.rrf_k = rrf_k
        self.retrieval_pool = retrieval_pool
        self.lexical_store = LexicalStore(db)

    def search(self, query: str) -> Dict[str, Any]:
        pool = self.retrieval_pool

        # ── Channel 1: Vector search (no filter) ──
        vector_results, _ = self.vs.search(
            query_text=query,
            limit=pool,
            query_filter=None,
            with_payload=True,
        )

        vec_rank: Dict[str, int] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}
        for rank, hit in enumerate(vector_results):
            payload = hit.get("payload") or {}
            book_id = str(payload.get("id", "")).strip()
            if not book_id:
                continue
            vec_rank[book_id] = rank + 1
            payload_map[book_id] = payload

        # ── Channel 2: BM25 search ──
        bm25_results = self.lexical_store.search(query, limit=pool)

        bm25_rank: Dict[str, int] = {}
        for rank, res in enumerate(bm25_results):
            item = res["item"]
            book_id = str(item.get("id", "")).strip()
            if not book_id:
                continue
            bm25_rank[book_id] = rank + 1
            if book_id not in payload_map:
                payload_map[book_id] = item

        # ── RRF Fusion ──
        all_ids = set(vec_rank.keys()) | set(bm25_rank.keys())
        absent_rank = max(len(vec_rank), len(bm25_rank)) + 1
        k = self.rrf_k

        rrf_scores: Dict[str, float] = {}
        for book_id in all_ids:
            r_vec = vec_rank.get(book_id, absent_rank)
            r_bm25 = bm25_rank.get(book_id, absent_rank)
            rrf_scores[book_id] = 1.0 / (k + r_vec) + 1.0 / (k + r_bm25)

        # Sort by RRF score descending and take top-K
        sorted_ids = sorted(rrf_scores.keys(), key=lambda bid: rrf_scores[bid], reverse=True)

        results = []
        for rank, book_id in enumerate(sorted_ids[: self.k]):
            payload = payload_map.get(book_id, {})
            # Enrich with DB metadata
            if not payload.get("name") or not payload.get("tags"):
                db_item = self.db.get_item(book_id)
                if db_item:
                    payload = {**db_item, **payload}
            results.append(_item_to_result(payload, rank + 1))

        return {"results": results}


# ═══════════════════════════════════════════════════════════════════════
# Baseline 4: Naive Weighted Hybrid (Linear Combination)
# ═══════════════════════════════════════════════════════════════════════

class NaiveWeightedHybridBaseline:
    """Standard hybrid retrieval with Linear Weighted Combination.

    Normalizes BM25 and Vector scores to [0, 1] then computes:
    score = w_bm25 * bm25_norm + w_vec * vec_norm
    """

    def __init__(
        self,
        vs: VectorStore,
        db: Database,
        w_bm25: float,
        w_vec: float,
        k: int = 10,
        retrieval_pool: int = 1000,
    ):
        self.vs = vs
        self.db = db
        self.w_bm25 = w_bm25
        self.w_vec = w_vec
        self.k = k
        self.retrieval_pool = retrieval_pool
        self.lexical_store = LexicalStore(db)

    def search(self, query: str) -> Dict[str, Any]:
        pool = self.retrieval_pool

        # ── Channel 1: Vector search ──
        vector_results, _ = self.vs.search(
            query_text=query,
            limit=pool,
            query_filter=None,
            with_payload=True,
        )

        vec_scores: Dict[str, float] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}
        for hit in vector_results:
            payload = hit.get("payload") or {}
            book_id = str(payload.get("id", "")).strip()
            if not book_id:
                continue
            vec_scores[book_id] = float(hit.get("score", 0.0))
            payload_map[book_id] = payload

        # ── Channel 2: BM25 search ──
        bm25_results = self.lexical_store.search(query, limit=pool)

        bm25_scores: Dict[str, float] = {}
        for res in bm25_results:
            item = res["item"]
            book_id = str(item.get("id", "")).strip()
            if not book_id:
                continue
            bm25_scores[book_id] = float(res.get("score", 0.0))
            if book_id not in payload_map:
                payload_map[book_id] = item

        # ── Normalize Scores ──
        def normalize(scores_dict: Dict[str, float]) -> Dict[str, float]:
            if not scores_dict: return {}
            vals = list(scores_dict.values())
            max_val = max(vals)
            min_val = min(vals)
            if max_val == min_val:
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

        norm_vec = normalize(vec_scores)
        norm_bm25 = normalize(bm25_scores)

        # ── Linear Fusion ──
        all_ids = set(norm_vec.keys()) | set(norm_bm25.keys())
        final_scores: Dict[str, float] = {}

        for book_id in all_ids:
            s_vec = norm_vec.get(book_id, 0.0)
            s_bm25 = norm_bm25.get(book_id, 0.0)
            final_scores[book_id] = self.w_vec * s_vec + self.w_bm25 * s_bm25

        sorted_ids = sorted(final_scores.keys(), key=lambda bid: final_scores[bid], reverse=True)

        results = []
        for rank, book_id in enumerate(sorted_ids[: self.k]):
            payload = payload_map.get(book_id, {})
            if not payload.get("name") or not payload.get("tags"):
                db_item = self.db.get_item(book_id)
                if db_item:
                    payload = {**db_item, **payload}
            results.append(_item_to_result(payload, rank + 1))

        return {"results": results}


# ═══════════════════════════════════════════════════════════════════════
# Unified Run Generator
# ═══════════════════════════════════════════════════════════════════════

class BaselineRunGenerator:
    """Generate baseline experiment runs for all three strategies."""

    def __init__(self, k: int = 10):
        self.k = k
        self.db = Database()
        self.BASELINE_CONFIGS = [
            {
                "name": "baseline_bm25_only",
                "label": "Pure BM25 (Lexical Baseline)",
                "engine_cls": "bm25",
            },
            {
                "name": "baseline_dense_only",
                "label": "Pure Dense (Naive Vector RAG)",
                "engine_cls": "dense",
            },
            {
                "name": "baseline_naive_hybrid_rrf",
                "label": "Naive Hybrid (BM25 + Vector + RRF, k=60)",
                "engine_cls": "naive_hybrid",
            },
        ]
        
        # 增加 9 種線性權重配置 (BM25:Vector 權重 0.1:0.9 到 0.9:0.1)
        for w_bm25_int in range(1, 10):
            w_bm25 = w_bm25_int / 10.0
            w_vec = round(1.0 - w_bm25, 1)
            self.BASELINE_CONFIGS.append({
                "name": f"baseline_naive_hybrid_w{w_bm25:.1f}_v{w_vec:.1f}",
                "label": f"Naive Weighted Hybrid (BM25={w_bm25:.1f}, Vec={w_vec:.1f})",
                "engine_cls": "naive_weighted",
                "w_bm25": w_bm25,
                "w_vec": w_vec,
            })

    def _build_engine(self, config: Dict[str, Any], vs: Optional[VectorStore] = None):
        engine_cls = config["engine_cls"]
        if engine_cls == "bm25":
            return BM25Baseline(db=self.db, k=self.k)
        elif engine_cls == "dense":
            if vs is None:
                raise ValueError("VectorStore required for dense baseline")
            return PureDenseBaseline(vs=vs, db=self.db, k=self.k)
        elif engine_cls == "naive_hybrid":
            if vs is None:
                raise ValueError("VectorStore required for naive hybrid baseline")
            return NaiveHybridBaseline(vs=vs, db=self.db, k=self.k, rrf_k=60)
        elif engine_cls == "naive_weighted":
            if vs is None:
                raise ValueError("VectorStore required for naive weighted baseline")
            return NaiveWeightedHybridBaseline(
                vs=vs, db=self.db, k=self.k,
                w_bm25=config["w_bm25"], w_vec=config["w_vec"]
            )
        else:
            raise ValueError(f"Unknown engine class: {engine_cls}")

    def generate_run(
        self,
        queries_config: List[Dict[str, Any]],
        config: Dict[str, Any],
        output_dir: Path,
        run_suffix: str = "",
    ) -> None:
        engine_name = config["name"]
        engine_cls = config["engine_cls"]
        label = config["label"]

        print(
            f"\n[Baseline] Starting: {label}"
            f" (engine_name={engine_name}, suffix={run_suffix or 'none'})"
        )

        # Only create VectorStore when needed (avoids overhead for BM25-only)
        vs = None
        if engine_cls in ("dense", "naive_hybrid", "naive_weighted"):
            vs = VectorStore(collection_name="novels")

        engine = self._build_engine(config, vs)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{engine_name}{run_suffix}.json"

        run_data: List[Dict[str, Any]] = []
        processed_query_ids: set = set()

        # Resume support
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                for item in existing_data:
                    if "error" not in item:
                        run_data.append(item)
                        processed_query_ids.add(item.get("query_id"))
                print(
                    f"   Loaded {len(processed_query_ids)} completed queries from "
                    f"existing file. Resuming..."
                )
            except Exception as exc:
                print(f"   ⚠ Could not load existing file: {exc}")

        try:
            pending = [q for q in queries_config if q["id"] not in processed_query_ids]
            for q_conf in pending:
                q_id = q_conf["id"]
                query = q_conf["query"]
                print(f"   - [{engine_cls}] Processing query: {query[:40]}...")

                try:
                    response = engine.search(query)
                    results = response.get("results", [])

                    run_data.append({
                        "query_id": q_id,
                        "query": query,
                        "model_id": None,
                        "parser_variant": None,
                        "execution_metadata": {},
                        "parse_metadata": {},
                        "parsed_criteria": [],
                        "tag_intent": None,
                        "reference_tags": [],
                        "results": results,
                    })
                except Exception as query_err:
                    print(f"     ⚠ Error processing query {q_id}: {query_err}")
                    run_data.append({
                        "query_id": q_id,
                        "query": query,
                        "model_id": None,
                        "parser_variant": None,
                        "execution_metadata": {},
                        "parse_metadata": {},
                        "parsed_criteria": [],
                        "tag_intent": None,
                        "reference_tags": [],
                        "results": [],
                        "error": str(query_err),
                    })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(run_data, f, ensure_ascii=False, indent=2)

            print(f"[{engine_name}] Run complete! Saved to {output_path}")
        finally:
            if vs is not None:
                vs.client.close()

    def generate_all_baselines(
        self,
        queries_config: List[Dict[str, Any]],
        output_dir: Path,
        run_suffix: str = "",
    ) -> None:
        """Run all three baseline experiments sequentially."""
        for config in self.BASELINE_CONFIGS:
            self.generate_run(
                queries_config=queries_config,
                config=config,
                output_dir=output_dir,
                run_suffix=run_suffix,
            )


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate baseline experiment runs (BM25 / Dense / Naive Hybrid)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent trials to run per baseline",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/runs",
        help="Directory for generated run files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return per query",
    )
    args = parser.parse_args()

    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        raise SystemExit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        sample_queries = json.load(f)

    repeats = max(1, args.repeats)
    output_root = Path(args.experiment_dir)
    batch_name = datetime.now().strftime("baseline_batch_%Y%m%d_%H%M%S")
    output_folder = output_root / batch_name
    print(f"Baseline batch output directory: {output_folder}")

    generator = BaselineRunGenerator(k=args.k)

    for repeat_index in range(1, repeats + 1):
        run_suffix = f"_run{repeat_index:02d}" if repeats > 1 else ""
        print(f"\n=== Trial {repeat_index}/{repeats} ===")

        generator.generate_all_baselines(
            queries_config=sample_queries,
            output_dir=output_folder,
            run_suffix=run_suffix,
        )

    print("\n✅ All baseline experiments finished!")
