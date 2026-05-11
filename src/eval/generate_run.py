import asyncio
import concurrent.futures
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.database import Database
from src.core.engine import HybridEngine
from src.core.api_utils import _is_retryable
from src.core.llm import DEFAULT_PARSER_VARIANT
from src.core.model_catalog import normalize_model_id
from src.core.vector_store import VectorStore


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class RunGenerator:
    """
    Generate experiment runs for the fixed production retrieval path.
    """

    def __init__(
        self,
        k_per_engine: int = 10,
        model_id: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> None:
        self.k = k_per_engine
        self.model_id = model_id
        self.rerank = rerank
        self.db = Database()

    async def _search_once(
        self,
        engine: HybridEngine,
        query: str,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await engine.search(
            query,
            limit=self.k,
            model_id=self.model_id,
            explain=False,
            cache_namespace=cache_namespace,
        )

    @staticmethod
    def _retry_delay_seconds(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Use a fixed retry interval for retryable query failures."""
        _ = attempt, max_delay
        return base_delay

    def _search_with_retry(
        self,
        engine: HybridEngine,
        query: str,
        q_id: str,
        cache_namespace: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run a single query until it succeeds.

        Retryable socket / connectivity failures are retried forever with
        a fixed interval. Non-retryable exceptions still bubble up to the
        caller so they can be recorded as a real query failure.
        """
        attempt = 0
        while True:
            try:
                response = asyncio.run(self._search_once(engine, query, cache_namespace=cache_namespace))
                return response, {
                    "query_attempts": attempt + 1,
                    "query_retry_count": attempt,
                    "first_attempt_success": attempt == 0,
                }
            except Exception as exc:
                if not _is_retryable(exc):
                    setattr(
                        exc,
                        "query_execution_metadata",
                        {
                            "query_attempts": attempt + 1,
                            "query_retry_count": attempt,
                            "first_attempt_success": attempt == 0,
                        },
                    )
                    raise

                attempt += 1
                delay = self._retry_delay_seconds(attempt)
                print(
                    f"     [Retry] Query {q_id} hit a retryable error: {exc}. "
                    f"Retrying in {delay:.1f}s (attempt {attempt})..."
                )
                time.sleep(delay)

    def _process_single_query(self, q_conf: Dict[str, Any], engine: HybridEngine, engine_name: str, run_suffix: str) -> Dict[str, Any]:
        q_id = q_conf["id"]
        query = q_conf["query"]

        print(f"   - Processing query: {query[:30]}...")

        try:
            response, execution_metadata = self._search_with_retry(
                engine,
                query,
                q_id,
                cache_namespace=run_suffix or engine_name,
            )
            results = response.get("results", [])
            parsed_criteria = response.get("parsed_criteria", [])
            parse_metadata = response.get("parse_metadata", {})
            extracted_results = []

            for rank, res in enumerate(results):
                item = res.get("item", {})
                b_id = str(item.get("id", "")).strip()
                if not b_id:
                    continue

                author_name = item.get("author") or item.get("user", {}).get("name", "")
                extracted_results.append(
                    {
                        "book_id": b_id,
                        "title": item.get("name", ""),
                        "author": author_name,
                        "intro": item.get("intro", ""),
                        "words_total": item.get("words_total", 0),
                        "publish_status": item.get("publish_status", ""),
                        "tags": item.get("tags", []),
                        "rank": rank + 1,
                    }
                )

            return {
                "query_id": q_id,
                "query": query,
                "model_id": normalize_model_id(self.model_id),
                "parser_variant": DEFAULT_PARSER_VARIANT,
                "execution_metadata": execution_metadata,
                "parse_metadata": parse_metadata,
                "parsed_criteria": parsed_criteria,
                "tag_intent": response.get("tag_intent"),
                "reference_tags": response.get("reference_tags", []),
                "results": extracted_results,
            }
        except Exception as query_err:
            print(f"     ?? Error processing query {q_id}: {query_err}")
            return {
                "query_id": q_id,
                "query": query,
                "model_id": normalize_model_id(self.model_id),
                "parser_variant": DEFAULT_PARSER_VARIANT,
                "execution_metadata": getattr(query_err, "query_execution_metadata", {}),
                "parsed_criteria": [],
                "parse_metadata": getattr(query_err, "parser_metadata", {}),
                "results": [],
                "error": str(query_err),
            }

    def generate_run(
        self,
        queries_config: List[Dict[str, Any]],
        engine_name: str,
        output_dir: Path,
        semantic_weight: float = 0.3,
        attribute_weight: float = 0.7,
        run_suffix: str = "",
        enable_bm25: bool = False,
        bm25_weight: float = 0.3,
        bm25_bonus_max: Optional[float] = None,
        bm25_fusion_mode: Optional[str] = None,
        fusion_strategy: Optional[str] = None,
        rrf_k: int = 60,
        # Dynamic routing parameters (auto mode only)
        routing_tag_threshold: int = 1,
        routing_weighted_ws: float = 0.35,
        routing_weighted_wa: float = 0.65,
        routing_weighted_bm25: bool = True,
        routing_rrf_bm25: bool = False,
    ) -> None:
        fusion_label = fusion_strategy or "weighted"
        print(
            f"\n[Batch] Starting Experiment: {engine_name} "
            f"(fusion={fusion_label}, W1: {semantic_weight}, W2: {attribute_weight}, "
            f"BM25 Enabled: {enable_bm25}, BM25 Weight: {bm25_weight} (recall-only), "
            f"BM25 Bonus Max: {bm25_bonus_max}, "
            f"rrf_k={rrf_k}, "
            f"engine=HybridEngine, "
            "fixed retrieval path, "
            f"model={normalize_model_id(self.model_id)}, "
            f"parser_variant={DEFAULT_PARSER_VARIANT}, "
            f"run_suffix={run_suffix or 'none'})"
        )

        vs = VectorStore(collection_name="novels")
        engine = HybridEngine(
            db=self.db,
            vs=vs,
            semantic_weight=semantic_weight,
            attribute_weight=attribute_weight,
            enable_bm25=enable_bm25,
            bm25_weight=bm25_weight,
            bm25_bonus_max=bm25_bonus_max,
            bm25_fusion_mode=bm25_fusion_mode,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
            routing_tag_threshold=routing_tag_threshold,
            routing_weighted_ws=routing_weighted_ws,
            routing_weighted_wa=routing_weighted_wa,
            routing_weighted_bm25=routing_weighted_bm25,
            routing_rrf_bm25=routing_rrf_bm25,
            rerank=self.rerank,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{engine_name}{run_suffix}.json"

        run_data: List[Dict[str, Any]] = []
        processed_query_ids = set()

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
                print(f"   ?? Could not load existing file: {exc}")

        try:
            pending_queries = [q for q in queries_config if q["id"] not in processed_query_ids]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._process_single_query, q_conf, engine, engine_name, run_suffix): q_conf
                    for q_conf in pending_queries
                }
                
                for future in concurrent.futures.as_completed(futures):
                    run_data.append(future.result())

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(run_data, f, ensure_ascii=False, indent=2)

            print(f"[{engine_name}] Run complete! Saved to {output_path}")
        finally:
            vs.client.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment runs")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent trials to run per experiment",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/runs",
        help="Directory for generated run files",
    )
    parser.add_argument(
        "--disable-bm25",
        action="store_true",
        help="Disable BM25 retrieval in HybridEngine",
    )
    parser.add_argument(
        "--bm25-mode",
        type=str,
        choices=["compare", "on", "off"],
        default="compare",
        help="Run both BM25 OFF/ON in one batch, or force a single variant",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.3,
        help="Legacy BM25 recall setting retained for compatibility",
    )
    args = parser.parse_args()

    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        raise SystemExit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        sample_queries = json.load(f)

    # Allow experiments to run BM25 ON/OFF in a single batch.
    if args.bm25_mode == "compare":
        experiments = [
            {
                "name": "gemma4_default_parser_bm25_off",
                "model_id": "gemma-4-31b-it",
                "enable_bm25": False,
            },
            {
                "name": "gemma4_default_parser_bm25_on",
                "model_id": "gemma-4-31b-it",
                "enable_bm25": True,
                "bm25_weight": 0.1,
            },
        ]
    else:
        enable_bm25 = args.bm25_mode == "on"
        experiments = [
            {
                "name": f"gemma4_default_parser_bm25_{args.bm25_mode}",
                "model_id": "gemma-4-31b-it",
                "enable_bm25": enable_bm25,
                "bm25_weight": 0.1 if enable_bm25 else args.bm25_weight,
            }
        ]

    repeats = max(1, args.repeats)
    output_root = Path(args.experiment_dir)
    batch_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    output_folder = output_root / batch_name
    print(f"Batch output directory: {output_folder}")

    for repeat_index in range(1, repeats + 1):
        run_suffix = f"_run{repeat_index:02d}" if repeats > 1 else ""
        print(f"\n=== Trial {repeat_index}/{repeats} ===")
        for exp in experiments:
            model_id = normalize_model_id(exp.get("model_id"))
            generator = RunGenerator(
                k_per_engine=10,
                model_id=model_id,
                rerank=exp.get("rerank", None),
            )
            enable_bm25 = exp.get("enable_bm25", not args.disable_bm25)
            bm25_weight = exp.get("bm25_weight", args.bm25_weight)
            try:
                generator.generate_run(
                    queries_config=sample_queries,
                    engine_name=exp["name"],
                    output_dir=output_folder,
                    semantic_weight=0.4,
                    attribute_weight=0.6,
                    run_suffix=run_suffix,
                    enable_bm25=enable_bm25,
                    bm25_weight=bm25_weight,
                )
            except Exception as exc:
                print(
                    f"Failed experiment {exp['name']} on model {model_id} "
                    f"(parser_variant={DEFAULT_PARSER_VARIANT}, {run_suffix or 'single'}): {exc}"
                )

    print("\nFixed-path experiments finished!")
