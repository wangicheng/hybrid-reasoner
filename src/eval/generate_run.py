import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.database import Database
from src.core.engine import HybridEngine
from src.core.vector_store import VectorStore


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class RunGenerator:
    """
    Generate experiment runs for the tag-description-context ablation.

    The generator can switch between:
    - using tag descriptions in the LLM prompt
    - embedding or skipping LLM-generated keywords

    This script keeps the run file format compatible with merge_and_pool.py:
    a JSON array of per-query result objects.
    """

    def __init__(
        self,
        k_per_engine: int = 10,
        use_tag_descriptions: bool = True,
        embed_generated_keywords: bool = True,
        model_id: Optional[str] = None,
    ) -> None:
        self.k = k_per_engine
        self.use_tag_descriptions = use_tag_descriptions
        self.embed_generated_keywords = embed_generated_keywords
        self.model_id = model_id
        self.db = Database()

    async def _search_once(
        self,
        engine: HybridEngine,
        query: str,
    ) -> Dict[str, Any]:
        return await engine.search(
            query,
            limit=self.k,
            model_id=self.model_id,
            explain=False,
        )

    def generate_run(
        self,
        queries_config: List[Dict[str, Any]],
        engine_name: str,
        output_dir: Path,
        semantic_weight: float = 0.3,
        attribute_weight: float = 0.7,
        use_tag_descriptions: Optional[bool] = None,
        embed_generated_keywords: Optional[bool] = None,
    ) -> None:
        resolved_use_tag_descriptions = (
            self.use_tag_descriptions
            if use_tag_descriptions is None
            else use_tag_descriptions
        )
        resolved_embed_generated_keywords = (
            self.embed_generated_keywords
            if embed_generated_keywords is None
            else embed_generated_keywords
        )

        print(
            f"\n[Batch] Starting Experiment: {engine_name} "
            f"(W1: {semantic_weight}, W2: {attribute_weight}, "
            f"tag_descriptions={resolved_use_tag_descriptions}, "
            f"embed_generated_keywords={resolved_embed_generated_keywords})"
        )

        vs = VectorStore(collection_name="novels")
        engine = HybridEngine(
            db=self.db,
            vs=vs,
            semantic_weight=semantic_weight,
            attribute_weight=attribute_weight,
            use_tag_descriptions=resolved_use_tag_descriptions,
            embed_generated_keywords=resolved_embed_generated_keywords,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{engine_name}.json"

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
            for q_conf in queries_config:
                q_id = q_conf["id"]
                query = q_conf["query"]

                if q_id in processed_query_ids:
                    print(f"   - Skipping query: {q_id} (already completed)")
                    continue

                print(f"   - Processing query: {query[:30]}...")

                try:
                    response = asyncio.run(self._search_once(engine, query))
                    results = response.get("results", [])
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

                    run_data.append(
                        {
                            "query_id": q_id,
                            "query": query,
                            "results": extracted_results,
                        }
                    )
                except Exception as query_err:
                    print(f"     ?? Error processing query {q_id}: {query_err}")
                    run_data.append(
                        {
                            "query_id": q_id,
                            "query": query,
                            "results": [],
                            "error": str(query_err),
                        }
                    )

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

    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        print(f"Error: {queries_path} not found!")
        raise SystemExit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        sample_queries = json.load(f)

    EXPERIMENTS = [
        {
            "name": "exp_tagctx_3-7",
            "w1": 0.3,
            "w2": 0.7,
            "use_tag_descriptions": True,
            "embed_generated_keywords": True,
        },
        {
            "name": "exp_tagctx_5-5",
            "w1": 0.5,
            "w2": 0.5,
            "use_tag_descriptions": True,
            "embed_generated_keywords": True,
        },
        {
            "name": "exp_tagctx_7-3",
            "w1": 0.7,
            "w2": 0.3,
            "use_tag_descriptions": True,
            "embed_generated_keywords": True,
        },
        {
            "name": "exp_tagctx_noembed_3-7",
            "w1": 0.3,
            "w2": 0.7,
            "use_tag_descriptions": True,
            "embed_generated_keywords": False,
        },
        {
            "name": "exp_tagctx_noembed_5-5",
            "w1": 0.5,
            "w2": 0.5,
            "use_tag_descriptions": True,
            "embed_generated_keywords": False,
        },
        {
            "name": "exp_tagctx_noembed_7-3",
            "w1": 0.7,
            "w2": 0.3,
            "use_tag_descriptions": True,
            "embed_generated_keywords": False,
        },
    ]

    generator = RunGenerator(k_per_engine=10)
    output_folder = Path("data/experiments/runs")

    for exp in EXPERIMENTS:
        try:
            generator.generate_run(
                queries_config=sample_queries,
                engine_name=exp["name"],
                output_dir=output_folder,
                semantic_weight=exp["w1"],
                attribute_weight=exp["w2"],
                use_tag_descriptions=exp["use_tag_descriptions"],
                embed_generated_keywords=exp["embed_generated_keywords"],
            )
        except Exception as exc:
            print(f"Failed experiment {exp['name']}: {exc}")

    print("\nTag-description-context experiments finished!")
