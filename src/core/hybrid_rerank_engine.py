import time
from typing import Any, Dict, Optional

from src.core.database import Database
from src.core.engine import HybridEngine
from src.core.single_prompt_engine import SinglePromptLLMEngine
from src.core.vector_store import VectorStore


class HybridRerankEngine:
    """Two-stage engine: Hybrid Engine (Recall) -> Single Prompt LLM (Rerank Top-K)."""

    ENGINE_NAME = "HybridRerankEngine"
    PARSER_VARIANT = "hybrid_rerank_catalog"

    def __init__(
        self,
        db: Optional[Database] = None,
        vs: Optional[VectorStore] = None,
        semantic_weight: Optional[float] = None,
        attribute_weight: Optional[float] = None,
        allowed_book_ids: Optional[set[str]] = None,
        rerank_top_k: int = 100,
    ) -> None:
        self.db = db if db is not None else Database()
        self.vs = vs if vs is not None else VectorStore(collection_name="novels")
        self.allowed_book_ids = allowed_book_ids
        self.semantic_weight = semantic_weight
        self.attribute_weight = attribute_weight
        self.rerank_top_k = rerank_top_k
        self.hybrid_engine = HybridEngine(
            db=self.db,
            vs=self.vs,
            semantic_weight=self.semantic_weight,
            attribute_weight=self.attribute_weight,
            allowed_book_ids=self.allowed_book_ids,
        )

    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()

        # Stage 1: Retrieval via HybridEngine
        hybrid_response = await self.hybrid_engine.search(
            user_query=user_query,
            limit=self.rerank_top_k,
            model_id=model_id,
            explain=False,
            cache_namespace=cache_namespace,
        )

        hybrid_results = hybrid_response.get("results", [])
        if not hybrid_results:
            hybrid_response["engine"] = self.ENGINE_NAME
            hybrid_response["parser_variant"] = self.PARSER_VARIANT
            return hybrid_response

        # Extract top K ids from the hybrid results
        top_k_ids = set()
        for res in hybrid_results:
            book_id = str(res.get("item", {}).get("id", "")).strip()
            if book_id:
                top_k_ids.add(book_id)

        # Stage 2: Rerank via SinglePromptLLMEngine using the narrowed down candidates
        llm_engine = SinglePromptLLMEngine(
            db=self.db,
            include_intro=True,
            intro_char_limit=None,
            allowed_book_ids=top_k_ids,
            mode="rerank",
        )
        
        llm_response = await llm_engine.search(
            user_query=user_query,
            limit=limit,
            model_id=model_id,
            explain=explain,
            cache_namespace=cache_namespace,
        )

        # Merge metadata
        llm_response["engine"] = self.ENGINE_NAME
        llm_response["parser_variant"] = self.PARSER_VARIANT
        if "parse_metadata" not in llm_response:
            llm_response["parse_metadata"] = {}
        llm_response["parse_metadata"]["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        llm_response["parse_metadata"]["hybrid_retrieved_count"] = len(hybrid_results)
        
        # Preserve original parsed criteria and tag intent from Hybrid engine for tracing
        llm_response["parsed_criteria"] = hybrid_response.get("parsed_criteria", [])
        llm_response["tag_intent"] = hybrid_response.get("tag_intent", {})
        llm_response["search_terms"] = hybrid_response.get("search_terms", "")
        llm_response["generated_keywords"] = hybrid_response.get("generated_keywords", [])

        # The results from llm_response are already ranked and formatted by SinglePromptLLMEngine
        return llm_response
