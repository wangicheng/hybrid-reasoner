import json
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.config import settings


class VectorStore:
    def __init__(self, collection_name: str = "items"):
        if settings.QDRANT_PATH == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            path = Path(settings.QDRANT_PATH)
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path.resolve()))

        self.collection_name = collection_name
        from src.core.api_utils import get_current_api_key

        self._current_api_key = get_current_api_key()
        self.genai_client = genai.Client(api_key=self._current_api_key)
        self.embedding_model = "gemini-embedding-001"
        self._ensure_collection()
        self._ensure_payload_indexes()

    def _update_api_key_on_rate_limit(self) -> None:
        from src.core.api_utils import get_api_key_rotator

        rotator = get_api_key_rotator()
        new_key = rotator.on_rate_limit_error()
        self._current_api_key = new_key
        self.genai_client = genai.Client(api_key=new_key)
        print(f"[VectorStore] API key rotated. Current index: {rotator.current_index}")

    def _ensure_collection(self) -> None:
        self._ensure_named_collection(self.collection_name)

    def _ensure_named_collection(self, collection_name: str) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == collection_name for collection in collections):
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=3072,
                distance=rest.Distance.COSINE,
            ),
        )

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes on the main collection for metadata pre-filtering.

        Indexes are idempotent — re-creating an existing index is a no-op in Qdrant.
        Only applies to the primary collection (self.collection_name), not tag
        collections.
        """
        index_specs = [
            ("publish_status", rest.PayloadSchemaType.KEYWORD),
            ("words_total", rest.PayloadSchemaType.INTEGER),
            ("tags", rest.PayloadSchemaType.KEYWORD),
        ]
        for field_name, field_schema in index_specs:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                # Index may already exist or collection may not support it;
                # either way this is non-fatal.
                pass

    @staticmethod
    def build_metadata_filter(
        hard_constraints: Dict[str, Any],
    ) -> Optional[rest.Filter]:
        """Build a Qdrant metadata filter from extracted hard constraints.

        Translates the dict produced by ``HybridEngine._extract_hard_constraints()``
        into a ``rest.Filter`` that Qdrant can apply server-side during ANN search.

        Implements graceful degradation through tolerant filtering:
        - Word count range: applies ±20% tolerance buffer
        - Status & words: uses OR logic to soften constraints (疑罪從無)
          By using `should` instead of `must`, items with missing metadata
          are automatically included (benefit of the doubt).
        
        Returns ``None`` when no filterable constraints are present, so the
        caller can safely pass the result straight to ``query_filter``.
        """
        must_conditions: List[rest.Condition] = []
        should_conditions: List[rest.Condition] = []
        must_not_conditions: List[rest.Condition] = []

        # 1. Status filter with tolerance: move to `should` for soft matching
        #    This allows documents with missing status to pass through (疑罪從無)
        status_filter = hard_constraints.get("status_filter")
        if status_filter:
            status_values = []
            if status_filter == "completed":
                status_values = ["completed", "已完結", "完結"]
            elif status_filter == "ongoing":
                status_values = ["ongoing", "連載中", "連載"]
            
            if status_values:
                # Use `should` instead of `must` to allow missing status fields
                should_conditions.append(
                    rest.FieldCondition(
                        key="publish_status",
                        match=rest.MatchAny(any=status_values),
                    )
                )

        # 2. Word-count range with 20% tolerance + soft matching
        #    ±20% complacency: words_min → 0.8×, words_max → 1.2×
        #    Soft matching: items with missing words_total pass through (疑罪從無)
        words_min = hard_constraints.get("words_min")
        words_max = hard_constraints.get("words_max")
        if words_min is not None or words_max is not None:
            # Apply 20% tolerance buffer
            relaxed_min = int(words_min * 0.8) if words_min is not None else None
            relaxed_max = int(words_max * 1.2) if words_max is not None else None
            
            range_kwargs: Dict[str, Any] = {}
            if relaxed_min is not None:
                range_kwargs["gte"] = relaxed_min
            if relaxed_max is not None:
                range_kwargs["lte"] = relaxed_max
            
            # Use `should` instead of `must` to allow missing words_total fields
            should_conditions.append(
                rest.FieldCondition(
                    key="words_total",
                    range=rest.Range(**range_kwargs),
                )
            )

        # 3. Negative tags (must NOT contain) ────────────────────────────
        #    Negative constraints stay as hard filters (no tolerance)
        negative_tags = hard_constraints.get("negative_tag_terms", [])
        for neg_tag in negative_tags:
            if not neg_tag:
                continue
            must_not_conditions.append(
                rest.FieldCondition(
                    key="tags",
                    match=rest.MatchValue(value=neg_tag),
                )
            )

        # 4. Author filter — Qdrant substring match needs a full-text index
        #    which we don't have yet; kept as post-filter in engine.py.

        if not should_conditions and not must_not_conditions:
            return None

        # Build filter: should_conditions (soft OR), must_not_conditions (hard NOT)
        # By using only `should` (not `must`), missing metadata items naturally pass through
        return rest.Filter(
            should=should_conditions if should_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.client.get_collections().collections
        except Exception as exc:
            print(f"[VectorStore] Failed to inspect collections: {exc}")
            return False

        return any(collection.name == collection_name for collection in collections)

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> List[str]:
        if isinstance(raw_tags, str):
            try:
                raw_tags = json.loads(raw_tags)
            except Exception:
                return []
        if isinstance(raw_tags, (list, tuple)):
            normalized: List[str] = []
            seen = set()
            for tag in raw_tags:
                value = str(tag).strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                normalized.append(value)
            return normalized
        return []

    def _scroll_collection_tags(self, collection_name: str) -> List[str]:
        if not self.collection_exists(collection_name):
            return []

        tags: List[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                tag = payload.get("tag")
                if tag:
                    tags.append(str(tag).strip())

            if offset is None:
                break

        return tags

    def sync_tag_collection(
        self,
        tags: Any,
        tag_descriptions: Optional[Dict[str, str]] = None,
        collection_name: str = "novel_tags",
    ) -> None:
        normalized_tags = self._normalize_tags(tags)
        if not normalized_tags:
            raise ValueError("Tag collection sync requires at least one tag.")

        # 1. Sync Symmetric Collection (novel_tags)
        self._sync_single_tag_collection(
            normalized_tags,
            [f"這部作品的類型偏向{tag}" for tag in normalized_tags],
            collection_name
        )

        # 2. Sync Asymmetric/Description Collection (novel_tags_desc)
        if tag_descriptions:
            desc_texts = []
            valid_desc_tags = []
            for tag in normalized_tags:
                desc = tag_descriptions.get(tag)
                if desc:
                    desc_texts.append(f"{tag}：{desc}")
                    valid_desc_tags.append(tag)
            
            if desc_texts:
                self._sync_single_tag_collection(
                    valid_desc_tags,
                    desc_texts,
                    f"{collection_name}_desc"
                )

    def _sync_single_tag_collection(
        self,
        tags: List[str],
        texts_to_embed: List[str],
        collection_name: str,
    ) -> None:
        current_tags = self._scroll_collection_tags(collection_name)
        if (
            current_tags
            and len(current_tags) == len(tags)
            and set(current_tags) == set(tags)
        ):
            return

        if self.collection_exists(collection_name):
            print(
                f"[VectorStore] Rebuilding '{collection_name}' with "
                f"{len(tags)} allowed tags."
            )
            self.client.delete_collection(collection_name=collection_name)
        else:
            print(
                f"[VectorStore] Creating '{collection_name}' with "
                f"{len(tags)} allowed tags."
            )

        self._ensure_named_collection(collection_name)

        tag_vectors = self._embed_with_retry(
            texts_to_embed,
            task_type="RETRIEVAL_DOCUMENT",
        )
        points = []
        for tag, vector in zip(tags, tag_vectors):
            points.append(
                rest.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{tag}")),
                    vector=vector,
                    payload={"tag": tag},
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
        print(f"[VectorStore] '{collection_name}' synced with {len(points)} tags.")

    def _embed_with_retry(
        self,
        text: Any,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> Any:
        from src.core.api_utils import (
            _is_retryable,
            get_api_key_rotator,
            get_rate_limiter,
            is_rate_limit_error,
        )
        import time

        attempt = 0
        rotator = get_api_key_rotator()
        num_keys = len(rotator.api_keys)
        # We want to try all keys, plus some retries for transient errors.
        # At minimum, we should attempt num_keys times if it's a rate limit.
        max_attempts = max(10, num_keys + 2)
        is_list = isinstance(text, list)

        while attempt < max_attempts:
            try:
                get_rate_limiter().wait(self._current_api_key)
                response = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                if is_list:
                    return [list(embedding.values) for embedding in response.embeddings]
                return list(response.embeddings[0].values)
            except Exception as exc:
                attempt += 1
                error_text = str(exc)
                is_quota_error = is_rate_limit_error(exc)
                if is_quota_error and attempt < max_attempts:
                    # If we have multiple keys, try to rotate. 
                    # We should rotate at least until we've tried every key once.
                    if num_keys > 1:
                        print(
                            f"[VectorStore] Embedding failed (Quota). "
                            f"Rotating API key (attempt {attempt}/{max_attempts})."
                        )
                        self._update_api_key_on_rate_limit()
                        # Optional: small sleep to avoid immediate back-to-back 429s if keys are related
                        time.sleep(1)
                        continue
                    else:
                        # Single key: must wait.
                        delay = 10 * attempt
                        print(f"[VectorStore] Quota reached for single key. Waiting {delay}s...")
                        time.sleep(delay)
                        continue

                if _is_retryable(exc) and attempt < max_attempts:
                    backoff_seconds = 2 * attempt
                    print(
                        f"[VectorStore] Embedding failed ({error_text[:80]}). "
                        f"Retrying in {backoff_seconds}s, attempt {attempt}."
                    )
                    time.sleep(backoff_seconds)
                    continue
                raise
        
        raise RuntimeError("Max embed retries exceeded")

    def search(
        self,
        query_text: str,
        limit: int = 10000,
        query_filter: Optional[rest.Filter] = None,
        with_payload: bool = True,
        collection_name: Optional[str] = None,
        allowed_point_ids: Optional[set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        vector = self._embed_with_retry(query_text, task_type="RETRIEVAL_QUERY")
        target_collection = collection_name or self.collection_name

        if allowed_point_ids:
            subset_filter = rest.Filter(
                must=[
                    rest.HasIdCondition(has_id=list(allowed_point_ids))
                ]
            )
            if query_filter:
                # Merge filters
                query_filter = rest.Filter(
                    must=(query_filter.must or []) + subset_filter.must,
                    should=query_filter.should,
                    must_not=query_filter.must_not,
                )
            else:
                query_filter = subset_filter

        response = self.client.query_points(
            collection_name=target_collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=with_payload,
        )

        results = [
            {"id": point.id, "score": point.score, "payload": point.payload}
            for point in response.points
        ]
        return results, vector

    def search_individual(
        self,
        query_list: List[str],
        limit: int = 10000,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query_list:
            return []

        target_collection = collection_name or self.collection_name
        if not self.collection_exists(target_collection):
            raise RuntimeError(
                f"Qdrant collection '{target_collection}' is missing."
            )

        query_vectors = self._embed_with_retry(query_list, task_type="RETRIEVAL_QUERY")

        aggregated_scores: Dict[Any, float] = {}
        payload_cache: Dict[Any, Any] = {}
        for vector in query_vectors:
            response = self.client.query_points(
                collection_name=target_collection,
                query=vector,
                limit=limit * 2,
                with_payload=True,
            )
            for point in response.points:
                point_id = point.id
                score = float(point.score)
                if point_id not in aggregated_scores or score > aggregated_scores[point_id]:
                    aggregated_scores[point_id] = score
                    if point.payload:
                        payload_cache[point_id] = point.payload

        sorted_results = sorted(
            aggregated_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        return [
            {"id": point_id, "score": score, "payload": payload_cache.get(point_id, {})}
            for point_id, score in sorted_results
        ]

    def search_tags(
        self,
        query_text: str,
        limit: int = 10000,
        similarity_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        if not self.collection_exists("novel_tags"):
            raise RuntimeError("Qdrant collection 'novel_tags' is missing.")

        query_vector = self._embed_with_retry(query_text, task_type="RETRIEVAL_QUERY")
        
        has_desc_collection = self.collection_exists("novel_tags_desc")
        
        # 1. Search Symmetric
        sym_scores: Dict[str, float] = {}
        response_sym = self.client.query_points(
            collection_name="novel_tags",
            query=query_vector,
            limit=limit,
            score_threshold=similarity_threshold,
            with_payload=True,
        )
        for point in response_sym.points:
            if point.payload and "tag" in point.payload:
                sym_scores[point.payload["tag"]] = float(point.score)

        # 2. Search Description
        desc_scores: Dict[str, float] = {}
        if has_desc_collection:
            response_desc = self.client.query_points(
                collection_name="novel_tags_desc",
                query=query_vector,
                limit=limit,
                score_threshold=similarity_threshold,
                with_payload=True,
            )
            for point in response_desc.points:
                if point.payload and "tag" in point.payload:
                    desc_scores[point.payload["tag"]] = float(point.score)

        # 3. Hybrid Fusion
        results = []
        all_tags = set(sym_scores.keys()) | set(desc_scores.keys())
        for tag in all_tags:
            s_sym = sym_scores.get(tag, 0.0)
            s_desc = desc_scores.get(tag, 0.0)
            
            if has_desc_collection:
                combined_score = (s_sym * 0.7) + (s_desc * 0.3)
            else:
                combined_score = s_sym
            
            if combined_score >= similarity_threshold:
                results.append({"tag": tag, "score": combined_score})
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def batch_map_tags(
        self,
        target_tags: List[str],
        similarity_threshold: float = 0.6,
        limit_per_tag: int = 10000,
    ) -> List[Dict[str, float]]:
        if not target_tags:
            return []

        if not self.collection_exists("novel_tags"):
            raise RuntimeError("Qdrant collection 'novel_tags' is missing.")

        try:
            # We use the same query vector for both searches as it's the same template: 
            # "這部作品的類型偏向{tag}"
            query_vectors = self._embed_with_retry(
                [f"這部作品的類型偏向{tag}" for tag in target_tags],
                task_type="RETRIEVAL_QUERY",
            )
            
            has_desc_collection = self.collection_exists("novel_tags_desc")
            results: List[Dict[str, float]] = []
            
            for vector in query_vectors:
                # 1. Get scores from Symmetric collection (weight 0.7)
                sym_scores: Dict[str, float] = {}
                response_sym = self.client.query_points(
                    collection_name="novel_tags",
                    query=vector,
                    limit=limit_per_tag,
                    score_threshold=similarity_threshold,
                    with_payload=True,
                )
                for point in response_sym.points:
                    if point.payload and "tag" in point.payload:
                        sym_scores[point.payload["tag"]] = float(point.score)

                # 2. Get scores from Asymmetric/Description collection (weight 0.3)
                desc_scores: Dict[str, float] = {}
                if has_desc_collection:
                    response_desc = self.client.query_points(
                        collection_name="novel_tags_desc",
                        query=vector,
                        limit=limit_per_tag,
                        score_threshold=similarity_threshold,
                        with_payload=True,
                    )
                    for point in response_desc.points:
                        if point.payload and "tag" in point.payload:
                            desc_scores[point.payload["tag"]] = float(point.score)

                # 3. Hybrid Weight Fusion
                # Formula: Score = (Symmetric Score * 0.7) + (Description Score * 0.3)
                mapping: Dict[str, float] = {}
                all_tags = set(sym_scores.keys()) | set(desc_scores.keys())
                
                for tag in all_tags:
                    s_sym = sym_scores.get(tag, 0.0)
                    s_desc = desc_scores.get(tag, 0.0)
                    
                    if has_desc_collection:
                        # If we have both, apply 0.7 / 0.3
                        # If one is missing from a search result, we treat it as 0.0 (or we could use sym only)
                        # The report implies both contribute.
                        combined_score = (s_sym * 0.7) + (s_desc * 0.3)
                    else:
                        combined_score = s_sym
                    
                    if combined_score >= similarity_threshold:
                        mapping[tag] = combined_score
                
                results.append(mapping)
            return results
        except Exception as exc:
            print(f"[VectorStore] Batch tag mapping failed: {exc}")
            return []


    def add_items(self, items: List[Dict[str, Any]]) -> None:
        import time

        valid_items = []
        texts_to_embed = []
        for item in items:
            parts = [
                f"Title: {item.get('name', '')}",
                f"Intro: {item.get('intro', '')}",
            ]
            text = "\n".join(part for part in parts if part.strip()).strip()
            if not text:
                text = str(item.get("name", "")).strip()
            if not text:
                continue
            texts_to_embed.append(text)
            valid_items.append(item)

        existing_ids = set()
        try:
            scroll_point = None
            while True:
                points, scroll_point = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=scroll_point,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                )
                existing_ids.update(point.id for point in points)
                if scroll_point is None:
                    break
        except Exception:
            pass

        print(f"[VectorStore] Found {len(existing_ids)} existing items. Skipping them.")

        batch_size = 50
        for index in range(0, len(texts_to_embed), batch_size):
            batch_texts = []
            batch_items = []
            for text, item in zip(
                texts_to_embed[index : index + batch_size],
                valid_items[index : index + batch_size],
            ):
                if item["id"] in existing_ids:
                    continue
                batch_texts.append(text)
                batch_items.append(item)

            if not batch_texts:
                continue

            response = None
            attempt = 0
            num_keys = len(get_api_key_rotator().api_keys)
            max_attempts = max(10, num_keys + 2)
            while attempt < max_attempts:
                try:
                    response = self.genai_client.models.embed_content(
                        model=self.embedding_model,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                    )
                    break
                except Exception as exc:
                    attempt += 1
                    error_text = str(exc)
                    from src.core.api_utils import is_rate_limit_error, _is_retryable
                    
                    if is_rate_limit_error(exc) and attempt < max_attempts:
                        print(f"[VectorStore] Rate limit detected (attempt {attempt}). Rotating API key.")
                        self._update_api_key_on_rate_limit()
                        time.sleep(2)
                        continue
                    
                    if _is_retryable(exc) and attempt < max_attempts:
                        delay = 2 * attempt
                        print(f"[VectorStore] Transient error. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                        
                    raise

            if response is None:
                continue

            points = [
                rest.PointStruct(
                    id=item["id"],
                    vector=embedding.values,
                    payload=item,
                )
                for item, embedding in zip(batch_items, response.embeddings)
            ]
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)

            if index + batch_size < len(texts_to_embed):
                time.sleep(2)
