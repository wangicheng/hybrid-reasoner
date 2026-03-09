"""
純重排序延遲測試 (Isolated Rerank Latency Test)
================================================
此腳本將共同流程（LLM 解析、Qdrant 檢索、Criteria 計分）只執行一次，
然後針對三個策略分別計時「僅重排序 (Rerank-Only)」的耗時，
以公平比較三種策略在排序層面的真實成本差異。
"""
import asyncio
import json
import os
import sys
import time
import copy

# Fix stdout encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.llm import parse_query
from src.core.vector_store import VectorStore
from src.core.database import Database
from src.core.reranker import Reranker
from src.core.llm_reranker import LLMReranker
from src.config import settings


async def run_isolated_rerank_test():
    db = Database()
    vs = VectorStore(collection_name="novels")
    reranker = Reranker()
    llm_reranker = LLMReranker()

    queries_path = "data/experiments/queries.json"
    if not os.path.exists(queries_path):
        print(f"Error: {queries_path} not found.")
        return

    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    strategies = ["score_only", "hybrid_fusion", "original_llm_reranker_top10"]

    print("\n" + "=" * 70)
    print("⚗️  [純重排序延遲測試] Isolated Rerank Latency Test")
    print(f"📚 題庫數量: {len(queries_data)} 題")
    print(f"⚙️  對比策略: {', '.join(strategies)}")
    print("📌 說明: 共同流程 (LLM 解析 + Qdrant 檢索 + Criteria 計分) 只執行一次，")
    print("        然後分別計時三個策略「純重排序」的耗時。")
    print("=" * 70 + "\n")

    # 累計各策略的純重排序時間
    metrics = {s: {"total_rerank_time": 0.0, "query_count": 0} for s in strategies}

    for idx, q_data in enumerate(queries_data, 1):
        query_text = q_data["query"]
        print(f"\n[{idx}/{len(queries_data)}] Query: {query_text[:50].replace(chr(10), ' ')}...")

        # ========================================
        # 共同流程 (只跑一次)
        # ========================================
        shared_start = time.time()

        # 1. LLM Parse
        parse_result = parse_query(query_text)

        # 2. Build Qdrant Filter (reuse BaseEngine logic)
        from src.core.engine import HybridReasonerEngine
        temp_engine = HybridReasonerEngine(db=db, vs=vs)
        qdrant_filter = temp_engine._build_qdrant_filter(parse_result.criteria)

        base_terms = " ".join(parse_result.search_terms) or parse_result.original_query
        expanded_terms = base_terms
        if parse_result.generated_keywords:
            cleaned_keywords = [kw.replace(" ", "") for kw in parse_result.generated_keywords]
            expansion_str = " ".join(cleaned_keywords)
            expanded_terms += f" {expansion_str}"

        # 3. Qdrant Vector Search (Stage 1)
        STAGE1_LIMIT = 200
        vector_results, query_vector = vs.search(
            expanded_terms,
            limit=STAGE1_LIMIT,
            query_filter=qdrant_filter,
            with_payload=True
        )

        # Auto-Relax
        is_relaxed = False
        if len(vector_results) < 3 and qdrant_filter is not None:
            vector_results, query_vector = vs.search(
                base_terms,
                limit=STAGE1_LIMIT,
                query_filter=None,
                with_payload=True
            )
            is_relaxed = True

        # Build candidates
        candidates_map = {}
        vector_score_map = {}
        payload_map = {}
        for hit in vector_results:
            item = db.get_item(hit["id"])
            if item and str(item.get("name", "")).strip():
                bid = str(item["id"])
                candidates_map[bid] = item
                vector_score_map[bid] = hit["score"]
                if hit.get('payload'):
                    payload_map[bid] = hit['payload']

        candidates = list(candidates_map.values())

        # 4. Cross-Encoder Semantic Feature Scoring (shared across all strategies)
        semantic_criteria_list = [c for c in parse_result.criteria if c.name == "semantic_similarity"]
        semantic_scores_map = {}
        if semantic_criteria_list and reranker and candidates:
            for sc in semantic_criteria_list:
                if hasattr(sc.parameters, 'model_dump'):
                    sc_params = sc.parameters.model_dump()
                else:
                    sc_params = sc.parameters.dict()
                query_text_feat = sc_params.get("query_text", "")
                if not query_text_feat:
                    continue
                feature_scores = await reranker.score_feature(query_text_feat, candidates)
                score_map_for_feature = {}
                for ci, item in enumerate(candidates):
                    score_map_for_feature[str(item["id"])] = feature_scores[ci]
                semantic_scores_map[query_text_feat] = score_map_for_feature

        # 5. Base Scoring (shared)
        vector_norm_map = {}
        for bid, raw_v in vector_score_map.items():
            vector_norm_map[bid] = temp_engine._normalize_vector_score(float(raw_v))

        base_scored_items = []
        for item in candidates:
            bid = str(item["id"])
            v_score = vector_score_map.get(bid, 0.0)
            v_norm = vector_norm_map.get(bid, temp_engine._normalize_vector_score(float(v_score)))
            score_val, breakdown = temp_engine.calculate_score(
                item,
                parse_result.criteria,
                vector_score=v_score,
                normalized_vector_score=v_norm,
                semantic_scores_map=semantic_scores_map,
            )
            base_scored_items.append({
                "item": item,
                "score": float(score_val),
                "vector_score": v_score,
                "breakdown": breakdown,
                "payload": payload_map.get(str(item["id"]), {})
            })

        shared_time = time.time() - shared_start
        print(f"  ⏱️ 共同流程耗時: {shared_time:.2f}s (candidates: {len(candidates)})")

        # ========================================
        # 各策略「純重排序」計時
        # ========================================

        rerank_query = " ".join(parse_result.search_terms) if parse_result.search_terms else query_text
        alpha = settings.RERANK_FUSION_ALPHA or 0.95
        try:
            alpha = float(alpha)
        except:
            alpha = 0.95

        for strategy in strategies:
            # Deep copy to avoid mutation
            scored_items = copy.deepcopy(base_scored_items)

            rerank_start = time.time()

            # --- score_only: 不做任何額外排序，直接用 base score ---
            if strategy == "score_only":
                for entry in scored_items:
                    entry["final_sort_score"] = float(entry["score"])

            # --- hybrid_fusion: Cross-Encoder rerank ---
            elif strategy == "hybrid_fusion":
                base_candidates = [entry['item'] for entry in scored_items]
                reranked_items = await reranker.rerank(rerank_query, base_candidates, top_k=len(base_candidates))

                rerank_map = {}
                for rank_idx, item in enumerate(reranked_items):
                    item_id = str(item.get("id"))
                    rerank_map[item_id] = {
                        "rerank_score": float(item.get("rerank_score", 0.0)),
                        "rerank_rank": rank_idx + 1,
                    }

                for entry in scored_items:
                    item_id = str(entry["item"].get("id"))
                    rerank_info = rerank_map.get(item_id)
                    if rerank_info:
                        entry["rerank_score"] = rerank_info["rerank_score"]
                    else:
                        entry["rerank_score"] = 0.0

                # Fusion
                base_scores = [float(e["score"]) for e in scored_items]
                rerank_scores = [float(e.get("rerank_score", 0.0)) for e in scored_items]
                from src.core.engine import HybridReasonerEngine as HRE
                norm_base = HRE._minmax_normalize(base_scores)
                norm_rerank = HRE._minmax_normalize(rerank_scores)
                for i, entry in enumerate(scored_items):
                    entry["final_sort_score"] = (1.0 - alpha) * norm_base[i] + alpha * norm_rerank[i]

            # --- original_llm_reranker_top10: LLM rerank ---
            elif strategy == "original_llm_reranker_top10":
                llm_ranked = llm_reranker.rerank(
                    query=rerank_query,
                    candidates=[entry["item"] for entry in scored_items],
                    top_k=10,
                )
                llm_rank_map = {str(r.get("id")): r for r in llm_ranked}
                for entry in scored_items:
                    item_id = str(entry["item"].get("id"))
                    llm_info = llm_rank_map.get(item_id)
                    if llm_info:
                        entry["llm_rerank_score"] = float(llm_info.get("llm_rerank_score", 0.0))
                    else:
                        entry["llm_rerank_score"] = 0.0

                # Fusion
                base_scores = [float(e["score"]) for e in scored_items]
                llm_scores = [float(e.get("llm_rerank_score", 0.0)) for e in scored_items]
                from src.core.engine import HybridReasonerEngine as HRE
                norm_base = HRE._minmax_normalize(base_scores)
                norm_llm = HRE._minmax_normalize(llm_scores)
                for i, entry in enumerate(scored_items):
                    entry["final_sort_score"] = (1.0 - alpha) * norm_base[i] + alpha * norm_llm[i]

            rerank_time = time.time() - rerank_start

            # Sort
            scored_items.sort(key=lambda x: float(x.get("final_sort_score", x["score"])), reverse=True)

            metrics[strategy]["total_rerank_time"] += rerank_time
            metrics[strategy]["query_count"] += 1

            print(f"  └─ [{strategy:<27}] 純重排序耗時: {rerank_time:>6.3f}s")

    # ========================================
    # 總結報告
    # ========================================
    print("\n" + "═" * 70)
    print(" 📊 純重排序延遲總結 (Isolated Rerank Latency Summary)")
    print("═" * 70)

    sorted_strategies = sorted(strategies, key=lambda s: metrics[s]["total_rerank_time"] / max(1, metrics[s]["query_count"]))

    for strategy in sorted_strategies:
        stat = metrics[strategy]
        avg = stat["total_rerank_time"] / max(1, stat["query_count"])
        total = stat["total_rerank_time"]
        count = stat["query_count"]
        print(f"【{strategy}】")
        print(f"  ⏱️ 平均純重排序延遲: {avg:.3f} 秒")
        print(f"  📈 總耗時: {total:.2f} 秒 (共 {count} 題)")
        print("-" * 35)


if __name__ == "__main__":
    asyncio.run(run_isolated_rerank_test())
