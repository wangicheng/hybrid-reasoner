import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from google import genai
from google.genai import types

from src.core.api_utils import _is_retryable, get_api_key_rotator, get_rate_limiter
from src.core.database import Database
from src.core.engine import HybridEngine
from src.core.vector_store import VectorStore
from src.eval.llm_judge import LLMJudge
from src.eval.tag_rules import apply_hard_filters

def normalize_tags(tags):
    if isinstance(tags, str):
        return [tags.strip()] if tags.strip() else []
    return [str(t).strip() for t in tags if str(t).strip()]

def compute_tag_metrics(positive_terms, golden_rules):
    req = normalize_tags(golden_rules.get("required_tags", []))
    blk = normalize_tags(golden_rules.get("blocked_tags", []))
    if req:
        hits = sum(1 for t in positive_terms if t in req)
        prec = hits / len(positive_terms) if positive_terms else 0.0
        covered = sum(1 for r in req if r in positive_terms)
        recall = covered / len(req)
    else:
        hits, covered, prec, recall = 0, 0, 0.0, 1.0
    violations = [t for t in positive_terms if t in blk] if blk else []
    avoidance = 1.0 - (len(violations) / len(blk)) if blk else 1.0
    f1 = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else 0.0
    return {"precision": prec, "recall": recall, "f1": f1, "avoidance": avoidance}

MODEL_ID_RERANKER = "gemma-4-31b-it"
K = 10
CANDIDATE_LIMIT = 100

QUERIES_PATH = Path("data/experiments/queries.json")
OUTPUT_DIR = Path("data/experiments/reranking_ab")
REPORT_PATH = Path("docs/reranking_ab_report.md")

BASELINE_ORIGINAL_AVG = 1.930
BASELINE_PURE_LLM_AVG = 1.50

class PointwiseReranker:
    def __init__(self, model_id: str = MODEL_ID_RERANKER):
        self.model_id = model_id
        self.rotator = get_api_key_rotator()
        self.client = genai.Client(api_key=self.rotator.get_current_key())
        self.rate_limiter = get_rate_limiter()

    def _rotate_api_key(self) -> None:
        new_key = self.rotator.on_rate_limit_error()
        self.client = genai.Client(api_key=new_key)
        print(f"  [reranker] API key rotated. Current index: {self.rotator.current_index}")

    async def _score_single(self, query: str, title: str, tags: str, intro: str, rank: int) -> Dict[str, Any]:
        prompt = f"""\
You are an expert web novel recommender. Rate how well this book matches the user's query.

User Query: {query}

Book Title: {title}
Tags: {tags}
Intro: {intro}

Score the relevance from 0 to 100.
0 = Completely irrelevant.
100 = Perfect match.

Output ONLY a valid JSON object with exactly two keys: "reasoning" (string, brief explanation in Traditional Chinese) and "score" (integer 0-100). Do not include markdown blocks like ```json.
"""
        attempt = 0
        while attempt < 10:
            try:
                self.rate_limiter.wait()
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                
                if not response.text:
                    raise ValueError("Empty response")
                
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text.rsplit("\n", 1)[0]
                text = text.strip()
                
                parsed = json.loads(text)
                score = int(parsed.get("score", 0))
                return {"score": score, "reasoning": parsed.get("reasoning", ""), "original_rank": rank}
            except Exception as exc:
                attempt += 1
                if not _is_retryable(exc):
                    return {"score": 0, "reasoning": f"Error: {exc}", "original_rank": rank}
                
                error_text = str(exc)
                if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                    self._rotate_api_key()
                
                await asyncio.sleep(2.0)
        return {"score": 0, "reasoning": "Max retries reached", "original_rank": rank}

    async def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = []
        for i, c in enumerate(candidates):
            title = c.get("name", "")
            tags = ", ".join(c.get("tags", []))
            intro = str(c.get("intro", ""))[:500]
            tasks.append(self._score_single(query, title, tags, intro, i))
        
        sem = asyncio.Semaphore(10)
        async def sem_task(t):
            async with sem:
                return await t
                
        results = await asyncio.gather(*(sem_task(t) for t in tasks))
        
        for c, r in zip(candidates, results):
            c["rerank_score"] = r["score"]
            c["rerank_reasoning"] = r["reasoning"]
            
        reranked = sorted(candidates, key=lambda x: (x.get("rerank_score", 0), -x.get("original_rank", 0)), reverse=True)
        return reranked

def load_json(path: Path):
    if not path.exists(): return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

async def run_experiment():
    queries = load_json(QUERIES_PATH)
    
    db = Database()
    vs = VectorStore(collection_name="novels")
    engine = HybridEngine(db=db, vs=vs, semantic_weight=0.4, attribute_weight=0.6)
    reranker = PointwiseReranker()
    judge = LLMJudge(model_id="gemini-2.5-flash-lite")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    
    all_results = []
    completed_ids = set()
    if results_path.exists():
        try:
            existing = load_json(results_path)
            for item in existing:
                all_results.append(item)
                completed_ids.add(item["query_id"])
            print(f"Resuming: {len(completed_ids)} done.")
        except Exception: pass

    try:
        for qi, qc in enumerate(queries, 1):
            q_id = qc["id"]
            query = qc["query"]
            golden = qc.get("golden_rules", {})
            
            if q_id in completed_ids:
                print(f"[{qi}/{len(queries)}] Skip {q_id}")
                continue
                
            print(f"\n[{qi}/{len(queries)}] {q_id}: {query[:50]}...")
            qr = {"query_id": q_id, "query": query, "golden_rules": golden}
            
            try:
                print("  [Retrieval] Fetching top 100...")
                engine_resp = await engine.search(query, limit=CANDIDATE_LIMIT, model_id=MODEL_ID_RERANKER, cache_namespace=f"rerank_{q_id}", explain=False)
                results = engine_resp.get("results", [])
                pos = engine_resp.get("tag_intent", {}).get("positive_terms", [])
                neg = engine_resp.get("tag_intent", {}).get("negative_terms", [])
                tm = compute_tag_metrics(pos, golden)
                
                qr["original"] = {"positive_terms": pos, "negative_terms": neg, "tag_metrics": tm}
                
                candidates = []
                for rank, res in enumerate(results):
                    item = res.get("item", {})
                    b_id = str(item.get("id", "")).strip()
                    if not b_id: continue
                    candidates.append({
                        "book_id": b_id,
                        "name": item.get("name", ""),
                        "tags": item.get("tags", []),
                        "intro": item.get("intro", ""),
                        "words_total": item.get("words_total", 0),
                        "publish_status": item.get("publish_status", ""),
                        "is_animated": item.get("is_animated", False),
                        "original_rank": rank + 1
                    })
                
                print(f"  [Reranker] Scoring {len(candidates)} candidates with Point-wise {MODEL_ID_RERANKER}...")
                reranked_candidates = await reranker.rerank_candidates(query, candidates)
                
                def judge_list(cands):
                    scores = []
                    extracted = []
                    for res in cands[:K]:
                        b_id = res["book_id"]
                        title = res["name"]
                        tags = res["tags"]
                        tags_str = ", ".join(str(t) for t in tags)
                        intro = res["intro"]
                        
                        j = judge.judge_single(query=query, title=title, tags=tags_str, intro=intro)
                        scores.append(j["score"])
                        
                        extracted.append({
                            "book_id": b_id, "title": title, "tags": tags,
                            "judge_score": j["score"], "reasoning": j.get("reasoning", ""),
                            "rerank_score": res.get("rerank_score", 0),
                            "original_rank": res.get("original_rank", 0)
                        })
                    avg = sum(scores)/len(scores) if scores else 0.0
                    good = sum(1 for s in scores if s>=2)/len(scores) if scores else 0.0
                    strong = sum(1 for s in scores if s>=3)/len(scores) if scores else 0.0
                    return scores, extracted, avg, good, strong

                print("  [Judge] Evaluating original top 10...")
                o_scores, o_extr, o_avg, o_good, o_strong = judge_list(candidates)
                qr["original"].update({"judge_scores": o_scores, "avg_score": o_avg, "good_rate": o_good, "strong_rate": o_strong, "results": o_extr})
                print(f"    Original Avg: {o_avg:.2f} Good: {o_good:.0%}")
                
                print("  [Judge] Evaluating reranked top 10...")
                r_scores, r_extr, r_avg, r_good, r_strong = judge_list(reranked_candidates)
                qr["reranked"] = {"judge_scores": r_scores, "avg_score": r_avg, "good_rate": r_good, "strong_rate": r_strong, "results": r_extr}
                print(f"    Reranked Avg: {r_avg:.2f} Good: {r_good:.0%}")
            
            except Exception as e:
                print(f"  [ERROR] Query failed: {e}")
                qr["error"] = str(e)
                qr["original"] = {"avg_score": 0, "good_rate": 0, "strong_rate": 0}
                qr["reranked"] = {"avg_score": 0, "good_rate": 0, "strong_rate": 0}

            all_results.append(qr)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"  Saved ({len(all_results)}/{len(queries)})")
            
    finally:
        vs.client.close()

def generate_report():
    data = load_json(OUTPUT_DIR / "results.json")
    if not data:
        return "No data to generate report."

    def m(vals): return sum(vals)/len(vals) if vals else 0.0
    def f(v): return f"{v:.3f}"
    def p(v): return f"{v:.1%}"
    
    orig_avgs = [r.get("original", {}).get("avg_score", 0) for r in data if "error" not in r]
    orig_goods = [r.get("original", {}).get("good_rate", 0) for r in data if "error" not in r]
    orig_strongs = [r.get("original", {}).get("strong_rate", 0) for r in data if "error" not in r]
    
    rr_avgs = [r.get("reranked", {}).get("avg_score", 0) for r in data if "error" not in r]
    rr_goods = [r.get("reranked", {}).get("good_rate", 0) for r in data if "error" not in r]
    rr_strongs = [r.get("reranked", {}).get("strong_rate", 0) for r in data if "error" not in r]

    ca = m(orig_avgs)
    cg = m(orig_goods)
    cs = m(orig_strongs)
    
    ta = m(rr_avgs)
    tg = m(rr_goods)
    ts = m(rr_strongs)
    
    L = []
    L.append("# Gemma 4 Point-wise Reranking 實驗報告\n")
    L.append("## 1. 實驗總結\n")
    L.append("在現有的雙軌召回架構（語意向量 + 標籤映射）基礎上，先召回 100 本書，再使用 `gemma-4-31b-it` 對每本書進行 Point-wise 獨立評分 (0-100分) 並重新排序取 Top 10。\n")
    
    L.append("## 2. 核心指標比較\n")
    L.append("| 指標 | 原始版本 (No Rerank) | Gemma 4 Rerank | 純 LLM 版本 (Baseline) |")
    L.append("|------|----------------------|----------------|------------------------|")
    L.append(f"| **Avg Score** | {f(ca)} | {f(ta)} | {f(BASELINE_PURE_LLM_AVG)} |")
    L.append(f"| **Good Rate (≥2)** | {p(cg)} | {p(tg)} | - |")
    L.append(f"| **Strong Rate (=3)** | {p(cs)} | {p(ts)} | - |\n")
    
    L.append("## 3. 逐查詢比較\n")
    L.append("| Query | Original Avg | Reranked Avg | Δ |")
    L.append("|-------|--------------|--------------|---|")
    for r in data:
        qi = r["query_id"]
        if "error" in r:
            L.append(f"| {qi} | ERROR | ERROR | - |")
            continue
        oa = r.get("original", {}).get("avg_score", 0)
        ra = r.get("reranked", {}).get("avg_score", 0)
        delta = f"+{ra-oa:.2f}" if ra>=oa else f"{ra-oa:.2f}"
        L.append(f"| {qi} | {f(oa)} | {f(ra)} | {delta} |")
        
    return "\n".join(L)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report = generate_report()
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"Report written to {REPORT_PATH}")
    else:
        asyncio.run(run_experiment())
        report = generate_report()
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"Report written to {REPORT_PATH}")
