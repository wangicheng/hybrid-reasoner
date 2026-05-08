"""
Tag Description A/B Test — Standalone script comparing Gemma4 tag output
with vs without tag descriptions injected into the LLM prompt.

Produces:
  - data/experiments/tag_desc_ab/results.json
  - docs/tag_description_ab_report.md
"""
import asyncio, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.core.database import Database
from src.core.engine import HybridEngine
from src.core.llm import parse_query
from src.core.vector_store import VectorStore
from src.core.api_utils import _is_retryable
from src.eval.llm_judge import LLMJudge
from src.eval.tag_rules import normalize_tags, apply_hard_filters

MODEL_ID = "gemma-4-31b-it"
K = 10
TAGS_PATH = Path("data/all_tags.json")
TAG_DESC_PATH = Path("data/tag_descriptions.json")
QUERIES_PATH = Path("data/experiments/queries.json")
OUTPUT_DIR = Path("data/experiments/tag_desc_ab")
REPORT_PATH = Path("docs/tag_description_ab_report.md")


def load_json(p): 
    with open(p, "r", encoding="utf-8") as f: return json.load(f)


def build_tag_lists(tags, descs):
    tags_only = tuple(tags)
    tags_with_desc = tuple(
        f"{t}: {descs[t]}" if t in descs else t for t in tags
    )
    return tags_only, tags_with_desc


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
    f1 = (2*prec*recall/(prec+recall)) if (prec+recall) > 0 else 0.0
    return {"precision": prec, "recall": recall, "f1": f1,
            "avoidance": avoidance, "violations": violations,
            "required_hits": hits, "required_covered": covered}


def search_with_retry(engine, query, q_id, model_id, cache_ns):
    attempt = 0
    while True:
        try:
            return asyncio.run(engine.search(
                query, limit=K, model_id=model_id,
                explain=False, cache_namespace=cache_ns))
        except Exception as exc:
            if not _is_retryable(exc): raise
            attempt += 1
            print(f"  [Retry] {q_id} attempt {attempt}: {exc}")
            time.sleep(1.0)


def run_experiment():
    tags = load_json(TAGS_PATH)
    descs = load_json(TAG_DESC_PATH)
    queries = load_json(QUERIES_PATH)
    tags_only, tags_with_desc = build_tag_lists(tags, descs)

    print(f"Tags: {len(tags)} | Descriptions: {len(descs)} | Queries: {len(queries)}")

    db = Database()
    vs = VectorStore(collection_name="novels")
    engine = HybridEngine(db=db, vs=vs, semantic_weight=0.4, attribute_weight=0.6)
    judge = LLMJudge(model_id="gemini-2.5-flash-lite")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"

    all_results, completed_ids = [], set()
    if results_path.exists():
        try:
            existing = load_json(results_path)
            for item in existing:
                if item.get("control") and item.get("treatment"):
                    all_results.append(item)
                    completed_ids.add(item["query_id"])
            print(f"Resuming: {len(completed_ids)} done.")
        except Exception: pass

    conditions = [("control", tags_only), ("treatment", tags_with_desc)]

    for qi, qc in enumerate(queries, 1):
        q_id, query = qc["id"], qc["query"]
        golden = qc.get("golden_rules", {})
        if q_id in completed_ids:
            print(f"[{qi}/{len(queries)}] Skip {q_id}")
            continue

        print(f"\n[{qi}/{len(queries)}] {q_id}: {query[:50]}...")
        qr = {"query_id": q_id, "query": query, "golden_rules": golden}

        for cond_name, tag_list in conditions:
            print(f"  --- {cond_name.upper()} ---")
            cache_ns = f"ab_{cond_name}_{q_id}"

            try:
                # Temporarily swap the engine's tag list for this condition
                original_tags = engine.all_tags_cache
                engine.all_tags_cache = tag_list
                try:
                    response = search_with_retry(engine, query, q_id, MODEL_ID, cache_ns)
                finally:
                    engine.all_tags_cache = original_tags

                results = response.get("results", [])
                tag_intent = response.get("tag_intent", {})
                pos = tag_intent.get("positive_terms", [])
                neg = tag_intent.get("negative_terms", [])
                tm = compute_tag_metrics(pos, golden)

                scores = []
                extracted = []
                for rank, res in enumerate(results[:K]):
                    item = res.get("item", {})
                    b_id = str(item.get("id", "")).strip()
                    if not b_id: continue
                    title = item.get("name", "")
                    btags = item.get("tags", [])
                    intro = item.get("intro", "")
                    tags_str = ", ".join(str(t) for t in btags) if isinstance(btags, list) else str(btags)

                    j = judge.judge_single(query=query, title=title, tags=tags_str, intro=intro)
                    scores.append(j["score"])
                    meta = {"words_total": item.get("words_total",0),
                            "publish_status": item.get("publish_status",""),
                            "tags": btags, "is_animated": item.get("is_animated",False)}
                    extracted.append({"rank": rank+1, "book_id": b_id, "title": title,
                                      "tags": btags, "judge_score": j["score"],
                                      "reasoning": j.get("reasoning",""),
                                      "passes_filter": apply_hard_filters(golden, meta)})
                    print(f"    [{rank+1}] {title[:30]} | {j['score']}/3")

                avg = sum(scores)/len(scores) if scores else 0.0
                good = sum(1 for s in scores if s>=2)/len(scores) if scores else 0.0
                strong = sum(1 for s in scores if s>=3)/len(scores) if scores else 0.0

                qr[cond_name] = {"positive_terms": pos, "negative_terms": neg,
                                 "tag_metrics": tm, "judge_scores": scores,
                                 "avg_score": avg, "good_rate": good, "strong_rate": strong,
                                 "results": extracted}
                print(f"  [{cond_name.upper()}] Avg:{avg:.2f} Good:{good:.0%} Recall:{tm['recall']:.0%}")
            except Exception as exc:
                print(f"  [ERROR] {cond_name} {q_id}: {exc}")
                qr[cond_name] = {"error": str(exc), "positive_terms": [],
                                 "tag_metrics": {}, "judge_scores": [],
                                 "avg_score": 0, "good_rate": 0, "strong_rate": 0, "results": []}

        all_results.append(qr)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  Saved ({len(all_results)}/{len(queries)})")

    vs.client.close()
    return all_results


def generate_report(results_path=OUTPUT_DIR / "results.json"):
    data = load_json(results_path)

    def agg(key, sub=None):
        vals = []
        for r in data:
            v = r.get(key, {})
            if sub: v = v.get("tag_metrics", {})
            val = v.get(sub or "avg_score") if isinstance(v, dict) else None
            if val is not None: vals.append(val)
        return sum(vals)/len(vals) if vals else 0.0

    def collect(cond, field, sub=None):
        vals = []
        for r in data:
            v = r.get(cond, {})
            if sub: v = v.get("tag_metrics", {})
            val = v.get(sub or field) if isinstance(v, dict) else None
            if val is not None: vals.append(val)
        return vals

    def m(vals): return sum(vals)/len(vals) if vals else 0.0
    def f(v): return f"{v:.3f}"
    def p(v): return f"{v:.1%}"
    def d(c,t): return f"+{t-c:.3f}" if t>=c else f"{t-c:.3f}"
    def dp(c,t): return f"+{t-c:.1%}" if t>=c else f"{t-c:.1%}"

    ca, ta = m(collect("control","avg_score")), m(collect("treatment","avg_score"))
    cg, tg = m(collect("control","good_rate")), m(collect("treatment","good_rate"))
    cs, ts = m(collect("control","strong_rate")), m(collect("treatment","strong_rate"))
    cr, tr = m(collect("control","recall","recall")), m(collect("treatment","recall","recall"))
    cp, tp = m(collect("control","precision","precision")), m(collect("treatment","precision","precision"))
    cf, tf = m(collect("control","f1","f1")), m(collect("treatment","f1","f1"))
    cv, tv = m(collect("control","avoidance","avoidance")), m(collect("treatment","avoidance","avoidance"))

    L = []
    L.append("# Tag Description A/B 實驗報告\n")
    L.append(f"> 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("## 1. 實驗設計\n")
    L.append("比較 Gemma4 在 prompt 中**有/無標籤敘述**對查詢解析和推薦品質的影響。\n")
    L.append("| 項目 | Control | Treatment |")
    L.append("|------|---------|-----------|")
    L.append(f"| Tag List | 名稱 | 名稱+敘述 |")
    L.append(f"| 模型 | `{MODEL_ID}` | `{MODEL_ID}` |")
    L.append(f"| 查詢數 | {len(data)} | {len(data)} |")
    L.append(f"| Top-K | {K} | {K} |\n")

    L.append("## 2. 整體指標比較\n")
    L.append("| 指標 | Control | Treatment | Δ |")
    L.append("|------|---------|-----------|---|")
    L.append(f"| **Avg Score** | {f(ca)} | {f(ta)} | {d(ca,ta)} |")
    L.append(f"| **Good Rate (≥2)** | {p(cg)} | {p(tg)} | {dp(cg,tg)} |")
    L.append(f"| **Strong Rate (=3)** | {p(cs)} | {p(ts)} | {dp(cs,ts)} |")
    L.append(f"| **Tag Recall** | {p(cr)} | {p(tr)} | {dp(cr,tr)} |")
    L.append(f"| **Tag Precision** | {p(cp)} | {p(tp)} | {dp(cp,tp)} |")
    L.append(f"| **Tag F1** | {p(cf)} | {p(tf)} | {dp(cf,tf)} |")
    L.append(f"| **Blocked Avoidance** | {p(cv)} | {p(tv)} | {dp(cv,tv)} |\n")

    winner = "Treatment（有標籤敘述）" if ta>ca+0.05 else "Control（無標籤敘述）" if ca>ta+0.05 else "兩者持平"
    L.append(f"> **整體勝出**: {winner}\n")

    L.append("## 3. 逐查詢比較\n")
    L.append("| Query | Ctrl Avg | Treat Avg | Δ | Ctrl Recall | Treat Recall | Δ |")
    L.append("|-------|----------|-----------|---|-------------|--------------|---|")
    for r in data:
        qi = r["query_id"]
        c = r.get("control",{}); t = r.get("treatment",{})
        cs_=c.get("avg_score",0); ts_=t.get("avg_score",0)
        cr_=c.get("tag_metrics",{}).get("recall",0); tr_=t.get("tag_metrics",{}).get("recall",0)
        L.append(f"| {qi} | {f(cs_)} | {f(ts_)} | {d(cs_,ts_)} | {p(cr_)} | {p(tr_)} | {dp(cr_,tr_)} |")
    L.append("")

    L.append("## 4. 標籤輸出比較\n")
    for r in data:
        qi = r["query_id"]
        qs = r["query"][:60].replace("\n"," ")
        req = r.get("golden_rules",{}).get("required_tags",[])
        blk = r.get("golden_rules",{}).get("blocked_tags",[])
        cp_ = r.get("control",{}).get("positive_terms",[])
        tp_ = r.get("treatment",{}).get("positive_terms",[])
        L.append(f"### {qi}: {qs}...")
        L.append(f"- Required: {', '.join(req) if req else '(無)'}")
        L.append(f"- Blocked: {', '.join(blk) if blk else '(無)'}")
        L.append(f"- Control: {', '.join(cp_) if cp_ else '(空)'}")
        L.append(f"- Treatment: {', '.join(tp_) if tp_ else '(空)'}")
        only_c = set(cp_)-set(tp_); only_t = set(tp_)-set(cp_)
        if only_c: L.append(f"- 🔴 只在Control: {', '.join(only_c)}")
        if only_t: L.append(f"- 🟢 只在Treatment: {', '.join(only_t)}")
        if not only_c and not only_t: L.append("- ⚪ 兩組相同")
        L.append("")

    L.append("## 5. 結論\n")
    if ta > ca + 0.05:
        L.append(f"加入標籤敘述**有效提升**推薦品質（Δ = {d(ca,ta)}）。")
    elif ca > ta + 0.05:
        L.append(f"加入標籤敘述**未能提升**推薦品質（Δ = {d(ca,ta)}）。")
    else:
        L.append(f"兩組表現**接近**（Δ = {d(ca,ta)}）。")
    L.append("")
    if tr > cr: L.append(f"Tag Recall 提升：{p(cr)} → {p(tr)}，表示敘述有助 LLM 選中 required tags。")
    elif cr > tr: L.append(f"Tag Recall 下降：{p(cr)} → {p(tr)}，敘述可能讓 LLM 過於保守。")
    L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nExperiment done in {(time.time()-t0)/60:.1f}min")
    report = generate_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f: f.write(report)
    print(f"Report: {REPORT_PATH}")
