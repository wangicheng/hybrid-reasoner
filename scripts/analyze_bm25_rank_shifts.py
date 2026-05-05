"""
Deep analysis of why BM25 bonus scoring degrades strict-only performance.

Compares s4a6_b000 vs s4a6_b010 to identify:
  1. Which books ENTER top-10 because of BM25 boost
  2. Which books get PUSHED OUT because of BM25 boost
  3. Their annotation scores (do new entrants have lower quality?)
"""
import csv
import json
from pathlib import Path

BATCH = Path("data/experiments/runs/batch_20260505_030239_bm25_final")
ANNOTATIONS_PATH = Path("data/experiments/annotations/annotated.csv")

# Load annotations
annotations: dict[str, dict[str, float]] = {}
with ANNOTATIONS_PATH.open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        qid = row.get("Query ID", "").strip()
        bid = row.get("Book ID", "").strip()
        try:
            sc = float(row.get("Score (0-3)", "0") or 0)
        except ValueError:
            sc = 0.0
        if qid and bid:
            annotations.setdefault(qid, {})[bid] = sc


def load_run(name: str) -> list[dict]:
    return json.loads((BATCH / name).read_text(encoding="utf-8"))


def extract_bm25_metrics(results: list[dict]) -> dict[str, float]:
    return {str(r["book_id"]): r.get("bm25_metric", 0.0) for r in results}


def analyze_pair(name_base: str, name_bm25: str, label: str) -> None:
    run_base = load_run(name_base)
    run_bm25 = load_run(name_bm25)

    entered_all, dropped_all = [], []

    for e0, eb in zip(run_base, run_bm25):
        qid = str(e0["query_id"])
        ids0 = [str(r["book_id"]) for r in e0["results"][:10]]
        ids_b = [str(r["book_id"]) for r in eb["results"][:10]]
        ann = annotations.get(qid, {})
        bm25_metrics = extract_bm25_metrics(eb["results"])

        for bid in set(ids_b) - set(ids0):   # entered
            entered_all.append({
                "qid": qid,
                "book_id": bid,
                "ann_score": ann.get(bid, -1.0),
                "bm25_metric": round(bm25_metrics.get(bid, 0.0), 3),
            })

        for bid in set(ids0) - set(ids_b):   # dropped
            dropped_all.append({
                "qid": qid,
                "book_id": bid,
                "ann_score": ann.get(bid, -1.0),
                "bm25_metric": 0.0,
            })

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {name_base}  →  {name_bm25}")
    print(f"{'='*70}")
    print(f"  Total rank-shift events: {len(entered_all) + len(dropped_all)}")
    print(f"    Books ENTERED top-10 via BM25: {len(entered_all)}")
    print(f"    Books DROPPED from top-10 by BM25: {len(dropped_all)}")

    if entered_all:
        avg_e = sum(r["ann_score"] for r in entered_all) / len(entered_all)
        dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
        for r in entered_all:
            k = int(r["ann_score"]) if r["ann_score"] >= 0 else -1
            dist[k] = dist.get(k, 0) + 1
        print(f"\n  ENTERED avg annotation: {avg_e:.3f}")
        print(f"  Score distribution: 0={dist[0]}  1={dist[1]}  2={dist[2]}  3={dist[3]}  (unknown={dist[-1]})")
        print("  Worst entrants (low annotation):")
        for r in sorted(entered_all, key=lambda x: x["ann_score"])[:8]:
            print(f"    q={r['qid']}  book={r['book_id']}  ann={r['ann_score']}  bm25={r['bm25_metric']}")

    if dropped_all:
        avg_d = sum(r["ann_score"] for r in dropped_all) / len(dropped_all)
        dist = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
        for r in dropped_all:
            k = int(r["ann_score"]) if r["ann_score"] >= 0 else -1
            dist[k] = dist.get(k, 0) + 1
        print(f"\n  DROPPED avg annotation:  {avg_d:.3f}")
        print(f"  Score distribution: 0={dist[0]}  1={dist[1]}  2={dist[2]}  3={dist[3]}  (unknown={dist[-1]})")
        print("  Best dropped books (high annotation, now ousted):")
        for r in sorted(dropped_all, key=lambda x: -x["ann_score"])[:8]:
            print(f"    q={r['qid']}  book={r['book_id']}  ann={r['ann_score']}")

    # Summarize the quality trade-off
    if entered_all and dropped_all:
        avg_e = sum(r["ann_score"] for r in entered_all) / len(entered_all)
        avg_d = sum(r["ann_score"] for r in dropped_all) / len(dropped_all)
        delta = avg_e - avg_d
        print(f"\n  ★ Quality trade-off: ENTERED avg {avg_e:.3f} vs DROPPED avg {avg_d:.3f}")
        if delta < 0:
            print(f"  ⚠  BM25 is swapping OUT good books (avg {avg_d:.3f}) for worse ones (avg {avg_e:.3f}) Δ={delta:+.3f}")
        else:
            print(f"  ✓  BM25 is upgrading quality: new entrants score higher Δ={delta:+.3f}")


# Compare key pairs
analyze_pair("s4a6_b000_sameparse.json", "s4a6_b010_sameparse.json",
             "s4a6: Effect of β_max 0.00 → 0.10")

analyze_pair("s4a6_b000_sameparse.json", "s4a6_b002_sameparse.json",
             "s4a6: Effect of β_max 0.00 → 0.02 (smallest dose)")

analyze_pair("s3a7_b000_sameparse.json", "s3a7_b002_sameparse.json",
             "s3a7: Effect of β_max 0.00 → 0.02")

analyze_pair("s4a6_b000_sameparse.json", "s4a6_b020_sameparse.json",
             "s4a6: Effect of β_max 0.00 → 0.20 (large dose)")
