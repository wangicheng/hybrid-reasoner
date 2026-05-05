import csv, json, math, sys
from pathlib import Path
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BATCH = Path("data/experiments/runs/batch_20260505_030239_bm25_final")
ANNOTATIONS_PATH = Path("data/experiments/annotations/annotated.csv")

annotations = {}
with ANNOTATIONS_PATH.open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        qid = row.get("Query ID", "").strip()
        bid = row.get("Book ID", "").strip()
        try:
            sc = float(row.get("Score (0-3)", "0") or 0)
        except Exception:
            sc = 0.0
        if qid and bid:
            annotations.setdefault(qid, {})[bid] = sc

def ndcg10(qid, results):
    rels = [annotations.get(qid, {}).get(str(r["book_id"]), 0.0) for r in results[:10]]
    ideal = sorted(annotations.get(qid, {}).values(), reverse=True)[:10]
    def dcg(r):
        return sum((2**v - 1) / math.log2(i + 2) for i, v in enumerate(r))
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0

run_b000 = json.loads((BATCH / "s4a6_b000_sameparse.json").read_text(encoding="utf-8"))
run_b010 = json.loads((BATCH / "s4a6_b010_sameparse.json").read_text(encoding="utf-8"))
run_s3b0 = json.loads((BATCH / "s3a7_b000_sameparse.json").read_text(encoding="utf-8"))

print("Per-query NDCG: s4a6_b000 vs s4a6_b010 vs s3a7_b000")
print(f"  {'Query':<8} {'b000':>8} {'b010':>8} {'s3_b0':>8}  Delta")
ndcgs_b000, ndcgs_b010, ndcgs_s3 = [], [], []
for e0, e10, es3 in zip(run_b000, run_b010, run_s3b0):
    qid = str(e0["query_id"])
    n0 = ndcg10(qid, e0["results"])
    n10 = ndcg10(qid, e10["results"])
    ns3 = ndcg10(qid, es3["results"])
    ndcgs_b000.append(n0)
    ndcgs_b010.append(n10)
    ndcgs_s3.append(ns3)
    flag = ""
    if n10 - n0 > 0.05:
        flag = "<-- big gain"
    elif n0 - n10 > 0.05:
        flag = "<-- big loss"
    print(f"  {qid:<8} {n0:>8.4f} {n10:>8.4f} {ns3:>8.4f}  {n10-n0:+.4f} {flag}")

print(f"  {'MEAN':<8} {mean(ndcgs_b000):>8.4f} {mean(ndcgs_b010):>8.4f} {mean(ndcgs_s3):>8.4f}")

# Check bm25_metric distribution in top-10 for b010
print()
print("In s4a6_b010: bm25_metric distribution in top-10 slots:")
zero = pos = 0
for entry in run_b010:
    for r in entry["results"][:10]:
        m = r.get("bm25_metric", 0.0)
        if m == 0:
            zero += 1
        else:
            pos += 1
total = zero + pos
print(f"  bm25_metric=0: {zero} ({zero/total:.1%})")
print(f"  bm25_metric>0: {pos} ({pos/total:.1%})")

# Understand WHY s4a6_b000 NDCG is so low (0.553 from CSV)
# -- check the annotation coverage
print()
print("Annotation coverage for s4a6_b000 top-10:")
missing = present = 0
for entry in run_b000:
    qid = str(entry["query_id"])
    for r in entry["results"][:10]:
        bid = str(r["book_id"])
        if bid in annotations.get(qid, {}):
            present += 1
        else:
            missing += 1
print(f"  Annotated: {present}, Missing (treated as 0): {missing}")
