"""Extract tag mapping details from a run JSON and save as CSV.

Usage:
    python -m src.eval.extract_tag_mappings --run-file <path/to/run.json>
If --run-file is omitted, the script picks the latest file matching
`data/experiments/runs/**/gemma4_dual_track_tagmap.json`.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

def find_latest_run():
    base = Path("data/experiments/runs")
    candidates = list(base.rglob("gemma4_dual_track_tagmap.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-file", type=str, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_file) if args.run_file else find_latest_run()
    if not run_path or not run_path.exists():
        print(f"Run file not found: {args.run_file or '<auto>'}")
        sys.exit(1)

    out_csv = run_path.with_name(run_path.stem + "_mappings.csv")

    with run_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        qid = entry.get("query_id")
        query = entry.get("query", "")
        tag_mapping = entry.get("tag_mapping") or []
        # If tag_mapping absent, fallback to tag_intent
        if not tag_mapping:
            # build minimal mapping from tag_intent positive_terms
            ti = entry.get("tag_intent") or {}
            pos = ti.get("positive_terms") or []
            for term in pos:
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "term": term,
                    "is_exact": True,
                    "mapped_tag": term,
                    "raw_score": 1.0,
                    "scaled_score": 1.0,
                })
            continue

        for t in tag_mapping:
            term = t.get("term")
            is_exact = bool(t.get("is_exact"))
            mappings = t.get("mappings") or []
            if not mappings:
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "term": term,
                    "is_exact": is_exact,
                    "mapped_tag": "",
                    "raw_score": "",
                    "scaled_score": "",
                })
                continue
            for m in mappings:
                rows.append({
                    "query_id": qid,
                    "query": query,
                    "term": term,
                    "is_exact": is_exact,
                    "mapped_tag": m.get("tag", ""),
                    "raw_score": m.get("raw_score", ""),
                    "scaled_score": m.get("scaled_score", ""),
                })

    # Write CSV
    fieldnames = ["query_id", "query", "term", "is_exact", "mapped_tag", "raw_score", "scaled_score"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} mapping rows to: {out_csv}")

if __name__ == "__main__":
    main()
