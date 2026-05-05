#!/usr/bin/env python
import os
import re
import json
import csv
import argparse
from typing import List, Dict, Set

from src.core.vector_store import VectorStore
from src.core.lexical_store import LexicalStore
from src.core.database import Database


def extract_run_index(filename: str) -> str:
    m = re.search(r"run(\d+)\.json", filename)
    return m.group(1) if m else ""


def normalize_id_from_payload(hit: Dict) -> str:
    payload = hit.get("payload") or {}
    return str(payload.get("id") or payload.get("book_id") or hit.get("id") or "").strip()


def normalize_id_from_item(item: Dict) -> str:
    return str(item.get("id") or item.get("book_id") or item.get("bookId") or "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="data/experiments/runs/batch_20260504_042650")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exp_dir = args.experiment_dir
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"Experiment dir not found: {exp_dir}")

    files = sorted([f for f in os.listdir(exp_dir) if f.endswith(".json")])
    on_files = [f for f in files if "_bm25_on_" in f]
    off_files = [f for f in files if "_bm25_off_" in f]

    pairs = []
    for on in on_files:
        idx = extract_run_index(on)
        match_off = next((o for o in off_files if extract_run_index(o) == idx), None)
        if match_off:
            pairs.append((on, match_off))

    if not pairs:
        raise SystemExit("No on/off run pairs found in experiment dir")

    # init stores once
    db = Database()
    vs = VectorStore(collection_name="novels")
    lexical = LexicalStore(db)

    out_csv = args.out or os.path.join(exp_dir, "bm25_diagnostic.csv")
    header = [
        "pair",
        "query_id",
        "query_text",
        "bm25_added_count",
        "bm25_total",
        "vector_total",
        "top10_changed",
        "bm25_added_in_top10",
        "bm25_sample",
    ]

    rows = []
    aggregates = {
        "pairs": len(pairs),
        "queries_checked": 0,
        "queries_with_bm25_additions": 0,
        "queries_with_top10_changes": 0,
        "total_bm25_added": 0,
    }

    for pair_idx, (on_file, off_file) in enumerate(pairs, start=1):
        on_path = os.path.join(exp_dir, on_file)
        off_path = os.path.join(exp_dir, off_file)
        try:
            on_data = json.load(open(on_path, "r", encoding="utf-8"))
            off_data = json.load(open(off_path, "r", encoding="utf-8"))
        except Exception as exc:
            print(f"Failed to load pair {on_file}/{off_file}: {exc}")
            continue

        off_map = {entry.get("query_id") or entry.get("query"): entry for entry in off_data}

        for entry in on_data:
            aggregates["queries_checked"] += 1
            qid = entry.get("query_id") or entry.get("query")
            query_text = entry.get("search_terms") or entry.get("generated_keywords") or entry.get("query") or ""

            # vector recall
            try:
                vector_results, _ = vs.search(query_text, limit=10000, with_payload=True)
            except Exception as exc:
                print(f"Vector search failed for query {qid}: {exc}")
                vector_results = []

            vector_ids: Set[str] = set()
            for hit in vector_results:
                hid = normalize_id_from_payload(hit)
                if hid:
                    vector_ids.add(hid)

            # bm25 recall
            try:
                bm25_results = lexical.search(query_text, limit=1000)
            except Exception as exc:
                print(f"BM25 search failed for query {qid}: {exc}")
                bm25_results = []

            bm25_ids: Set[str] = set()
            for r in bm25_results:
                item = r.get("item") or {}
                bid = normalize_id_from_item(item)
                if bid:
                    bm25_ids.add(bid)

            bm25_only = bm25_ids - vector_ids

            # top10 comparison using exported run results
            top10_on = [str(x.get("book_id") or x.get("id") or "") for x in entry.get("results", [])][:10]
            top10_off = []
            off_entry = off_map.get(entry.get("query_id") or entry.get("query"))
            if off_entry:
                top10_off = [str(x.get("book_id") or x.get("id") or "") for x in off_entry.get("results", [])][:10]

            top10_changed = top10_on != top10_off
            bm25_in_top10 = any(bid in bm25_only for bid in top10_on if bid)

            if bm25_only:
                aggregates["queries_with_bm25_additions"] += 1
                aggregates["total_bm25_added"] += len(bm25_only)

            if top10_changed:
                aggregates["queries_with_top10_changes"] += 1

            rows.append([
                f"{pair_idx}:{on_file}",
                qid,
                (entry.get("query") or "").replace("\n", " ")[:200],
                len(bm25_only),
                len(bm25_ids),
                len(vector_ids),
                int(top10_changed),
                int(bm25_in_top10),
                ";".join(list(bm25_only)[:5]),
            ])

    # write CSV
    try:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"Diagnostic CSV written to: {out_csv}")
    except Exception as exc:
        print(f"Failed to write CSV: {exc}")

    # print summary
    print("\nBM25 Diagnostic Summary")
    print(json.dumps(aggregates, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
