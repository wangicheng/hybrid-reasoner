# BM25 Diagnostic Report

Experiment: `batch_20260504_042650`

## Summary

- Run pairs checked: 5
- Queries checked: 120
- Queries with BM25-only additions: 0
- Queries with top-10 changes between BM25 on/off: 0
- Total BM25-only additions: 0

## Per-query CSV

The detailed per-query output is saved at:

- `bm25_diagnostic.csv`

The lightweight BM25-only diagnostic is saved at:

- `bm25_diagnostic_bm25only.csv`

## Interpretation

For every query in this batch, the BM25 recall set did not add any candidate that was absent from vector recall in the full diagnostic run. That matches the earlier observation that the BM25 on/off runs are identical at the top-10 level.

The CSV includes one row per query with these fields:

- `bm25_added_count`
- `bm25_total`
- `vector_total`
- `top10_changed`
- `bm25_added_in_top10`
