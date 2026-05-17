"""Deep inspection of cl30 and cl50 experiments for fairness."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def deep_inspect(path_str, label):
    p = Path(path_str)
    if not p.exists():
        print(f"[MISSING] {label}: {p}")
        return None
    data = json.load(open(p, "r", encoding="utf-8"))
    
    print(f"\n{'='*70}")
    print(f"[{label}] {p.name}")
    print(f"{'='*70}")
    print(f"Total queries: {len(data)}")
    
    # Check parser variant consistency
    parsers = set()
    models = set()
    query_ids = set()
    all_book_ids = {}
    
    for d in data:
        parsers.add(d.get("parser_variant", "?"))
        models.add(d.get("model_id", "?"))
        qid = d.get("query_id", "?")
        query_ids.add(qid)
        
        results = d.get("results", [])
        book_ids = [r.get("book_id") for r in results]
        all_book_ids[qid] = book_ids
        
    print(f"Parser variants: {parsers}")
    print(f"Models: {models}")
    print(f"Query IDs ({len(query_ids)}): {sorted(query_ids)}")
    
    # Expected 24 queries
    expected = {f"q{i}" for i in range(1, 25)}
    missing = expected - query_ids
    if missing:
        print(f"[WARNING] Missing queries: {sorted(missing)}")
    else:
        print(f"[OK] All 24 queries present")
    
    return data

# Load both
cl30 = deep_inspect("data/experiments/runs/batch_20260515_234137/gemma4_default_parser_bm25_on_cl30.json", "CL30")
cl50 = deep_inspect("data/experiments/runs/batch_20260516_123344/gemma4_default_parser_bm25_on_cl50.json", "CL50")

# Check the DEFAULT_PARSER_VARIANT in current code
print("\n\n" + "="*70)
print("CHECKING CURRENT DEFAULT PARSER VARIANT")
print("="*70)
try:
    from src.core.llm import DEFAULT_PARSER_VARIANT
    print(f"Current DEFAULT_PARSER_VARIANT: {DEFAULT_PARSER_VARIANT}")
except Exception as e:
    print(f"Could not import: {e}")

# Check config
try:
    from src.config import settings
    print(f"\nCurrent config:")
    print(f"  RERANK_ENABLED: {settings.RERANK_ENABLED}")
    print(f"  RERANK_CANDIDATE_LIMIT: {settings.RERANK_CANDIDATE_LIMIT}")
    print(f"  RERANK_PERMUTATIONS: {settings.RERANK_PERMUTATIONS}")
    print(f"  RERANK_SHUFFLE_SEED: {settings.RERANK_SHUFFLE_SEED}")
    print(f"  RERANK_MODEL_ID: {settings.RERANK_MODEL_ID}")
    print(f"  SEMANTIC_WEIGHT: {settings.SEMANTIC_WEIGHT}")
    print(f"  ATTRIBUTE_WEIGHT: {settings.ATTRIBUTE_WEIGHT}")
    print(f"  ENABLE_BM25: {settings.ENABLE_BM25}")
except Exception as e:
    print(f"Could not load config: {e}")

# Check the experiment.log for how these were run
print("\n\n" + "="*70)
print("EXPERIMENT LOG (last 50 lines)")
print("="*70)
try:
    lines = open("experiment.log", "r", encoding="utf-8").readlines()
    for line in lines[-50:]:
        print(line.rstrip())
except:
    print("No experiment.log found")
