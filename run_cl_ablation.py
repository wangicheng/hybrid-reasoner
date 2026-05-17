"""Run candidate-limit ablation experiments into a flat output directory."""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.generate_run import RunGenerator
from src.config import settings
from src.core.model_catalog import normalize_model_id

OUTPUT_DIR = Path("data/experiments/runs/batch_cl_ablation_all")
QUERIES_PATH = Path("data/experiments/queries.json")
CACHE_NAMESPACE = "cl_ablation"

# Candidate limits to run (skip 20 and 30 - already done)
CANDIDATE_LIMITS = [40, 50, 75, 200]

# Fixed experiment parameters
MODEL_ID = "gemma-4-31b-it"
SEMANTIC_WEIGHT = 0.4
ATTRIBUTE_WEIGHT = 0.6
ENABLE_BM25 = True
BM25_WEIGHT = 0.1
SHUFFLE_SEED = 42

settings.RERANK_MODEL_ID = MODEL_ID
settings.RERANK_SHUFFLE_SEED = SHUFFLE_SEED

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    sample_queries = json.load(f)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for cl in CANDIDATE_LIMITS:
    engine_name = f"gemma4_default_parser_bm25_on_cl{cl}"
    output_path = OUTPUT_DIR / f"{engine_name}.json"
    
    # Check if already complete
    if output_path.exists():
        try:
            data = json.load(open(output_path, "r", encoding="utf-8"))
            complete = [d for d in data if "error" not in d]
            if len(complete) >= 24:
                print(f"\n[SKIP] CL={cl}: Already complete with {len(complete)} queries")
                continue
            else:
                print(f"\n[RESUME] CL={cl}: Resuming from {len(complete)}/24 queries")
        except:
            pass
    
    print(f"\n{'='*60}")
    print(f"Starting CL={cl} experiment")
    print(f"{'='*60}")
    
    generator = RunGenerator(
        k_per_engine=10,
        model_id=MODEL_ID,
        rerank=None,  # Use default (enabled)
        cache_namespace=CACHE_NAMESPACE,
    )
    
    try:
        generator.generate_run(
            queries_config=sample_queries,
            engine_name=engine_name,
            output_dir=OUTPUT_DIR,
            semantic_weight=SEMANTIC_WEIGHT,
            attribute_weight=ATTRIBUTE_WEIGHT,
            run_suffix="",
            enable_bm25=ENABLE_BM25,
            bm25_weight=BM25_WEIGHT,
            rerank_candidate_limit=cl,
            rerank_shuffle_seed=SHUFFLE_SEED,
        )
    except Exception as exc:
        print(f"[ERROR] CL={cl} failed: {exc}")
    
    # Small delay between experiments
    time.sleep(2)

print("\n" + "="*60)
print("All candidate-limit experiments finished!")
print("="*60)

# Final validation
for cl in [20, 30] + CANDIDATE_LIMITS:
    p = OUTPUT_DIR / f"gemma4_default_parser_bm25_on_cl{cl}.json"
    if p.exists():
        data = json.load(open(p, "r", encoding="utf-8"))
        errors = sum(1 for d in data if "error" in d)
        print(f"  CL={cl}: {len(data)} queries, {errors} errors ({'OK' if len(data)==24 and errors==0 else 'ISSUE'})")
    else:
        print(f"  CL={cl}: MISSING")
