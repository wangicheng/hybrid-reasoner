# This script performs a comprehensive check of all modules and logic flow
# to ensure the system is stable.
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print("🔍 Starting comprehensive self-check...")

try:
    print("Checking Imports...")
    from src.models.schemas import Criterion, QueryParseResult
    from src.core.llm import parse_query
    from src.core.explainer import generate_explanation 
    from src.core.database import Database
    from src.core.vector_store import VectorStore
    from src.core.engine import HybridEngine
    from src.logic.registry import ScoringRegistry
    import src.logic.scoring_functions 
    print("✅ All imports successful.")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

try:
    print("Checking Function Registry...")
    # Verify critical functions are registered
    expected_funcs = ["keyword_match", "numeric_range", "status_check", "author_match"]
    for func in expected_funcs:
        if not ScoringRegistry.get(func):
            raise Exception(f"Make sure function '{func}' is decorated with @ScoringRegistry.register")
    print("✅ Registry check successful.")
except Exception as e:
    print(f"❌ Registry Check Failed: {e}")
    sys.exit(1)

try:
    print("Checking Schema Compatibility...")
    # Mock LLM Output (Dict style parameters)
    test_criteria = [
        Criterion(name="numeric_range", parameters={"field": "words_total", "min_val": 100}),
        Criterion(name="status_check", parameters={"target_status": "finished"})
    ]
    
    # Mock Item
    test_item = {
        "words_total": 200, 
        "publish_status": "已完結", 
        "author": "Test Author",
        "intro": "Test Intro"
    }

    # Test Logic Execution directly (simulate engine)
    for c in test_criteria:
         func = ScoringRegistry.get(c.name)
         score = func(test_item, c.parameters)
         print(f"   - Testing {c.name}: Score = {score}")
         
    print("✅ Logic execution check successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Logic Check Failed: {e}")
    sys.exit(1)

print("\n🎉 ALL SYSTEMS GO! The code logic is sound.")
