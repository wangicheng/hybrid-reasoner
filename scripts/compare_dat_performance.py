
import json
import asyncio
from pathlib import Path
from datetime import datetime
from src.eval.generate_run import RunGenerator
from src.config import settings

async def run_comparison():
    # 讀取測試查詢
    queries_path = Path("data/experiments/queries.json")
    if not queries_path.exists():
        print("Error: queries.json not found!")
        return
    
    with open(queries_path, "r", encoding="utf-8") as f:
        sample_queries = json.load(f)

    # 建立輸出目錄
    batch_name = datetime.now().strftime("dat_comparison_%Y%m%d_%H%M%S")
    output_folder = Path("data/experiments/runs") / batch_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1. 執行 DAT OFF (Baseline)
    print("\n>>> 🚀 Running Trial: DAT OFF (Static Alpha = 0.5)")
    settings.ENABLE_DAT = False
    generator_off = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
    generator_off.generate_run(
        queries_config=sample_queries,
        engine_name="baseline_dat_off",
        output_dir=output_folder,
        fusion_strategy="weighted",
        semantic_weight=0.5, # 當 DAT 關閉時，內部其實會用 settings.DAT_DEFAULT_ALPHA
        enable_bm25=True
    )

    # 2. 執行 DAT ON (Dynamic Alpha)
    print("\n>>> 🚀 Running Trial: DAT ON (LLM Dynamic Tuning)")
    settings.ENABLE_DAT = True
    generator_on = RunGenerator(k_per_engine=10, model_id="gemma-4-31b-it")
    generator_on.generate_run(
        queries_config=sample_queries,
        engine_name="experiment_dat_on",
        output_dir=output_folder,
        fusion_strategy="weighted",
        enable_bm25=True
    )

    print(f"\n✅ Comparison run finished! Results saved in: {output_folder}")
    print("Next step: Use src/eval/ir_metrics.py to calculate nDCG and compare.")

if __name__ == "__main__":
    asyncio.run(run_comparison())
