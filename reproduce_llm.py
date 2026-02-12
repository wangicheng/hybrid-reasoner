import sys
import os
import traceback
import json

# Setup path to include project root
sys.path.insert(0, os.path.abspath("c:\\Users\\sean\\桌面\\mlgb_project\\hybrid-reasoner-1.2.1ver"))

# Load .env manually
env_path = os.path.abspath("c:\\Users\\sean\\桌面\\mlgb_project\\hybrid-reasoner-1.2.1ver\\.env")
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

from src.core.llm import parse_query
from src.core.engine import HybridEngine

# Initialize engine once
print("Initializing HybridEngine...")
try:
    engine = HybridEngine()
except Exception as e:
    print(f"Failed to init engine: {e}")
    engine = None

def test_query(query: str, model_id: str, f_out):
    f_out.write(f"\n{'#'*20} Testing Query: '{query}' with Model: {model_id} {'#'*20}\n")
    os.environ["LLM_MODEL_ID"] = model_id
    
    try:
        # 1. Test Parse
        f_out.write("[Step 1] Parsing Query...\n")
        result = parse_query(query)
        f_out.write("Parse Result:\n")
        if hasattr(result, 'model_dump_json'):
             f_out.write(result.model_dump_json(indent=2))
        else:
             f_out.write(result.json(indent=2))
        f_out.write("\n")
        
        # 2. Test Search
        if engine:
            f_out.write("[Step 2] Executing Search...\n")
            try:
                search_result = engine.search(query, limit=5)
                f_out.write(f"Search Results Found: {len(search_result.get('results', []))}\n")
                f_out.write("Top Result Breakdown:\n")
                if search_result.get('results'):
                     top_res = search_result['results'][0]
                     f_out.write(json.dumps(top_res.get('breakdown', []), indent=2, ensure_ascii=False))
                     f_out.write(f"\nTotal Score: {top_res.get('score')}\n")
                     f_out.write(f"Vector Score: {top_res.get('vector_score')}\n")
                else:
                     f_out.write("No results found.\n")
            except Exception as se:
                f_out.write(f"Search Error: {se}\n")
                f_out.write(traceback.format_exc())
        else:
            f_out.write("[Step 2] Skipped (Engine not initialized)\n")

        return result
    except Exception as e:
        f_out.write(f"\nError in Parse: {str(e)}\n")
        f_out.write(traceback.format_exc())
        return None

if __name__ == "__main__":
    queries = [
        "語意檢索： 使用向量搜尋相近內容",
        "找一本關於魔法的奇幻小說",
        "字數超過一百萬字的完結小說"
    ]

    models = ["gemma-3-27b-it", "gemini-3-flash-preview"]
    
    output_file = "reproduce_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for model_id in models:
            for q in queries:
                test_query(q, model_id, f)
            f.write("\n" + "="*40 + "\n")
