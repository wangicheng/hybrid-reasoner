import asyncio
import sys
import os
import time
import json
import csv
from dotenv import load_dotenv

# Setup Environment
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.engine import HybridEngine

# 測試案例集 (Benchmark Suite)
TEST_CASES = [
    {
        "id": "T1_EXACT",
        "category": "精確搜尋",
        "query": "詭秘之主",
        "description": "測試對已知書名的精確命中能力",
        "expected_behavior": "第一名應為《詭秘之主》且分數極高"
    },
    {
        "id": "T2_GENANTIC",
        "category": "語意/主題",
        "query": "主角是廢柴但獲得系統逆襲的故事",
        "description": "測試對抽象劇情的理解 (非關鍵字匹配)",
        "expected_behavior": "應推薦含有「系統」、「廢柴流」、「無敵流」的回應"
    },
    {
        "id": "T3_CONSTRAINT", 
        "category": "硬性條件",
        "query": "字數超過200萬字的完結仙俠小說",
        "description": "測試對元數據 (Metadata) 過濾的準確性",
        "expected_behavior": "結果需全為「完結」且「字數>200萬」且分類為「仙俠」"
    },
    {
        "id": "T4_VAGUE",
        "category": "模糊/風格",
        "query": "看起來很輕鬆好笑的",
        "description": "測試對風格標籤 (Tags) 或簡介氛圍的捕捉",
        "expected_behavior": "應包含「搞笑」、「輕鬆」、「迪化」等標籤"
    },
    {
        "id": "T5_COMPLEX",
        "category": "混合複雜",
        "query": "不要後宮，單女主或無女主的科幻文",
        "description": "測試負向條件與多重篩選",
        "expected_behavior": "應過濾掉「後宮」標籤，優先展示「單女主」或「無CP」"
    }
]

async def run_benchmark():
    print("="*60)
    print("🚀 Hybrid Reasoner Search Benchmark Tool")
    print("="*60)
    print(f"Initializing Engine... (Model: {os.getenv('LLM_MODEL_ID', 'Default')})")
    
    try:
        engine = HybridEngine()
    except Exception as e:
        print(f"Engine Init Failed: {e}")
        return

    results_data = []

    for test in TEST_CASES:
        print(f"\n[Running {test['id']}] {test['query']}...")
        start_time = time.time()
        
        try:
            # 執行搜尋
            response = await engine.search(test['query'], limit=3)
            duration = time.time() - start_time
            
            top_result = response['results'][0] if response['results'] else None
            
            # 收集數據
            row = {
                "Case ID": test['id'],
                "Query": test['query'],
                "Time (s)": round(duration, 2),
                "Top 1 Title": top_result['item']['name'] if top_result else "NULL",
                "Top 1 Score": f"{top_result['score']:.4f}" if top_result else "0",
                "Top 1 Tags": ",".join(top_result['item'].get('tags', [])[:3]) if top_result else "",
                "Result Count": len(response['results']),
                "Interpretation": response['explanation'][:50] + "..." if 'explanation' in response else "N/A"
            }
            results_data.append(row)
            
            # Print precise details for user validation
            print(f"  ⏱️ Time: {duration:.2f}s")
            if top_result:
                print(f"  🏆 Top 1: {top_result['item']['name']} (Score: {top_result['score']:.2f})")
                print(f"  🏷️ Tags: {top_result['item'].get('tags', [])}")
                print(f"  📝 Intro Snippet: {top_result['item'].get('intro', '')[:60]}...")
            else:
                print("  ❌ No results found.")

        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            results_data.append({
                "Case ID": test['id'],
                "Query": test['query'],
                "Time (s)": "ERROR",
                "Top 1 Title": str(e),
                "Top 1 Score": "0", "Top 1 Tags": "", "Result Count": 0, "Interpretation": ""
            })

    # Output CSV Report
    csv_file = "benchmark_report.csv"
    keys = results_data[0].keys()
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_data)

    print("\n" + "="*60)
    print(f"✅ Benchmark Complete! Report saved to '{csv_file}'")
    print("="*60)
    print("請使用相同的 Query 在競品 App (如起點、番茄) 中搜尋，並比較「Top 1 Title」是否更符合預期。")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_benchmark())
