import sys
import os

# 將專案根目錄加入路徑，確保能 import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from src.core.engine import HybridEngine
import time

# --- 1. 定義黃金測試集 ---
TEST_QUERIES = [
    "字數超過200萬字的已完結小說",
    "主角智商在線，劇情佈局嚴謹",
    "已完結的網遊小說，但不要太舊"
]

def print_separator(char="-", length=80):
    print(char * length)

def analyze_scores():
    print("🚀 初始化 HybridEngine...")
    engine = HybridEngine()
    
    for query in TEST_QUERIES:
        print("\n")
        print_separator("=")
        print(f"🔍 測試查詢: 【 {query} 】")
        print_separator("=")

        # 執行搜尋
        start_time = time.time()
        # 這裡我們只取前 5 名來分析
        result = engine.search(query, limit=5)
        duration = time.time() - start_time

        # 1. 顯示 LLM 解析結果 (看看它把查詢轉成了什麼規則)
        print(f"🤖 LLM 解析意圖 (耗時 {duration:.2f}s):")
        for criteria in result['parsed_criteria']:
            # 兼容 Pydantic v1/v2 或 dict
            c_dict = criteria if isinstance(criteria, dict) else criteria.dict()
            name = c_dict.get('name')
            weight = c_dict.get('weight')
            params = c_dict.get('parameters', {})
            print(f"   - [規則] {name:<20} | 權重: {weight} | 參數: {params}")

        if not result['results']:
            print("❌ 無搜尋結果 (可能被 Filter 過濾光了)")
            continue

        print_separator("-")
        print(f"🏆 Top 5 排行榜與分數細節 (Score X-Ray):")
        
        # 2. 顯示詳細分數構成
        for rank, item in enumerate(result['results'], 1):
            book = item['item']
            final_score = item['score']
            vector_score = item.get('vector_score', 0.0)
            breakdown = item.get('breakdown', [])

            print(f"\n#{rank} [書名: {book.get('name', 'Unknown')}]")
            print(f"   📊 總分 (Final Score): {final_score:.6f}")
            print(f"   🧠 向量分數 (Vector):  {vector_score:.6f}  <-- 注意這個分數的變異程度")
            
            print(f"   📏 規則得分明細 (Rules):")
            for b in breakdown:
                c_name = b.get('criteria')
                w_score = b.get('weighted_score', 0)
                raw_score = b.get('raw_score', 0)
                weight = b.get('weight', 0)
                reason = b.get('reason', '')
                
                # 視覺化顯示：如果是 0 分用紅色(或標記)，滿分用綠色
                status_icon = "✅" if raw_score > 0 else "❌"
                if c_name == "semantic_similarity":
                    status_icon = "🧠"
                
                print(f"      {status_icon} {c_name:<20}: 原始 {raw_score:.4f} * 權重 {weight} = {w_score:.4f} | ({reason})")

        # 3. 變異數分析 (Variance Analysis)
        # 【修改】改為分析 normalized_score (正規化後的分數)，這才是真正影響排名的分數
        norm_scores = []
        for r in result['results']:
            for b in r.get('breakdown', []):
                if b['criteria'] == 'semantic_similarity':
                    norm_scores.append(b.get('normalized_score', 0.0))
                    break
        
        if norm_scores:
            v_max = max(norm_scores)
            v_min = min(norm_scores)
            v_spread = v_max - v_min
            print_separator(".")
            print(f"📉 統計洞察 (正規化後):")
            print(f"   正規化向量極距 (Spread): {v_spread:.4f} (Max: {v_max:.2f} - Min: {v_min:.2f})")
            
            if v_spread < 0.1:
                print("   ⚠️  警告: 即使正規化後，前五名的語意分數差異仍很小。")
                print("       可能原因：1. 這些書真的都很像 2. 向量拉伸區間 (0.35~0.65) 需要再調整。")
            else:
                print("   ✅ OK: 向量分數具備區分度，能有效影響排名。")

if __name__ == "__main__":
    analyze_scores()
