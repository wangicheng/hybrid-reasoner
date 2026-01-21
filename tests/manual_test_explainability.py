import sys
import os

# 將專案根目錄加入路徑，確保可以 import src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.core.engine import HybridEngine

def main():
    print("正在初始化 HybridEngine...")
    try:
        engine = HybridEngine()
    except Exception as e:
        print(f"初始化失敗: {e}")
        print("請確認您的環境變數 (.env) 與 Qdrant 服務是否設定正確。")
        return

    # 測試查詢
    query = "想要看輕鬆好笑的奇幻冒險，主角最好不要太笨"
    print(f"\n測試查詢: {query}")
    print("-" * 50)
    
    try:
        # 執行搜尋
        result_data = engine.search(query, limit=5)
        results = result_data.get("results", [])
        
        print(f"搜尋完成，共找到 {len(results)} 筆結果。\n")
        
        # 檢查前幾名的解釋
        for i, res in enumerate(results):
            rank = i + 1
            item = res.get('item', {})
            name = item.get('name', 'Unknown')
            score = res.get('score', 0.0)
            explanation = res.get('explanation')
            
            print(f"第 {rank} 名: {name} (Score: {score:.2f})")
            
            if explanation:
                print(f"💡 解析: {explanation}")
            else:
                # 預期第 4 名以後沒有解釋
                if rank <= 3:
                    print("⚠️ 警告: 前 3 名應該要有解釋，但此處為 None")
                else:
                    print("Explainability: (無 - 僅前 3 名生成，符合預期)")
            
            print("-" * 30)
            
    except Exception as e:
        print(f"搜尋過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
