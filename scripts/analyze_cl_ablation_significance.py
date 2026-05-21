import json
import os
from scipy.stats import ttest_rel
from itertools import combinations
import pandas as pd
from collections import defaultdict

def load_scores(file_path):
    """從 JSON 檔案中載入每個查詢的各項指標分數"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores = defaultdict(list)
    for item in data:
        # 這裡的鍵名需要與 src.eval.metrics 的輸出對應
        # 由於 metrics.py 直接打印標準輸出，我們需要先運行它並捕獲結果
        # 或者修改 metrics.py 將結果存為 json
        # 目前，我們先假設一個理想的 json 結構，後續再調整
        eval_metrics = item.get('evaluation_metrics', {}) # 假設評估結果存在這裡
        scores['NDCG'].append(eval_metrics.get('ndcg_at_10', 0.0))
        scores['Precision'].append(eval_metrics.get('precision_at_10', 0.0))
        scores['Recall'].append(eval_metrics.get('recall_at_10', 0.0))
        scores['F1-score'].append(eval_metrics.get('f1_at_10', 0.0))
        
        # 提取延遲
        parse_metadata = item.get('parse_metadata', {})
        latency = parse_metadata.get('latency_ms', 0.0)
        scores['Latency'].append(latency)
        
    return scores

def run_analysis(model_data, metric_name):
    """對指定指標進行成對 t-檢定並返回結果"""
    results = []
    model_names = sorted(model_data.keys(), key=lambda x: int(x.replace('cl', '')))

    for model1_name, model2_name in combinations(model_names, 2):
        scores1 = [d[metric_name] for d in model_data[model1_name]]
        scores2 = [d[metric_name] for d in model_data[model2_name]]
        
        if not scores1 or not scores2 or len(scores1) != len(scores2):
            continue

        mean1 = sum(scores1) / len(scores1)
        mean2 = sum(scores2) / len(scores2)
        
        t_stat, p_value = ttest_rel(scores1, scores2)
        
        results.append({
            "Model 1": model1_name,
            "Model 2": model2_name,
            f"Mean {metric_name} 1": f"{mean1:.4f}",
            f"Mean {metric_name} 2": f"{mean2:.4f}",
            "p-value": f"{p_value:.4f}",
            "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
        })
    return pd.DataFrame(results)

def main():
    """主函數，執行成對 t-檢定並產生 Markdown 報告"""
    # 這是個錯誤的路徑，因為 metrics.py 的輸出不是 json
    # 我們需要先解決數據源問題
    print("錯誤：本腳本設計用來讀取包含評估指標的JSON檔案，")
    print("但 `src/eval/metrics.py` 產生的是文字報告。")
    print("需要先修改 `src/eval/metrics.py` 以輸出JSON格式的詳細評估結果。")
    # 後續的程式碼暫時無法正確執行

# --- 以下為修改 `src/eval/metrics.py` 後才能使用的程式碼 ---

def main_after_fix():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # 假設 metrics.py 的輸出會存在 'data/experiments/eval_results'
    base_path = os.path.join(project_root, 'data', 'experiments', 'eval_results', 'batch_cl_ablation_all')
    
    # 獲取所有相關的 JSON 檔案
    files = [f for f in os.listdir(base_path) if f.startswith('gemma4_default_parser_bm25_on_cl') and f.endswith('.json')]
    
    # 建立模型名稱與檔案路徑的對應字典
    model_data = {}
    for file in files:
        # 從檔名中提取模型名稱，例如 'cl20'
        try:
            model_name = file.split('_cl')[1].split('.json')[0]
            model_name = f"cl{model_name}"
            file_path = os.path.join(base_path, file)
            model_data[model_name] = load_scores(file_path)
        except IndexError:
            print(f"無法從檔名 {file} 中解析模型名稱，已跳過。")
            continue

    # 確保所有模型都有相同數量的查詢分數
    num_queries = -1
    for model, scores in model_data.items():
        if num_queries == -1:
            num_queries = len(scores)
        elif len(scores) != num_queries:
            print("錯誤：各模型間的查詢數量不一致，無法進行配對 t-檢定。")
            return
            
    print(f"成功載入 {len(model_data)} 個模型，共 {num_queries} 個查詢的 NDCG 分數。\n")

    # 進行成對 t-檢定
    results = []
    model_names = sorted(model_data.keys(), key=lambda x: int(x.replace('cl', '')))

    for model1_name, model2_name in combinations(model_names, 2):
        scores1 = model_data[model1_name]
        scores2 = model_data[model2_name]
        
        mean1 = sum(scores1) / len(scores1)
        mean2 = sum(scores2) / len(scores2)
        
        # 執行配對 t-檢定
        t_stat, p_value = ttest_rel(scores1, scores2)
        
        results.append({
            "Model 1": model1_name,
            "Model 2": model2_name,
            "Mean NDCG 1": f"{mean1:.4f}",
            "Mean NDCG 2": f"{mean2:.4f}",
            "p-value": f"{p_value:.4f}",
            "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
        })

    # 使用 pandas 以表格形式美化輸出
    df = pd.DataFrame(results)
    print("成對 t-檢定結果 (Paired t-test Results):")
    print(df.to_string(index=False))
    
    print("\n---")
    print("如何解讀 p-value:")
    print("p-value 代表『觀測到的差異純屬偶然』的機率。")
    print(" - p < 0.05: 差異具有統計顯著性。我們有信心認為兩個模型的表現確實不同。")
    print(" - p >= 0.05: 差異不顯著。我們沒有足夠的證據說一個模型真的優於另一個。")


if __name__ == "__main__":
    main()
