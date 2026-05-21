import json
import os
from scipy.stats import ttest_rel
from itertools import combinations
import pandas as pd
from collections import defaultdict

def load_per_query_data(file_path):
    """從 per-query JSON 檔案中載入所有模型的數據"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_latency_data(base_run_path):
    """從原始 run 檔案中提取延遲數據"""
    latency_data = defaultdict(list)
    files = [f for f in os.listdir(base_run_path) if f.startswith('gemma4_default_parser_bm25_on_cl') and f.endswith('.json')]
    
    for file in files:
        try:
            model_name = f"cl{file.split('_cl')[1].split('.json')[0]}"
            file_path = os.path.join(base_run_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                run_data = json.load(f)
            
            # 確保查詢順序與評估時一致
            run_data.sort(key=lambda x: x['query_id'])
            
            for item in run_data:
                latency = item.get('parse_metadata', {}).get('latency_ms', 0.0)
                latency_data[model_name].append({'query_id': item['query_id'], 'Latency': latency})
        except (IndexError, FileNotFoundError) as e:
            print(f"讀取延遲數據時出錯: {e}")
            continue
    return latency_data

def run_significance_analysis(per_query_data, latency_data, metrics_to_analyze, k=10):
    """對指定指標進行成對 t-檢定並返回 Markdown 格式的報告"""
    
    md_report = ""
    
    # 整理模型名稱並排序
    engine_names = list(per_query_data.keys())
    model_names = sorted([name.split('_on_')[1] for name in engine_names], key=lambda x: int(x.replace('cl', '')))

    for metric in metrics_to_analyze:
        md_report += f"## 指標分析: {metric}\n\n"
        
        if metric == 'Latency':
            md_report += "延遲 (Latency) 指標衡量處理每個查詢所需的平均時間，單位為毫秒（ms）。**數值越低越好**。\n\n"
        else:
            md_report += f"{metric} 指標衡量檢索結果的質量。**數值越高越好**。\n\n"

        results = []
        for model1_prefix, model2_prefix in combinations(model_names, 2):
            model1_name = f"gemma4_default_parser_bm25_on_{model1_prefix}"
            model2_name = f"gemma4_default_parser_bm25_on_{model2_prefix}"

            if metric == 'Latency':
                scores1 = [item['Latency'] for item in latency_data.get(model1_prefix, [])]
                scores2 = [item['Latency'] for item in latency_data.get(model2_prefix, [])]
            else:
                metric_key = f"{metric.lower()}_at_{k}"
                scores1 = [item.get(metric_key, 0.0) for item in per_query_data.get(model1_name, [])]
                scores2 = [item.get(metric_key, 0.0) for item in per_query_data.get(model2_name, [])]

            if not scores1 or not scores2 or len(scores1) != len(scores2):
                print(f"警告：模型 {model1_prefix} 或 {model2_prefix} 的 {metric} 數據不完整，已跳過。")
                continue

            mean1 = sum(scores1) / len(scores1)
            mean2 = sum(scores2) / len(scores2)
            
            t_stat, p_value = ttest_rel(scores1, scores2)
            
            results.append({
                "Model 1": model1_prefix,
                "Model 2": model2_prefix,
                f"Mean {metric} 1": f"{mean1:.4f}",
                f"Mean {metric} 2": f"{mean2:.4f}",
                "p-value": f"{p_value:.4f}",
                "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
            })
        
        if not results:
            md_report += f"**注意：找不到指標 '{metric}' 的有效數據，無法進行分析。**\n\n"
            continue

        df = pd.DataFrame(results)
        md_report += df.to_markdown(index=False)
        md_report += "\n\n"

        # 自動產生簡要分析
        significant_changes = df[df['Significant (p < 0.05)'] == 'Yes']
        if not significant_changes.empty:
            last_sig_change = significant_changes.iloc[-1]
            model1 = last_sig_change['Model 1']
            model2 = last_sig_change['Model 2']
            
            non_sig_changes = df[(df['Significant (p < 0.05)'] == 'No') & (df['Model 1'] == model2)]
            
            md_report += f"**分析摘要 ({metric}):**\n"
            md_report += f"- 從 `{model1}` 調整至 `{model2}` 觀察到顯著的性能變化 (p < 0.05)。\n"
            
            if not non_sig_changes.empty:
                first_nonsig_model2 = non_sig_changes.iloc[0]['Model 2']
                md_report += f"- 然而，從 `{model2}` 繼續增加到 `{first_nonsig_model2}` 時，性能變化**不具有統計顯著性** (p >= 0.05)。\n"
                md_report += f"- **結論**: 對於 `{metric}` 指標，`{model2}` 似乎是成本效益較高的設定點。\n\n"
            else:
                md_report += f"- **結論**: 在測試範圍內，增加候選數量對 `{metric}` 指標持續帶來顯著影響。\n\n"
        else:
            md_report += f"**分析摘要 ({metric}):**\n- 在所有模型對比中，`{metric}` 指標的變化均**不具有統計顯著性**。\n\n"
            
    return md_report

def main():
    """主函數，讀取數據、執行分析並產生報告"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    per_query_metrics_path = os.path.join(project_root, 'data', 'experiments', 'eval_results', 'cl_ablation_per_query_metrics.json')
    base_run_path = os.path.join(project_root, 'data', 'experiments', 'runs', 'batch_cl_ablation_all')

    # 載入數據
    per_query_data = load_per_query_data(per_query_metrics_path)
    latency_data = get_latency_data(base_run_path)

    # 定義要分析的指標
    metrics_to_analyze = ['NDCG', 'Precision', 'Recall', 'F1-score', 'Latency']
    
    # 執行分析
    analysis_report = run_significance_analysis(per_query_data, latency_data, metrics_to_analyze, k=10)

    # --- 產生完整的 Markdown 報告 ---
    model_names_sorted = sorted([name.split('_on_')[1] for name in per_query_data.keys()], key=lambda x: int(x.replace('cl', '')))
    md_report = f"# Reranker候選數量消融實驗統計分析\n\n"
    md_report += f"分析日期: {pd.to_datetime('today').strftime('%Y-%m-%d')}\n"
    md_report += f"實驗數據目錄: `data/experiments/runs/batch_cl_ablation_all`\n"
    md_report += f"分析模型: {', '.join(model_names_sorted)}\n\n"
    md_report += "本報告旨在透過**配對t檢定 (Paired t-test)**，分析不同Reranker候選數量（cl20至cl200）對模型各項評估指標（NDCG, Precision, Recall, F1-score @10, Latency）的影響是否具有統計顯著性。\n\n"
    md_report += "p-value < 0.05 通常被認為差異是**顯著的**，代表觀測到的性能變化很可能不是由隨機因素造成的。\n\n"
    md_report += analysis_report

    # 將報告寫入檔案
    report_path = os.path.join(project_root, 'cl_ablation_significance_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
        
    print(f"分析報告已成功生成於: {report_path}")
    print("\n--- 報告內容預覽 ---")
    print(md_report)


if __name__ == "__main__":
    main()
