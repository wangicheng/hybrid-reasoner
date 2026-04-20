# Tag Description 12條修改後實驗報告（Gemma4 + Gemini Embedding 2）

## 1. 本次變更
已在 data/tag_descriptions.json 調整以下 12 個 tag 的描述邊界（加入觸發條件與排除條件）：

- 青春
- 歡樂
- 歡樂向
- 治癒
- 溫馨
- 異世界
- 奇幻
- 魔法
- 原生幻想
- 人外
- 日本輕小說
- 龍傲天

## 2. 測試設定

| 項目 | 值 |
| --- | --- |
| Query 數量 | 24 |
| 提取模型 | models/gemma-4-31b-it |
| 映射模型 | models/gemini-embedding-2-preview |
| 映射後預測欄位 | mapped_pred_tags |
| 輸出目錄 | data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2 |

註：本次採用 rerun2 結果（全程 Gemma4 成功），避免首輪 API 500 fallback 造成偏差。

## 3. 本次 A/B 結果（同一次 rerun2）

| 指標 | No Description | With Description | Delta (With - No) |
| --- | ---: | ---: | ---: |
| parse_success_rate | 1.000000 | 1.000000 | +0.000000 |
| required_exact_cover_rate | 1.000000 | 1.000000 | +0.000000 |
| blocked_clean_rate | 1.000000 | 1.000000 | +0.000000 |
| required_micro_precision | 0.303030 | 0.317460 | +0.014430 |
| required_micro_recall | 1.000000 | 1.000000 | +0.000000 |
| required_micro_f1 | 0.465116 | 0.481928 | +0.016811 |
| required_macro_precision | 0.058656 | 0.054315 | -0.004340 |
| required_macro_recall | 0.104167 | 0.104167 | +0.000000 |
| required_macro_f1 | 0.071875 | 0.069444 | -0.002431 |
| required_exact_match_rate | 0.071429 | 0.071429 | +0.000000 |
| raw_outside_taxonomy_rate | 0.027273 | 0.000000 | -0.027273 |
| avg_pred_tag_count | 4.583333 | 4.583333 | +0.000000 |

## 4. 與修改前 baseline（with description）對照

baseline: data/experiments/tag_description_ab_gemma4_31b_embed2
new run: data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2

| 指標 | Baseline With Desc | New With Desc | Delta (New - Old) |
| --- | ---: | ---: | ---: |
| required_exact_cover_rate | 1.000000 | 1.000000 | +0.000000 |
| required_micro_precision | 0.289855 | 0.317460 | +0.027605 |
| required_micro_recall | 1.000000 | 1.000000 | +0.000000 |
| required_micro_f1 | 0.449438 | 0.481928 | +0.032490 |
| required_macro_precision | 0.059350 | 0.054315 | -0.005035 |
| required_macro_recall | 0.104167 | 0.104167 | +0.000000 |
| required_macro_f1 | 0.072619 | 0.069444 | -0.003175 |
| required_exact_match_rate | 0.000000 | 0.071429 | +0.071429 |
| raw_outside_taxonomy_rate | 0.000000 | 0.000000 | +0.000000 |
| avg_pred_tag_count | 4.875000 | 4.583333 | -0.291667 |

## 5. 重點解讀

1. 精度方向有改善：
- with description 的 micro precision 與 micro F1 都提升（+0.0276, +0.0325）。

2. 標籤數量收斂：
- avg_pred_tag_count 從 4.875 降到 4.583，代表過度擴張有所下降。

3. exact match 明顯修正：
- required_exact_match_rate 從 0.0 提升到 0.0714。
- 典型案例是 q24 不再多打一個「日本輕小說」。

4. macro 指標小幅下降：
- macro_precision / macro_f1 略降，代表少數 query 的標籤組合仍有偏移。

## 6. 變更後行為特徵（with desc 相對 baseline）

- 有變動的 query: 18 / 24
- 主要新增 tag（次數）：鬥智(+3)、歡樂向(+2)、異世界(+2)、熱血(+2)
- 主要移除 tag（次數）：歡樂(-3)、溫馨(-3)、冒險(-2)、日本輕小說(-2)、治癒(-2)

這顯示新的描述邊界確實在抑制泛化情緒詞與來源推測詞（例如日本輕小說），同時把部分樣本改判到更明確語義標籤（如歡樂向、鬥智）。

## 7. 檔案索引

- 新 A/B summary: data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2/summary.md
- 新 A/B comparison: data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2/gemini_tag_desc_comparison.json
- 新 with desc report: data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2/gemini_with_tag_desc_report.json
- baseline vs new 變動明細: data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2/with_desc_changes_vs_old.json
- baseline vs new 影響摘要: data/experiments/tag_description_ab_gemma4_31b_embed2_desc12_rerun2/desc12_impact_summary.json
