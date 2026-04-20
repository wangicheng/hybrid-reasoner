# Tag 提取 A/B 實驗報告（Gemma4 + Gemini Embedding 2）

## 1. 實驗目的
本實驗聚焦於「標籤提取準確度」，驗證在相同提取模型下，加入 Tag 描述（with description）是否能提升提取品質。

- 不評估書籍推薦品質
- 僅評估標籤提取行為與 taxonomy 對齊品質

## 2. 實驗設定

| 項目 | 值 |
| --- | --- |
| Query 數量 | 24 |
| 含 required_tags 的 query | 14 |
| 含 blocked_tags 的 query | 10 |
| 提取模型 | models/gemma-4-31b-it |
| 映射模型 | models/gemini-embedding-2-preview |
| 評分用預測欄位 | mapped_pred_tags |
| A/B 條件 A | no_tag_description |
| A/B 條件 B | with_tag_description |

## 3. 實驗方法

### 3.1 流程
1. 對同一批 24 筆 query 分別執行兩組提取：
   - A 組：不提供 tag 描述
   - B 組：提供 tag 描述
2. 每筆 query 取得提取標籤後，對 taxonomy 外詞彙進行 embedding 映射（Gemini Embedding 2），得到 mapped_pred_tags。
3. 以 mapped_pred_tags 與標註欄位（required_tags、blocked_tags）計算指標。
4. 比較 A/B 各項指標與逐題差異。

### 3.2 指標定義
- parse_success_rate：解析成功比率
- required_exact_cover_rate：required_tags 是否被完整覆蓋（允許多預測）
- blocked_clean_rate：是否未命中 blocked_tags
- required_micro_precision / recall / F1：required task 的 micro 指標
- required_macro_precision / recall / F1：required task 的 macro 指標
- required_exact_match_rate：預測集合是否與 required_tags 完全一致
- raw_outside_taxonomy_rate：原始提取標籤中 taxonomy 外詞比例
- avg_pred_tag_count：平均每題預測標籤數

## 4. 核心指標比較（A/B）

| 指標 | No Description | With Description | Delta (With - No) |
| --- | ---: | ---: | ---: |
| parse_success_rate | 1.000000 | 1.000000 | +0.000000 |
| required_exact_cover_rate | 0.928571 | 1.000000 | +0.071429 |
| blocked_clean_rate | 1.000000 | 1.000000 | +0.000000 |
| required_micro_precision | 0.296875 | 0.289855 | -0.007020 |
| required_micro_recall | 0.950000 | 1.000000 | +0.050000 |
| required_micro_f1 | 0.452381 | 0.449438 | -0.002943 |
| required_macro_precision | 0.048065 | 0.059350 | +0.011285 |
| required_macro_recall | 0.093750 | 0.104167 | +0.010417 |
| required_macro_f1 | 0.061508 | 0.072619 | +0.011111 |
| required_exact_match_rate | 0.071429 | 0.000000 | -0.071429 |
| raw_outside_taxonomy_rate | 0.018519 | 0.000000 | -0.018519 |
| avg_pred_tag_count | 4.500000 | 4.875000 | +0.375000 |

## 5. 絕對數據（各組原始統計）

### 5.1 No Description

| 項目 | 數值 |
| --- | ---: |
| total_queries | 24 |
| parse_success | 24 |
| required_query_count | 14 |
| required_exact_cover_hits | 13 |
| blocked_query_count | 10 |
| blocked_clean_hits | 10 |
| raw_pred_tag_count | 108 |
| raw_outside_taxonomy_tag_count | 2 |

### 5.2 With Description

| 項目 | 數值 |
| --- | ---: |
| total_queries | 24 |
| parse_success | 24 |
| required_query_count | 14 |
| required_exact_cover_hits | 14 |
| blocked_query_count | 10 |
| blocked_clean_hits | 10 |
| raw_pred_tag_count | 117 |
| raw_outside_taxonomy_tag_count | 0 |

## 6. 逐題差異重點

### 6.1 Required cover 改善題
- 僅 q13 發生改善：False -> True

q13 重點：
- required_tags: [歡樂向]
- No Description 預測： [搞笑吐槽, 腦洞, 歡樂, 異世界, 冒險]
- With Description 預測： [搞笑吐槽, 歡樂, 歡樂向, 腦洞]
- 觀察：with description 成功補到關鍵 required tag「歡樂向」，因此 cover 提升。

### 6.2 Exact match 下降來源
- No Description 的 required_exact_match_qids： [q24]
- With Description 的 required_exact_match_qids： []

q24 重點：
- required_tags: [戀愛, 戰爭]
- No Description： [戰爭, 戀愛]（完全一致）
- With Description： [戰爭, 戀愛, 日本輕小說]（多預測 1 個 tag）
- 觀察：with description 提升召回傾向，但也提高了過度預測機率，壓低 exact match。

### 6.3 Taxonomy 外詞處理
- No Description 出現 taxonomy 外詞共 2 次，均為「智鬥」（q17、q18）
- With Description 外詞次數為 0
- 觀察：with description 對詞彙規範化有幫助，降低 taxonomy 外輸出。

## 7. 標籤分佈變化（With 相對 No）

### 7.1 增加較多的標籤
- 治癒 (+2)
- 原生幻想 (+2)
- 人外 (+2)
- 智斗 (+2)

### 7.2 減少的標籤
- 異世界 (-1)
- 冒險 (-1)
- 戰鬥 (-1)
- 奇幻 (-1)
- 穿越 (-1)
- 日本輕小說 (-1)
- 鬥智 (-1)
- 熱血 (-1)
- other (-1)

## 8. 分析與結論

### 8.1 整體判讀
- with description 在「關鍵標籤覆蓋」與「taxonomy 對齊」方面表現更好：
  - required_exact_cover_rate +0.071429
  - raw_outside_taxonomy_rate -0.018519
- 同時，with description 讓模型更傾向多給標籤（avg_pred_tag_count +0.375），導致：
  - micro precision 略降
  - required_exact_match_rate 下降

### 8.2 實務意義
- 若目標是「不要漏掉必要標籤」，with description 更合適。
- 若目標是「預測集合要非常精準且短」，需要再加一道後處理（例如 top-k 或置信度門檻）來控制過度預測。

### 8.3 建議後續
1. 固定 with description，加入預測標籤數量上限（例如 3-5）再跑同批評估。
2. 對 required task 分類做分層分析（單 required vs 多 required），確認 precision 下滑集中在哪類 query。
3. 針對易混淆對（如 智鬥/智斗、歡樂/歡樂向）建立規範映射表，降低語義近鄰誤差。

## 9. 輸入與結果檔案

- 單檔原始結果：
  - data/experiments/tag_description_ab_gemma4_31b_embed2/single_file_report_all_data.json
- 本報告：
  - data/experiments/tag_description_ab_gemma4_31b_embed2/final_analysis_report.md
