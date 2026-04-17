# Tag Template Mapping Experiment

這份實驗不再評估「同義詞彼此距離」的 intrinsic geometry，而是直接評估你的實際任務：

- 給定 LLM 產生的標籤文字
- 用 embedding 映射到系統中真實存在的 canonical tags
- 比較不同 template 對 canonical tag 端 embedding 的影響

## 任務定義

- 評估集來源: `data/experiments/tag_template/datasets/tag_template_eval_dataset.json`
- 候選 canonical tags: `data/all_tags.json`
- 查詢 query: 評估集中的同義詞 / 別名
- 正解 target: 每個 query 對應的 canonical label

腳本會固定查詢端 template，並只改變候選 canonical tag 端的 template。
這樣比較能隔離出「candidate label embedding 寫法」對映射效果的影響。
目前預設模板池已經整理成一批偏高品質的中長句模板，主打作品內容、題材、要素、特色等內容導向語氣，並排除較偏 meta 的「推薦/標記/描述方式」句型。

## 評估方式

對每個 query：

1. 將 query 套入固定 query template
2. 將所有 canonical tags 分別套入候選 template
3. 計算 query 與所有 canonical tags 的 cosine similarity
4. 依相似度排序，觀察正解 canonical tag 的排名

## 指標

- `Top-1 Accuracy`: 第一名是否就是正確 canonical tag
- `Top-5 Accuracy`: 前五名是否包含正確 canonical tag
- `MRR`: 正解排名的倒數平均值，越高越好
- `Macro-F1`: 以 Top-1 預測結果做多類別分類的 macro-F1
- `Micro-F1`: 以 Top-1 預測結果做多類別分類的 micro-F1

其中 `Top-1 Accuracy` 是最主要指標，因為最貼近「映射到正確 canonical tag」的成功率。

## 設定位置

設定集中在 [scripts/tag_template_experiment.py](/C:/dev/hybrid-reasoner/scripts/tag_template_experiment.py) 頂部的 `EXPERIMENT_CONFIG`。

常用設定有：

- `source`: 評估集
- `candidate_tags_source`: 候選 canonical tags 清單
- `query_template`: 固定的查詢 template
- `include_canonical_label_queries`: 是否把 canonical label 本身也納入 query
- `max_items`: 只取部分 label 做 pilot
- `query_task_type`: 查詢端 embedding task type
- `candidate_task_type`: 候選端 embedding task type
- `preview`: 是否只輸出資料設定，不實際做 embedding

## 執行方式

直接執行：

```powershell
python scripts/tag_template_experiment.py
```

如果只想先檢查設定：

- 把 `EXPERIMENT_CONFIG["preview"]` 改成 `True`
- 再執行同一個命令

如果要做小規模 pilot：

- 把 `EXPERIMENT_CONFIG["max_items"]` 設成較小的數字，例如 `10`

## 輸出

預設輸出到 `data/experiments/tag_template/runs/<model>/`：

- `dataset.json`: 這次 mapping 評估實際使用的 labels、queries 與 candidate tags
- `results.json`: 每個 template 的完整映射結果與逐 query 排名資料
- `summary.md`: template 排名與主要診斷摘要
- `embedding_cache/`: query 與 candidate embeddings 的 cache；下次執行只補算缺少的文字向量

## Cache 行為

embedding 會分成兩類快取：

- `query` cache: 固定 query template 下的所有 query phrases
- `candidate` cache: 每個候選 template 下的所有 canonical tags

如果 query template 沒變，query cache 可以跨不同 candidate template 共用。
如果某個 candidate template 已經算過，之後重跑只會補新的 tag 或新的文字。
