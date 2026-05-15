**執行資訊**

| 項目 | 值 |
| --- | --- |
| 批次 | data/experiments/runs/batch_20260512_235455 |
| 模型 | gemma-4-31b-it |
| 解析器變體 | semantic_sections_v3_taglite_three_call |
| Rerank | PermSC（5 次隨機排列），Kemeny-Young 近似（上限 5s），top_k=10，candidate_limit=100 |
| 融合權重 | semantic=0.4，attribute=0.6 |
| 查詢數 | 24 |

**評估指標**

| 模式 | Avg@10 | Good@10 | Strong@10 | Best@10 |
| --- | --- | --- | --- | --- |
| no-strict | 0.9833 | 32.9% | 28.8% | 2.9583 |
| strict-only | 1.9600 | 55.8% | 38.7% | 2.6313 |

**解析器遙測**

| 分支 | 成功率 | 首次成功率 | 平均請求數 | 平均重試數 | 平均延遲 (ms) |
| --- | --- | --- | --- | --- | --- |
| semantic_understanding | 100.0% (24/24) | 75.0% | 1.38 | 0.38 | 28826.14 |
| tag_projection | 100.0% (24/24) | 91.7% | 1.08 | 0.08 | 6139.29 |
| structured | 100.0% (24/24) | 83.3% | 1.17 | 0.17 | 9002.06 |

**深度分析**

1. **品質訊號分佈與規則層影響**
	- strict-only 在 Avg@10 與 Good@10 明顯高於 no-strict，顯示嚴格規則層在此批次承擔了主要品質訊號。
	- Best@10 在 no-strict 略高，可能代表規則收斂後對極端高分項目有壓縮效果，或規則過濾把部分高分項目移出候選。
	- 解讀方向：若目標偏向整體穩定度與可解釋性，strict-only 的提升是合理；若更重視「極端命中」，可考慮在規則層加入柔性回補策略或提高候選池。

2. **穩定性與失敗模式**
	- 全部查詢成功，顯示本輪 pipeline 穩定；parser 分支的重試集中於 semantic_understanding 與 structured。
	- semantic_understanding 的平均重試高於其他分支，與供應商 500/503 波動一致，屬於外部錯誤主導的重試成本。
	- 建議：若要降低整體成本，可在 semantic_understanding 加上更積極的短路規則（例如快速 fallback），或提高 key 池的調度密度以降低單 key 堵塞。

3. **延遲瓶頸**
	- semantic_understanding 平均延遲 28.8s，顯著高於 tag_projection 與 structured，仍是端到端延遲主要來源。
	- 本次 reranker 的改動不影響 parser 的延遲分佈，因此 latency 優化應優先集中於 semantic_understanding。
	- 建議：若產品需縮短等待時間，先縮短 semantic_understanding 的 timeout 或改為更輕量的 prompt/模型，通常比調整 rerank 更有效。

4. **PermSC + Kemeny 效果的可驗證性**
	- 本批次未提供關閉 rerank 的 baseline，因此無法量化本次 reranker 改動對 Avg@10/Good@10 的邊際影響。
	- 建議：以相同 queries 和模型跑一輪 `rerank=false`，形成 A/B，比對 no-strict 與 strict-only 指標，才能做因果判斷。
	- 若要評估召回提升，可提高 top_k 或 candidate_limit，再對比 Good@10 與 Best@10 的變化趨勢。

**產出物**

| 類型 | 路徑/指令 |
| --- | --- |
| Run JSON | data/experiments/runs/batch_20260512_235455/gemma4_default_parser.json |
| 指標指令 | python -m src.eval.metrics --experiment-dir data/experiments/runs/batch_20260512_235455 |
| 解析器遙測指令 | python -m src.eval.analyze_parser_runs --experiment-dir data/experiments/runs/batch_20260512_235455 |
