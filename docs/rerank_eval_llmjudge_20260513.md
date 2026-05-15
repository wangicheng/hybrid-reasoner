**簡報版本：LLM Judge 比較（6:4 / 7:3 / 8:2）**

**Slide 1｜實驗設定**

| 項目 | 內容 |
| --- | --- |
| LLM Judge | gemma-4-31b-it |
| Reranker | PermSC（5 次隨機排列）+ Kemeny-Young 近似（上限 10s） |
| rerank top_k | 10 |
| rerank candidate_limit | 100 |
| 解析器變體 | semantic_sections_v3_taglite_three_call |

**Slide 2｜批次與權重對照**

| 權重（tag:semantic） | 批次 | Run 名稱 |
| --- | --- | --- |
| 6:4 | data/experiments/runs/batch_20260512_235455 | gemma4_default_parser |
| 7:3 | data/experiments/runs/batch_20260513_184425 | gemma4_w73 |
| 8:2 | data/experiments/runs/batch_20260513_184425 | gemma4_w82 |

**Slide 3｜LLM Judge 覆蓋率（零缺漏保證）**

| 批次/權重組 | 需要評分候選數 | 已評分候選數 | 缺漏評分 |
| --- | --- | --- | --- |
| 6:4（batch_20260512_235455） | 240 | 240 | 0 |
| 7:3/8:2（batch_20260513_184425） | 368 | 368 | 0 |

**Slide 4｜LLM Judge 指標（no-strict）**

| 權重（tag:semantic） | Avg@10 | Good@10 | Strong@10 | Best@10 |
| --- | --- | --- | --- | --- |
| 6:4 | 2.6625 | 90.8% | 78.3% | 3.0000 |
| 7:3 | 2.7125 | 92.1% | 81.7% | 3.0000 |
| 8:2 | 2.6833 | 90.8% | 80.0% | 2.9167 |

**Slide 5｜LLM Judge 指標（strict-only）**

| 權重（tag:semantic） | Avg@10 | Good@10 | Strong@10 | Best@10 |
| --- | --- | --- | --- | --- |
| 6:4 | 1.9600 | 55.8% | 38.7% | 2.6313 |
| 7:3 | 1.8987 | 55.0% | 34.6% | 2.5688 |
| 8:2 | 1.8796 | 53.8% | 35.8% | 2.5938 |

**Slide 6｜主要結論（可直接放簡報）**

1. **no-strict：7:3 小幅最佳**
   - Avg@10/Good@10/Strong@10 皆高於 6:4 與 8:2。
   - 差距不大（1-2% 範圍），屬於微小優勢。
2. **strict-only：6:4 表現最佳**
   - Avg@10/Good@10/Strong@10 全面高於 7:3 與 8:2。
3. **8:2 不是穩定優勢**
   - Best@10 在 no-strict 低於 6:4 與 7:3；strict-only 的 Best@10 也未超出 6:4。
4. **注意批次差異**
   - 6:4 與 7:3/8:2 來自不同 batch；可比性存在 run-to-run 噪音，建議同批次再跑一次 6:4 做嚴格 A/B。

**Slide 7｜深度分析（以 LLM Judge 分數為核心）**

1. **為何 no-strict 分數普遍偏高**
   - LLM Judge 明確忽略硬性條件（狀態/字數），只看語意與題材，因此 no-strict 更容易被評高分。
   - 這代表「語意相關性」普遍不差，權重改動只能帶來微幅排序變化。

2. **strict-only 的結構性劣勢**
   - strict-only 會排除不符硬性條件的候選，但 LLM Judge 不計這些條件，造成評分天生吃虧。
   - 若產品目標是「語意 + 硬條件」，應以 no-strict + hard filter 的方式評估，而不是僅看 strict-only。

3. **權重效果為何看起來「不動」**
   - 24 筆 query 對權重差異的統計檢定力有限。
   - PermSC reranker 會吸收部分排序變化，使權重差異被稀釋。
   - 建議擴大 query 規模或提高候選難度（降低 top_k 或加入更接近的負例）以放大差異。

4. **Best@10 的解讀**
   - 7:3 與 6:4 都能達到 3.0，代表高峰命中仍可維持。
   - 8:2 的 Best@10 略低，可能代表過度偏重標籤帶來 top1 的語意偏移。

**Slide 8｜目前 reranker 架構（Kemeny-Young 近似流程）**

1. **候選生成與原始分數**
   - HybridEngine 先計算語意分數與標籤分數並加權融合，得到原始排序與 score。
   - 通過後置硬性篩選後，截取前 100 筆進入 reranker。

2. **PermSC 多次排列**
   - 針對 100 筆候選做 5 次「完全隨機排列」。
   - 每次排列用 LLM listwise prompt 選出 top_k=10 並排序。
   - 平行呼叫多把 API key，避免單 key rate limit 拖慢。

3. **Kemeny-Young 近似聚合**
   - 將 5 次排名結果轉成 pairwise 偏好矩陣。
   - 以 net-win 初始化排序，並用「相鄰交換」的局部搜尋近似 Kemeny。
   - 設定硬上限 10 秒，逾時即停止，避免卡死。

4. **輸出排序**
   - 先輸出 Kemeny 產生的排名（僅涵蓋 top_k 的聯集），其餘候選按 original_rank 依序補尾。
   - 最終只改排序，不重新計算分數。

**Slide 9｜與 PermSC 論文的一致性與差異**

**一致之處**
1. 多次隨機排列 + LLM listwise 排序，降低位置偏誤。
2. 使用 Kemeny-Young 的「中央排名」概念做聚合。

**差異之處（重要）**
1. **僅用 top_k 聯集做 Kemeny**：未對完整 100 筆做全域 Kemeny。
2. **近似方法為局部交換**：非論文中的 exact/完整近似演算法實作。
3. **未使用 sliding window**：長清單未分窗，完全依靠 top_k 取樣。
4. **只調整順序，不重算分數**：reranker 不產生新的 score，僅重排。

結論：架構方向與論文一致，但計算層級與演算法仍屬「工程近似版」，並未完全對齊論文中的精確/近似算法實作。

**產出物與指令**

| 類型 | 路徑/指令 |
| --- | --- |
| Run JSON (6:4) | data/experiments/runs/batch_20260512_235455/gemma4_default_parser.json |
| Run JSON (7:3/8:2) | data/experiments/runs/batch_20260513_184425/gemma4_w73.json, data/experiments/runs/batch_20260513_184425/gemma4_w82.json |
| LLM Judge 指令 | python -m src.eval.llm_judge --experiment-dir data/experiments/runs/batch_20260512_235455 --annotations-dir data/experiments/annotations_llm_gemma4 --model gemma-4-31b-it |
| 指標指令 | python -m src.eval.metrics --experiment-dir data/experiments/runs/batch_20260512_235455 --annotations-dir data/experiments/annotations_llm_gemma4 |
| 指標指令 | python -m src.eval.metrics --experiment-dir data/experiments/runs/batch_20260513_184425 --annotations-dir data/experiments/annotations_llm_gemma4 |
