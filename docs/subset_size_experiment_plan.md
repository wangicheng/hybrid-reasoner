# Subset Size Experiment Plan

## 目標

比較 `HybridEngine` 與 `SinglePromptLLMEngine` 在不同書庫規模下的表現。

核心做法：

- 每次查詢前，先從全書庫隨機抽樣出一個子集
- 兩個引擎使用完全相同的子集
- 對每個子集大小重複多次實驗
- 最後對每個子集大小做平均

這個設計能回答兩個問題：

1. `single_prompt_llm` 在書庫逐漸變大時，品質如何退化？
2. `hybrid` 是否能在更大的 candidate pool 下維持較好的品質與穩定性？

## 為什麼這比 full-catalog single prompt 更適合

直接把全部 `5059` 本書塞進同一個 prompt，`SinglePromptLLMEngine` 很可能：

- timeout
- provider 拒絕
- 回傳空結果
- 回傳品質極差的排序

這雖然可以作為反例，但很難形成一條平滑、可分析的性能曲線。

改成不同 subset size 的實驗後，我們可以得到：

- 可觀察的退化曲線
- single prompt 從「可用」到「失效」的轉折點
- hybrid 與 single prompt 在不同規模下的 gap

## 推薦實驗設計

### 自變數

- `subset_size`

建議先測：

- `50`
- `100`
- `250`
- `500`
- `1000`
- `2000`
- `3000`
- `4000`
- `5000`

理由：

- 全庫是 `5059`
- 目前 single prompt 全庫大小約 `1.82M` 字元
- 平均每本書在 single-prompt catalog 中約佔 `360` 字元上下
- 這組 size 大致能畫出從小型 prompt 到接近 full catalog 的曲線

### 重複次數

建議先用：

- `repeats = 5` 或 `10`

如果 API 成本允許，`10` 會比較適合做平均與畫 error bar。

## 最重要的抽樣原則

### 1. 兩個引擎必須共用同一份子集

同一個 `subset_size`、同一個 `repeat_index` 下：

- `HybridEngine`
- `SinglePromptLLMEngine`

都必須用完全相同的 `book_id` 集合。

否則差異會混入資料差，而不是引擎差。

### 2. 建議使用 nested subsets

推薦不要對每個 `subset_size` 都獨立重新抽樣，而是：

1. 對每個 repeat 先隨機打亂全書庫一次
2. `size=50` 用前 50 本
3. `size=100` 用前 100 本
4. `size=250` 用前 250 本
5. 依此類推

好處：

- 同一個 repeat 內，subset 是巢狀關係
- size 曲線更平滑
- 能更乾淨地觀察「只因為書變多」帶來的影響

## 建議的資料流

### Step 1: 先產生 subset manifest

新增一個 manifest 檔，記錄每次實驗用到哪些書：

```json
{
  "base_seed": 42,
  "catalog_size": 5059,
  "repeats": [
    {
      "repeat_index": 1,
      "permutation_book_ids": ["...", "...", "..."]
    }
  ]
}
```

之後每個 size 直接取 permutation 的前 N 個。

這樣的好處：

- 可重現
- 可審查
- 所有引擎都能重用同一份抽樣結果

### Step 2: 建立 subset-scoped engine

不是實際刪資料，而是在引擎層限制「只能看到這些 book ids」。

這樣比較安全，也比較容易重複跑。

### Step 3: 產出 run 檔

每個條件產生一個 run：

- engine type
- model id
- subset size
- repeat index

檔名建議像：

- `hybrid_size0050_run01.json`
- `single_prompt_size0050_run01.json`

### Step 4: 聚合分析

最後按 `subset_size` 匯總：

- 平均 `Avg@10`
- 平均 `strict Avg@10`
- query success rate
- empty-result rate
- timeout / error rate
- 平均 latency

## 需要修改的模組

### 1. `src/core/single_prompt_engine.py`

新增：

- `allowed_book_ids: Optional[set[str]]`

用途：

- 只把 subset 中的書組進 catalog prompt

### 2. `src/core/engine.py`

新增：

- `allowed_book_ids: Optional[set[str]]`

用途：

- vector retrieval 只查 subset
- tag recall 只保留 subset
- final candidates 只保留 subset

注意：

- 只在最後結果過濾是不夠的
- 如果 vector top-N 先被 subset 外的書佔滿，會扭曲 hybrid 的真實表現

因此 subset 約束應該盡量往前放到 retrieval 階段

### 3. `src/core/vector_store.py`

需要支援 subset filter。

最理想做法：

- 在 `search()` 傳入 Qdrant `has_id` filter

這樣 dense retrieval 一開始就只在 subset 中找。

### 4. `src/core/database.py`

至少要讓這些方法支援 subset：

- `get_all_items()`
- `search_by_tags_any()`
- `search_by_title_fuzzy()`
- `search_by_author()`

否則：

- single prompt 雖然只看 subset
- hybrid 的 parser side context 或 tag recall 還可能偷看到 subset 外的書

### 5. `src/eval/generate_run.py`

如果要保持簡單，建議不要把 subset 邏輯硬塞進現有檔案，而是另外新增：

- `src/eval/generate_subset_runs.py`

因為這個實驗比一般 run 多了：

- subset manifest
- size loop
- repeat loop
- engine matrix

獨立腳本會比較乾淨。

### 6. 新分析腳本

建議新增：

- `src/eval/analyze_subset_runs.py`

職責：

- 讀所有 subset run
- 按 `engine_type x subset_size` 分組
- 算平均與標準差
- 輸出可直接畫圖的 CSV

## 建議的 metadata

每個 run row 都建議加上：

- `engine_type`
- `subset_size`
- `repeat_index`
- `subset_seed`
- `subset_id`
- `catalog_size_before_subset`
- `catalog_size_after_subset`

`parse_metadata` 或 run-level metadata 也應記：

- prompt 字元數
- request count
- retry count
- latency
- 是否 error

## 評估指標

### 品質指標

- 現有 `Avg@10`
- `strict Avg@10`
- `coverage-adjusted strict Avg@10`

### 穩定性指標

- query success rate
- empty-result rate
- invalid-output rate
- timeout rate

### 成本代理指標

- 平均 latency
- 單 query request count
- prompt char count

## 很重要的控制變因

### 1. 模型必須固定

例如：

- hybrid: `gemma-4-31b-it`
- single prompt: `gemma-4-31b-it`

先固定同一模型，避免把模型差異混進來。

### 2. subset 要先生成，再共用

不要讓每個引擎自己抽樣。

正確順序是：

1. 先決定 subset
2. 再把同一 subset 餵給兩個引擎

### 3. 查詢集固定

沿用同一份 `queries.json`，不要不同 size 用不同 query。

## 一個容易被忽略的問題

有些 subset 可能把真正相關的好書抽掉，導致某些 query 在該 subset 下變成「本來就無解」。

這不是 bug，但分析時最好補一個欄位：

- `query_viable_in_subset`

定義方式可以是：

- 該 subset 中是否還存在至少一個已標註為高分的候選書

這能幫你區分：

- 引擎沒找好
- 還是資料子集本身已經沒有答案

## 建議的分析輸出

最終至少做兩張圖：

### 圖 1: 品質曲線

- X 軸：`subset_size`
- Y 軸：`Avg@10` 或 `strict Avg@10`
- 線條：
  - hybrid
  - single prompt

### 圖 2: 穩定性曲線

- X 軸：`subset_size`
- Y 軸：query success rate
- 線條：
  - hybrid
  - single prompt

如果 single prompt 在大 size 失敗率暴增，這張圖會很有說服力。

## 推薦實作順序

### Phase 1: 基礎設施

1. 生成 subset manifest
2. 讓 `SinglePromptLLMEngine` 支援 `allowed_book_ids`
3. 讓 `HybridEngine` 支援 `allowed_book_ids`

### Phase 2: 跑實驗

1. 新增 `generate_subset_runs.py`
2. 先跑小規模 smoke test：
   - size `50`
   - size `100`
   - repeats `2`
3. 確認 run 檔格式與 metadata 正確

### Phase 3: 分析

1. 新增 `analyze_subset_runs.py`
2. 匯出聚合 CSV
3. 畫曲線圖

## 如果只做最小可行版本

我建議 MVP 就做這些：

1. `allowed_book_ids`
2. nested subset manifest
3. `subset_sizes = [50, 100, 250, 500, 1000]`
4. `repeats = 5`
5. 比較 `strict Avg@10` + `query success rate`

這樣就足以支撐論文中的核心論點：

- 小書庫下，single prompt 可能還能勉強工作
- 書庫規模一上去，single prompt 的品質與穩定性明顯退化
- hybrid engine 在相同資料子集下更穩定，也更能維持檢索品質
