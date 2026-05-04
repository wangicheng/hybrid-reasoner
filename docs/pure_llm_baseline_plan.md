# Pure LLM Baseline Plan

## 目標

建立一個可被 `src.eval.generate_run` 直接呼叫的「純 LLM 對照組」引擎，用來和目前的 `HybridEngine` 比較。

這個 baseline 的定義建議如下：

- 允許使用 SQLite 當作資料來源
- 不使用 Qdrant / embedding retrieval
- 不使用 tag 映射、規則分數融合、後處理 hard filter
- 排名與選書決策由 LLM 完成
- 輸出格式盡量與 `HybridEngine.search()` 一致，讓 `generate_run` 幾乎不用改資料格式

## 為什麼不能把全資料庫一次丟給 LLM

目前資料庫約 `5059` 本作品。

- 每本作品的精簡卡片平均約 `210` 字元
- 全庫一次送入 prompt 約超過 `100` 萬字元
- 成本、延遲、context window 風險都太高

因此 pure LLM 版本應該採用「分塊篩選 + 全域重排」的兩階段架構。

不過如果研究目的就是要展示 naive 做法的侷限，也值得另外保留一個刻意糟糕的 baseline：

- `SinglePromptLLMEngine`

它會把整個 catalog 串成同一個 prompt，直接要求 LLM 從全庫選 top-k。

目前本地量到的單次 prompt 規模約為：

- `catalog_size = 5059`
- `prompt_chars ~= 1,822,580`
- `prompt_bytes_utf8 ~= 4,167,251`

這個 baseline 的價值不在效果，而在於提供一個容易失敗、容易退化、也最容易說明 hybrid engine 優勢的對照組。

## 推薦方案

### 方案名稱

`PureLLMEngine`

### 核心思路

1. 先把全書庫轉成穩定的「作品卡片」
2. 按固定大小切成 chunks
3. LLM 對每個 chunk 做局部篩選
4. 合併所有候選後，再做一次全域 rerank
5. 取前 `k` 筆並回傳與現有引擎相同的結果格式

這樣的好處是：

- 不用向量檢索，符合「純 LLM」定位
- 不需要手工規則打分
- `generate_run` 只要改引擎建立方式即可沿用
- 每個 chunk 都是固定切法，實驗可重現

## 建議的兩階段流程

### Stage A: Metadata Screening

先對全庫做粗篩，但只提供非常精簡的 metadata 卡片，例如：

```text
id=123 | title=... | author=... | status=... | words=... | tags=tag1,tag2,tag3 | cls=...
```

每個 chunk 約放 `150-250` 本，讓單次 prompt 保持穩定。

LLM 任務：

- 根據 query 從該 chunk 選出 top `2-4` 本候選
- 每筆只回傳 `book_id`、`confidence`、`short_reason`
- 不可捏造 ID

這一層的目的不是最終排序，而是用低成本從 5059 本縮到大約 `40-100` 本。

### Stage B: Rich Rerank

把 Stage A 保留下來的候選作品重新查回完整資料，組成較完整卡片，例如：

```text
id=123
title=...
author=...
status=...
words=...
tags=...
intro=... (截斷 120-220 字)
```

LLM 任務：

- 直接針對 query 做 end-to-end 判斷
- 回傳 top `k` 的 `book_id` 排序
- 可附簡短理由

這一層才是最終輸出。

## 為什麼先 metadata、後 intro

如果 Stage A 就把 intro 全送進去，單 query 呼叫次數和 token 成本會太高。

先用 metadata 粗篩有幾個優點：

- tags / status / classification 已經能處理大量明確需求
- 只對少量候選補 intro，可以把語意判斷成本留到最後
- 比起全量 intro 掃描，實驗更容易跑完多輪 repeats

## 與 HybridEngine 的公平比較邊界

建議把 pure LLM baseline 定義成：

- `HybridEngine`: LLM parser + vector retrieval + tag recall + score fusion + post-filter
- `PureLLMEngine`: LLM 直接看作品卡片並決定 shortlist/rank

這樣比較的是「混合式檢索流程」對上「不使用檢索索引、由 LLM 直接做選書」。

如果後續想做更細的 ablation，可以再加第二版：

- `PureLLMEngineParserAssisted`

也就是保留 `parse_query()` 做 query 壓縮，但候選選擇與排序仍完全交給 LLM。

目前第一優先建議先做不依賴 parser 的版本，因為論文敘述最直觀。

## 需要新增的模組

### 1. `src/core/pure_llm_engine.py`

負責：

- 載入全書庫
- 建立 metadata cards / rich cards
- chunking
- LLM screening
- LLM rerank
- 輸出統一格式

建議公開介面：

```python
class PureLLMEngine:
    async def search(
        self,
        user_query: str,
        limit: int = 5,
        model_id: Optional[str] = None,
        explain: bool = True,
        cache_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...
```

介面刻意與 `HybridEngine.search()` 對齊，方便 `generate_run` 共用。

### 2. `src/eval/generate_run.py`

建議把 experiment config 改成支援：

```python
{
    "name": "pure_llm_gemma4",
    "engine_type": "pure_llm",
    "model_id": "gemma-4-31b-it",
}
```

以及：

```python
{
    "name": "hybrid_gemma4",
    "engine_type": "hybrid",
    "model_id": "gemma-4-31b-it",
}
```

如此同一份 runner 就能產生兩種引擎的 run 檔。

### 3. 可選：`tests/test_pure_llm_engine.py`

至少先補這幾類測試：

- chunking 是否穩定
- LLM 回傳重複 ID 時是否正確去重
- LLM 回傳不存在 ID 時是否正確忽略
- 候選不足 `k` 時是否安全返回

## 回傳格式建議

盡量與現有 `HybridEngine.search()` 對齊：

```python
{
    "query": user_query,
    "parsed_criteria": [],
    "search_terms": user_query,
    "generated_keywords": [],
    "tag_intent": {
        "search_terms": user_query,
        "positive_terms": [],
        "negative_terms": [],
    },
    "related_books": [],
    "reference_tags": [],
    "parse_metadata": {...},
    "query_vector": [],
    "results": [...],
    "engine": "PureLLMEngine",
}
```

其中 `parse_metadata` 可以改成承載 pure LLM 的 runtime telemetry，例如：

- `engine_mode`
- `catalog_size`
- `chunk_count`
- `screening_chunk_size`
- `screening_survivors`
- `total_request_count`
- `retry_count`
- `latency_ms`

這樣 `generate_run` 不需要多加太多欄位，就能把 pure LLM 的執行資訊一起存下來。

## Prompt contract 建議

### Stage A output

```json
{
  "matches": [
    {
      "book_id": "123",
      "confidence": 0.88,
      "reason": "符合奇幻、冒險與已完結需求"
    }
  ]
}
```

限制：

- 只能從提供的 ID 中選
- 最多選 `N` 本
- 找不到就回空陣列

### Stage B output

```json
{
  "ranked_ids": ["123", "456", "789"],
  "notes": [
    {"book_id": "123", "reason": "最符合主題與排除條件"},
    {"book_id": "456", "reason": "主題接近但限制匹配較弱"}
  ]
}
```

## 實作順序建議

### Phase 1: 跑得起來

1. 新增 `PureLLMEngine`
2. 用固定 chunk + metadata screening + rich rerank 完成最小可跑版本
3. 改 `generate_run` 讓它能切換 `engine_type`
4. 先產出 1 個 pure LLM run 檔驗證格式

### Phase 2: 提升穩定性

1. 加 retry / rate limit metadata
2. 加非法 ID / 重複 ID 防護
3. 把 chunk 順序固定化，確保 repeat 可比較

### Phase 3: 實驗優化

1. 調整 chunk size
2. 調整 Stage A 每塊保留數量
3. 比較是否要在 Stage B 加 intro 長度
4. 視需要新增 parser-assisted 版本做 ablation

## 風險

### 1. 成本與延遲

pure LLM 版本即使不走 embedding，仍然會有多次 LLM 呼叫；這會明顯慢於目前 hybrid flow。

### 2. Chunk 漏召回

如果 Stage A 每塊只留太少本，可能把真正相關作品提早刷掉。

建議先從：

- `chunk_size = 180`
- `survivors_per_chunk = 3`

開始。

### 3. LLM 幻覺 ID

必須在程式端嚴格檢查：

- ID 是否存在於當前 chunk
- 排名 ID 是否存在於候選池

不合法結果一律丟棄。

## 第一版最重要的設計決策

如果只做一版，我建議：

- 不沿用 `parse_query()`
- 不使用 Qdrant
- 不使用 tag 映射與後過濾
- 直接做 `metadata screening -> rich rerank`
- 輸出格式完全貼齊 `HybridEngine.search()`

這是最容易解釋、最像真正對照組、也最容易接進 `generate_run` 的做法。
