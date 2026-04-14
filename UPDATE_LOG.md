### 核心更新：檢索前的意圖解析與查詢重構 (Pre-Retrieval)

為徹底解決否定詞（如「不要悲劇」）干擾 BM25 計分的問題，引入了全新的意圖解析與淨化處理流程。

#### 1. 結構化意圖拆解模組 (`src/core/query_preprocessor.py`)
- 新增 `build_query_intent` 流程，結合 LLM 與正則表達式，將查詢精確拆解為 **正向詞 (Positive Terms)**、**硬排除 (Hard Exclusions)** 與 **軟排除 (Soft Exclusions)**。
- 新增 `sanitize_search_terms` 工具，自動過濾否定修飾詞，確保 BM25 僅接收純淨的搜尋條件，阻止否定詞（例如「不要」或「悲劇」）帶來的錯誤增加相關性得分。
- 支援構建正向加否定排除的布林表達式 (`boolean_query`) 供除錯與後續進階過濾使用。

#### 2. LLM Prompt 增強與 Schema 擴充 (`src/core/llm.py` & `src/models/schemas.py`)
- Pydantic 模型新增 `QueryIntent` 及 `NegationConstraint` 給 LLM 建立標準化結構。
- 教導 LLM 將否定拘束分離出 `query_intent.hard_exclusions` 和 `soft_exclusions`。

#### 3. 負權重降權與排除機制整合 (`src/core/engine.py`)
- 引擎檢索開始前先呼叫 Pre-processor 取出結構化約束。
- 改為將淨化後的 `sanitized_bm25_query` 送進 `BM25Index`。
- **Negative Boosting**：若命中軟排除詞 (Soft Exclusions)，對該小說候選者執行即時扣分機制。
- **Hard Exclusions**：強硬過濾所有命中絕對排除意圖的小說。

### 核心更新：BM25 檢索與 RRF 融合機制初步整合

#### 1. 新增 BM25 檢索索引 (`src/core/database.py`)
- 建立了 `BM25Index` 類別，支援對小說簡介 (Intro) 與作者 (Author) 的搜尋。
- 整合 `jieba` 分詞系統，優化中文查詢語句的切分與匹配精準度。

#### 2. 實現 RRF (Reciprocal Rank Fusion) 融合演算法 (`src/core/engine.py`)
- 實作 `_rrf_fuse` 方法，結合「語意向量搜尋 (Vector Search)」與「詞彙關鍵字搜尋 (BM25)」的排名。
- 有效解決純語意搜尋在處理特定專有名詞或強類型關鍵字（如「盜墓」、「系統流」）時的偏移問題。

#### 3. 擴增檢索階段 (Multi-stage Retrieval)
- 在檢索流程中同步觸發 BM25 簡介檢索，並將結果併入候選集彙整，提高召回率 (Recall)。

#### 4. 精準標籤獎勵機制 (Exact Tag Bonus)
- 導入 `tag_exact_match` 邏輯，當查詢字詞精準命中書籍標籤時進行加分。
- 顯著提升特定分類標籤查詢的排序成效。

#### 5. 新增診斷與評估工具
- `diagnose_tags.py`：標籤命中分佈分析。
- `evaluate_comparison.py`：混合檢索 vs 純向量檢索之成效對照腳本。
- `queries.json`：多樣化的測試查詢集。

#### 6. 環境依賴更新
- `requirements.txt`：新增 `rank_bm25` 與 `jieba`。
