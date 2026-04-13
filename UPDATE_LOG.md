### 核心更新：BM25 檢索與 RRF 融合機制初步整合

本更新為檢索系統引入了詞彙層面的檢索能力，並透過融合演算法提升了整體相關性。

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
