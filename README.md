# Hybrid Reasoner (Novels) - Ver 1.1

這是一個結合 **向量檢索 (Vector Search)** 與 **規則評分 (Rule-based Scoring)** 的混合推薦系統。專案以小說推薦為範例，利用 LLM (Large Language Model) 解析使用者的自然語言查詢，將其轉換為具體的結構化搜尋條件，並結合語意相似度進行精確排序。

**Ver 1.1 更新重點：**

* 整合 **Google Gemini** 模型作為推理核心。
* 新增 **Logic Push-down** 優化，提升檢索效率。
* 新增 **Explainability**，讓系統能解釋推薦理由。
* 全新 **Web UI** 介面與 Windows 一鍵啟動腳本。

---

## 🧪 重排序策略實驗 (Rerank Strategy Experiment)

本分支 `experiment/rerank-strategy` 包含了針對三種不同重排序策略的深度評測結果。

### 📊 實驗核心總結

* **測試策略**：`score_only` (基礎版), `original_llm_reranker_top10` (LLM 大腦版), `hybrid_fusion` (Cross-Encoder 融合版)。
* **🏆 最佳方案**：**`original_llm_reranker_top10`**。在主觀盲測中獲得最高勝率 (43.75%)，且在處理「負面表述」(例如：不要異世界) 時展現了最佳的傷害控制能力。
* **⚡ 效能指標**：
  * `score_only`：延遲最低 (0s Rerank)，穩定度高。
  * `original_llm_reranker_top10`：額外增加約 12 秒延遲，但換取顯著的品味提升。
  * `hybrid_fusion`：耗時最長 (20s+)，且在輕小說場境下投資報酬率低。

### 📝 實驗報告與結果位置

* **實驗結果存放目錄**：[`experiments/results/`](./experiments/results/)
* **詳細評測報告 (Markdown)**：👉 **[重排序策略評測報告](./experiments/results/rerank_strategy_evaluation_report.md)**

---

## ✨ 特色 (Features)

* **混合檢索引擎**: 結合 Qdrant 向量資料庫與傳統欄位過濾/評分 (Logic Push-down supported)。
* **自然語言查詢 (Gemini Powered)**: 使用 Google Gemini 模型解析模糊的查詢意圖 (例如: "找一本字數超過十萬字的奇幻小說")，並生成結構化搜尋條件。
* **可解釋性 (Explainability)**: 系統不僅推薦書籍，還會告訴你*為什麼*這本書符合你的需求。
* **自動化爬蟲**: 從 MirrorFiction 抓取小說資料作為測試數據。
* **Modern Web UI**: 提供直觀的網頁搜尋介面。
* **Windows 一鍵啟動**: 透過 `run_web.bat` 快速啟動服務。

## 🛠️ 安裝 (Installation)

1. **複製專案**

    ```bash
    git clone https://github.com/wangicheng/hybrid-reasoner.git
    cd hybrid-reasoner
    git checkout 1.1ver
    ```

2. **建立虛擬環境 (建議)**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3. **安裝依賴套件**

    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ 設定 (Configuration)

1. 複製 `.env.example` 為 `.env`：

    ```bash
    # Windows (PowerShell)
    Copy-Item .env.example .env
    ```

2. 編輯 `.env` 檔案，填入您的 **Google Gemini API Key**：

    ```ini
    GOOGLE_API_KEY=your_gemini_api_key_here
    # 如有需要可調整 Qdrant 設定
    # QDRANT_HOST=localhost
    # QDRANT_PORT=6333
    ```

## 🚀 快速開始 (Quick Start)

### Windows 使用者 (推薦)

直接雙擊專案根目錄下的 **`run_web.bat`** 腳本，即可自動啟動伺服器並開啟瀏覽器。

---

### 手動啟動步驟

#### 1. 抓取資料與初始化 (首次執行)

如果是第一次執行，需要先抓取資料並建立索引：

```bash
# 抓取資料
python -m src.crawler

# 建立索引 (Seeding)
python -m src.main --seed
```

#### 2. 啟動 Web 服務

啟動 FastAPI 後端伺服器：

```bash
python -m src.web_api
# 或使用 uvicorn
python -m uvicorn src.web_api:app --reload
```

伺服器啟動後，請瀏覽器打開 [http://localhost:8000](http://localhost:8000) 即可看到搜尋介面。

#### 3. 命令列搜尋 (CLI Search)

也可以直接透過命令列測試搜尋與解釋功能：

```bash
python -m src.main --query "推薦幾本關於魔法與冒險的小說，字數要在20萬字以上"
```

## 📂 專案結構 (Project Structure)

```text
hybrid-reasoner/
├── data/                 # 資料存放區 (爬蟲結果, Qdrant 儲存檔)
├── src/
│   ├── core/             # 核心模組
│   │   ├── engine.py     # 混合檢索引擎 (Logic Push-down 實作)
│   │   ├── llm.py        # LLM 介面 (Gemini Adapter)
│   │   ├── vector_store.py # 向量資料庫介面
│   │   └── explainer.py  # 解釋生成模組
│   ├── logic/            # 評分邏輯
│   ├── web/              # 前端靜態檔案 (HTML/JS/CSS)
│   ├── crawler.py        # 爬蟲程式
│   ├── main.py           # CLI 入口
│   └── web_api.py        # Web Server 入口
├── run_web.bat           # Windows 啟動腳本
└── requirements.txt      # 依賴列表
```
