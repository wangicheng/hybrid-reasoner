# Hybrid Reasoner (Novels)

這是一個結合 **向量檢索 (Vector Search)** 與 **規則評分 (Rule-based Scoring)** 的混合推薦系統。專案以小說推薦為範例，利用 LLM (Large Language Model) 解析使用者的自然語言查詢，將其轉換為具體的結構化搜尋條件，並結合語意相似度進行精確排序。

## ✨ 特色 (Features)

*   **混合檢索引擎**: 結合 Qdrant 向量資料庫與傳統欄位過濾/評分。
*   **自然語言查詢**: 使用 OpenAI 模型解析查詢意圖 (例如: "找一本字數超過十萬字的奇幻小說")。
*   **自動化爬蟲**: 從 MirrorFiction 抓取小說資料作為測試數據。
*   **Web 介面**: 提供簡易的搜尋前端。
*   **可擴充評分邏輯**: 透過 `ScoringRegistry` 輕鬆新增評分規則。

## 🛠️ 安裝 (Installation)

1.  **複製專案**

2.  **建立虛擬環境 (建議)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安裝依賴套件**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ 設定 (Configuration)

1.  複製 `.env.example` 為 `.env`：
    ```bash
    cp .env.example .env
    # Windows (PowerShell)
    Copy-Item .env.example .env
    ```

2.  編輯 `.env` 檔案，填入您的 OpenAI API Key：
    ```ini
    OPENAI_API_KEY=sk-your-key-here
    # 如使用自定義 Endpoint 可修改
    # OPENAI_BASE_URL=...
    # LLM_MODEL_ID=gpt-4o
    ```

## 🚀 快速開始 (Quick Start)

### 1. 抓取資料 (Crawling)
執行爬蟲腳本抓取 MirrorFiction 的小說資料。
```bash
python -m src.crawler
# 預設會建立 data/books_crawled.json
```
*(注意：請先確認 `src/crawler.py` 腳本最後是否有呼叫執行的程式碼，若無則需手動執行)*

### 2. 建立索引與資料庫 (Seeding)
將抓取下來的資料寫入 Qdrant 向量資料庫與 SQLite/Memory。
```bash
python -m src.main --seed
```

### 3. 啟動 Web 服務 (Run Web Server)
啟動 FastAPI 後端伺服器。
```bash
python -m src.web_api
```
伺服器啟動後，請瀏覽器打開 [http://localhost:8000](http://localhost:8000) 即可看到搜尋介面。

### 4. 命令列搜尋 (CLI Search)
也可以直接透過命令列測試搜尋：
```bash
python -m src.main --query "推薦幾本關於魔法與冒險的小說，字數要在20萬字以上"
```

## 📂 專案結構 (Project Structure)

```text
c:\dev\hybrid-reasoner\
├── data/                 # 資料存放區 (爬蟲結果, Qdrant 儲存檔)
├── src/
│   ├── core/             # 核心模組
│   │   ├── engine.py     # 混合檢索引擎 (HybridEngine)
│   │   ├── llm.py        # LLM 查詢解析
│   │   ├── vector_store.py # 向量資料庫介面
│   │   └── database.py   # metadata 資料庫
│   ├── logic/            # 評分邏輯
│   │   ├── scoring_functions.py # 具體的評分函數 (字數, 更新頻率等)
│   │   └── registry.py   # 函數註冊表
│   ├── web/              # 前端靜態檔案 (HTML/JS/CSS)
│   ├── crawler.py        # 爬蟲程式
│   ├── main.py           # CLI 入口
│   └── web_api.py        # Web Server 入口
└── requirements.txt      # 依賴列表
```

## 📝 開發說明

*   **新增評分規則**: 在 `src/logic/scoring_functions.py` 中新增函數，並使用 `@ScoringRegistry.register` 裝飾器註冊。
*   **調整 LLM 解析**: 修改 `src/core/llm.py` 中的 Prompt 與 Schema 定義。
