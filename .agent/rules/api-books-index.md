---
trigger: model_decision
description: JSON 檔案格式描述：來自 **鏡文學 (MirrorFiction)** 的 API 回傳資料。它是一個標準的 RESTful API 回應格式，主要包含書籍列表的數據。
---

呼叫範例：

https://www.mirrorfiction.com/api/books/index?lang=zh-Hant&type=1&orderBy=click&sortedBy=desc&r=1&page=2

以下是詳細的結構解析：

### 第一層：根目錄 (Root)

整個 JSON 物件包含三個主要欄位，用來告訴前端（網頁或 App）這次請求的狀態：

1.  **`result`**: `"success"`
    *   代表 API 請求成功。
2.  **`meta`**: (物件)
    *   包含 **分頁資訊 (Pagination)**。這告訴你目前抓取的是第幾頁、總共有多少本書。
    *   `total`: 17518 (總書籍數)
    *   `per_page`: 20 (每一頁顯示 20 筆)
    *   `current_page`: 2 (目前在第 2 頁)
    *   `links`: 提供上一頁 (`previous`) 和下一頁 (`next`) 的 API 連結。
3.  **`data`**: (陣列 Array)
    *   這是核心內容。裡面包含了一個書籍物件的列表（List），每一項代表一本小說。

---

### 第二層：書籍物件詳情 (`data` 陣列中的每一項)

在 `data` 裡面的每一個 `{...}` 都代表一本書。以下是關鍵欄位的解釋：

#### 1. 基本資訊
*   **`id`**: 書籍的唯一識別碼 (例如: `15419`)。
*   **`name`**: 書名 (例如: `"女攻花樣虐渣實錄[GB]"`)。
*   **`slogan`**: 短標語/副標題。
*   **`intro`**: 書籍簡介。注意這裡面包含了 HTML 標籤 (如 `<p>`, `<br>`)，顯示時需要解析 HTML。
*   **`cover`**: 封面圖片的 URL 網址。
*   **`language`**: 語言代碼 (如 `zh-Hant` 繁體中文, `zh-Hans` 簡體中文)。

#### 2. 狀態與屬性
*   **`publish_status`**: 連載狀態。
    *   `"ongoing"`: 連載中。
    *   `"completed"`: 已完結。
*   **`restricted_age`**: 年齡限制。
    *   `18`: 限制級 (18禁)。
    *   `0`: 全年齡。
*   **`words_total`**:總字數。
*   **`chapters_total`**: 總章節數。
*   **`updated_at`**: 最後更新時間。

#### 3. 關聯物件 (Nested Objects)
這本書還包含了其他詳細資訊的子物件：

*   **`user` (作者資訊)**:
    *   `name`: 作者名稱。
    *   `nickname`: 筆名 (例如: `"良士"` 或 `"柚臻"`)。
    *   `intro`: 作者自我介紹。
    *   `avatar`: 作者頭像圖片連結。

*   **`statistic` (數據統計)**:
    *   `click_count`: 點擊/閱讀次數 (例如: `149615`)。
    *   `comment_count`: 留言數。
    *   `collection_count`: 收藏數。
    *   `bookmark_count`: 書籤數。

*   **`classification` (主分類)**:
    *   `name`: 分類名稱 (例如: `"LGBT"`, `"恐怖驚悚"`, `"推理犯罪"`).
    *   `icon`: 分類圖示。

*   **`tags` (標籤陣列)**:
    *   這是一個陣列，包含多個標籤物件。
    *   例如：`"name": "高H"`, `"name": "BDSM"`, `"name": "甜文"`, `"name": "百合"` 等等。這些是用來幫助搜尋和過濾的關鍵字。

#### 4. 電子書檔案
*   **`epub`**: 未加密的電子書下載連結 (通常 API 會擋，不一定能直接下載)。
*   **`epub_encrypt`**: 加密的電子書連結。