---
trigger: manual
description: JSON 檔案格式描述：來自 **鏡文學 (MirrorFiction)** 的 API 回傳資料。它是一個標準的 RESTful API 回應格式，主要包含單一本特定小說的完整資訊。
---

呼叫範例：

https://www.mirrorfiction.com/api/books/1001?lang=zh-Hant&include=classification,attribute.type,statistic,tags,user.userStatistic,chapters:limit(1|1):order(order_column|asc),news:limit(2|1):order(updated_at|desc),personalized

以下是詳細的結構解析：

### 第一層：回應狀態
*   **`result`**: `"success"` (API 請求成功)。
*   **`data`**: 這是一個**物件 (Object)**，而不是陣列。它包含了這本書所有的詳細資料。

---

### 第二層：書籍核心資料 (`data` 物件內部)

#### 1. 基本識別與內容
*   **`id`**: `1001` (書籍唯一 ID)。
*   **`name`**: `"我們不能是朋友"` (書名)。
*   **`slogan`**: `"劉以豪郭雪芙領銜主演偶像劇原著！"` (宣傳標語，這裡顯示這本書已被改編成電視劇)。
*   **`intro`**: 書籍簡介 (HTML 格式，內含電視劇資訊、電子書購買連結等)。
*   **`cover`**: 封面圖片網址。
*   **`language`**: `"zh-Hant"` (繁體中文)。

#### 2. 規格與狀態
*   **`words_total`**: `34786` (目前字數)。
*   **`chapters_total`**: `17` (目前章節數)。
*   **`publish_status`**: `"completed"` (已完結)。
*   **`restricted_age`**: `0` (普遍級，無年齡限制)。
*   **`tts`**: `true` (代表這本書支援 **Text-to-Speech**，也就是有語音朗讀功能)。
*   **`is_free`**: `true` (免費閱讀)。

#### 3. 下載資源
*   **`epub`** / **`epub_encrypt`**: 電子書檔案的下載連結。

---

### 第三層：關聯物件 (Nested Objects)

這部分是與「列表 API」最大的不同點，這裡包含了更多深層資訊：

#### A. `user` (作者資訊)
*   **`name`**: `"蔡芳紜"` (作者本名或註冊名)。
*   **`nickname`**: `"阿亞梅"` (對外筆名)。
*   **`intro`**: 作者簡介，提到她是編劇，作品被改編成電視劇。
*   **`userStatistic`**: 作者的創作數據（如總寫作字數、被追蹤數等）。

#### B. `chapters` (章節列表)
這是這本書的目錄。
*   **`data`**: 一個陣列，列出具體的章節。
    *   `name`: 章節名稱 (例如：`"-01-上市公司的代碼遊戲"`).
    *   `words`: 單章字數。
    *   **`voice` / `tts`**: 這是很特殊的欄位，包含了 **MP3 音檔連結**。
        *   `Female`: 女聲朗讀版網址。
        *   `Male`: 男聲朗讀版網址。
    *   `chapter_status`: `"open"` (可以閱讀)。

#### C. `statistic` (統計數據)
*   **`click_count`**: `949529` (將近 95 萬次點擊，非常熱門)。
*   **`rank_title`**: `"週書單排行榜 第 11 名"` (目前的榜單排名)。
*   **`bookmark_count`**: 書籤數。

#### D. `news` (相關新聞/公告)
這本書關聯的新聞或網站公告，通常顯示在書籍頁面的底部。
*   **`title`**: 新聞標題 (例如 `"2024 Highlights 鏡文學年度回顧"`).
*   **`content`**: 新聞內容 (HTML 格式，包含大量的排版和圖片)。
*   **`image`**: 新聞縮圖。

#### E. `tags` & `classification` & `attribute` (分類系統)
*   **`classification`**: 主分類是 `"愛情"` (ID: 1)。
*   **`attribute`**: 屬性是 `"長篇小說"` (定義為六萬字以上)。
*   **`tags`**:標籤包含 `"現代"`, `"都市"`, `"言情"`, `"鏡文學出版"`。

#### F. `personalized` (個人化狀態)
這通常是給已登入的使用者看的，用來判斷當前使用者與這本書的關係：
*   `recommended`: 是否已推薦過。
*   `collected`: 是否已收藏。
*   `purchased`: 是否已購買。
*   (這裡全是 `false`，可能代表未登入狀態或未互動過)。