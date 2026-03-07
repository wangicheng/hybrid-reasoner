# API Key 輪換機制 - 實作完成總結

## 背景
您的 `.env` 檔案包含 20 個 Google API Key，用於在達到 API 速率限制時，自動切換到下一個 key。

## 已實作的功能

### 1️⃣ 多 API Key 解析支援 (`src/config.py`)
- ✅ 從 `GOOGLE_API_KEY` 環境變數解析逗號分隔的多個 key
- ✅ 存儲為 `settings.GOOGLE_API_KEYS` 列表 (20 個 key)
- ✅ 支援 Pydantic Settings 配置

```python
# .env 格式支援：
GOOGLE_API_KEY=key1, key2, key3, ..., key20
```

### 2️⃣ API Key 輪換管理器 (`src/core/api_utils.py`)

#### 新增 `APIKeyRotator` 類
```python
rotator = APIKeyRotator(api_keys)
rotator.get_current_key()      # 取得當前 Key
rotator.rotate()               # 輪換到下一個 Key
rotator.on_rate_limit_error()  # 遇到速率限制時輪換
```

#### 全局實例
```python
rotator = get_api_key_rotator()        # 取得全局輪換器
current_key = get_current_api_key()    # 取得當前活動 Key
```

**特性**：
- 🔄 自動循環輪換 (超過 20 則回到第 0 個)
- 🔒 線程安全 (Thread-safe)
- 📊 自動日誌記錄 (每次輪換時提示)

### 3️⃣ VectorStore 整合 (`src/core/vector_store.py`)

#### 初始化時使用輪換 API Key
```python
class VectorStore:
    def __init__(self, collection_name="items"):
        api_key = get_current_api_key()  # ← 自動選擇當前 Key
        self.genai_client = genai.Client(api_key=api_key)
```

#### 遇到速率限制時自動輪換
```python
def _update_api_key_on_rate_limit(self):
    """在 RESOURCE_EXHAUSTED 錯誤時自動輪換"""
    new_key = rotator.on_rate_limit_error()
    self.genai_client = genai.Client(api_key=new_key)
```

### 4️⃣ LLM 模組整合 (`src/core/llm.py`)

#### Query 解析支援 API Key 輪換
- ✅ 使用 `get_current_api_key()` 初始化 genai client
- ✅ 遇到 429/RESOURCE_EXHAUSTED 時自動輪換
- ✅ 最多嘗試 N 個 API key (N = key 總數)
- ✅ 支援模型 fallback + API key fallback

```
User Query 
    ↓
Try Model 1 with Key 0
    ↓ (429 error) → Rotate to Key 1
Try Model 1 with Key 1
    ↓ (429 error) → Try Model 2 with Key 2
    ... etc
```

## 使用流程

### 場景 1：正常搜尋
```
使用者查詢
    ↓
parse_query() 使用當前 Key (e.g., Key 0)
    ↓
VectorStore.search() 使用當前 Key
    ↓
返回結果
```

### 場景 2：遇到速率限制
```
使用者查詢
    ↓
genai.Client() 遇到 429 RESOURCE_EXHAUSTED
    ↓
APIKeyRotator.on_rate_limit_error()
    ↓
切換到 Key 1，重試
    ↓
成功返回結果
```

## 測試結果 ✅

已通過測試驗證：
1. ✅ 20 個 API key 成功載入
2. ✅ 輪換器初始化正常
3. ✅ 當前 key 取得正常
4. ✅ 輪換邏輯正確
5. ✅ 自動循環回到第 0 個 key

```
✓ Test 1: Config loading
  Total API keys loaded: 20
  
✓ Test 2: Rotator initialization
  Rotator has 20 keys
  
✓ Test 3: Getting current key
  Current key: AIzaSyBpK9... ✓
  
✓ Test 4: API Key rotation (rotate 3 times)
  After rotate 1: index=1 ✓
  After rotate 2: index=2 ✓
  After rotate 3: index=3 ✓
  
✓ Test 5: Wrapping around (從第 19 個→ 第 0 個)
  Wrapping successful ✓
```

## 關鍵改動檔案

| 檔案 | 改動 | 狀態 |
|------|------|------|
| `src/config.py` | 新增 API key 解析邏輯 | ✅ |
| `src/core/api_utils.py` | 新增 APIKeyRotator 類 | ✅ |
| `src/core/vector_store.py` | 整合 API key 輪換 | ✅ |
| `src/core/llm.py` | 整合 API key 輪換 + 模型 fallback | ✅ |

## 後續使用建議

### 監控 log 輸出
```
[APIKeyRotator] Initialized with 20 API key(s)
[APIKeyRotator] Switched from key 0 to key 1  ← 表示遇到限制，已輪換
[VectorStore] API key rotated. Now using key index: 1
```

### 優化 rate limiting
目前的 `RateLimiter` 設定：
- 最小間隔：4 秒 (15 RPM，符合免費版)
- 搭配 API key 輪換，理論可達 20 × 15 RPM = ~300 RPM

### 未來擴展
1. 可在 `.env` 配置 rate limit 間隔
2. 可添加 key 健康檢查機制
3. 可記錄各 key 的使用統計

---

**實作完成日期**：2026-03-07  
**狀態**：✅ 準備就緒
