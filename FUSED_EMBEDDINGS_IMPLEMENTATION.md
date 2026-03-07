# 融合標籤向量化實現 - 完整文檔

## 實現概述

您的 Hybrid Reasoner 系統已成功升級為 **融合向量搜尋系統** (Fused Embedding Search)。系統現在使用包含 **書名、標籤、簡介** 的融合向量進行語意搜尋，而不是單獨使用書籍簡介。

## 🔧 改動詳情

### 1️⃣ 融合文本生成 (`src/core/vector_store.py`)

新增靜態方法 `build_fused_text()` - 將書籍信息融合為結構化文本：

```python
# 格式示例：
[TITLE] 書名 [/TITLE] [TAGS] tag1 tag2 tag3 [/TAGS] [ABSTRACT] 簡介文本 [/ABSTRACT]
```

**優點**：
- 結構化標記幫助 Embedding 模型更好區分各字段
- 比重簡單拼接多個字段的結果
- 標準的 NLP 融合方式

### 2️⃣ 融合向量存儲與計算 (`src/core/vector_store.py`)

新增方法 `add_fused_items()` - 為所有書籍生成融合向量：

**特性**：
- 使用獨立的 Qdrant collection：`novels_fused`
- 支援 API Key 輪換 (遇到速率限制時自動切換)
- 批次處理，避免超時
- 恢復邏輯 (支援斷點續傳)

### 3️⃣ 離線融合向量生成腳本

新增：`scripts/generate_fused_embeddings.py`

**用法**：
```bash
cd hybrid-reasoner
python scripts/generate_fused_embeddings.py
```

**流程**：
```
1. 讀取數據庫中的所有書籍
2. 對每本書籍生成融合文本
3. 向 API 發送融合文本進行 embedding
4. 將融合向量存儲到 novels_fused collection
```

**輸出示例**：
```
[1/3] 初始化資源...
      ✓ 數據庫連接成功
      ✓ Qdrant 連接成功 (collection: novels_fused)

[2/3] 讀取所有書籍資料...
      ✓ 讀取 1000 本書籍

[3/3] 生成融合向量並存儲...
      [add_fused_items] 發現 500 個現有項目，跳過...
      [add_fused_items] 已上傳 50 個融合向量
      ...
      [add_fused_items] 完成！已處理 1000 個項目
```

### 4️⃣ 搜尋引擎更新 (`src/core/engine.py`)

修改 `HybridEngine` 類：

```python
class HybridEngine:
    def __init__(self, db=None, vs=None, use_fused_vectors: bool = True):
        # 現在預設使用融合向量
        collection_name = "novels_fused" if use_fused_vectors else "novels"
        self.vs = VectorStore(collection_name=collection_name)
```

**改動點**：
- 新增 `use_fused_vectors` 參數 (預設 True)
- 自動選擇 `novels_fused` collection
- 其他評分規則保持不變 ✓

### 5️⃣ 測試腳本 (`test_fused_embeddings.py`)

驗證三項功能：

1. **融合文本生成** - 確認格式正確
2. **Collection 檢查** - 驗證 Qdrant 連接
3. **融合向量搜尋** - 測試檢索能力

**執行**：
```bash
python test_fused_embeddings.py
```

## 📊 數據流

### 舊系統
```
書籍資料 (書名、標籤、簡介)
    ↓
分別 embed 簡介
    ↓
存入 novels collection
    ↓
搜尋時：會損失標籤和書名的語意信息
```

### 新系統
```
書籍資料 (書名、標籤、簡介)
    ↓
融合為結構化文本 [TITLE]...[TAGS]...[ABSTRACT]...
    ↓
單一 embed 融合文本
    ↓
存入 novels_fused collection
    ↓
搜尋時：保留完整的語意信息
```

## 🚀 使用步驟

### 步驟 1: 檢查數據（前置條件）
```bash
python -c "from src.core.database import Database; print(len(Database().get_all_items()))"
```
確保數據庫有數據

### 步驟 2: 生成融合向量（一次性）
```bash
python scripts/generate_fused_embeddings.py
```
這會為所有書籍生成融合向量，存儲到 `novels_fused` collection

### 步驟 3: 啟動搜尋服務
```bash
python -m src.web_api
# 或使用 Windows 批次檔
run_web.bat
```

搜尋端會自動使用 `novels_fused` collection 進行查詢

### 步驟 4: 測試融合向量功能
```bash
python test_fused_embeddings.py
```

## ⚙️ 配置與自定義

### 使用非融合向量 (回退到舊系統)
```python
from src.core.engine import HybridEngine

engine = HybridEngine(use_fused_vectors=False)  # 使用舊的 novels collection
```

### 調整融合格式
編輯 `src/core/vector_store.py` 中的 `build_fused_text()` 方法：

```python
# 例如，添加分類信息：
def build_fused_text(item):
    classification = item.get('classification', '')
    # ... 修改融合邏輯
    fused = f"[CLASSIFICATION] {classification} [/CLASSIFICATION] ..."
    return fused
```

## 📈 效能改進

融合向量的優勢：

| 指標 | 舊系統 | 新系統 |
|------|--------|--------|
| 語意完整性 | 只包含簡介 | 書名 + 標籤 + 簡介 |
| 標籤匹配 | 需額外規則評分 | 融入 embedding |
| 搜尋精準度 | 中等 | 高 (預期) |
| 儲存空間 | A | 2A (新 collection) |

## ❓ 常見問題

### Q1: 融合向量需要重新計算嗎？
**A**: 需要。每次數據庫更新後：
```bash
python scripts/generate_fused_embeddings.py  # 覆蓋原有向量
```

### Q2: 可以同時保留舊向量嗎？
**A**: 可以。舊的 `novels` collection 仍然存在，可隨時切換：
```python
engine = HybridEngine(use_fused_vectors=False)  # 用舊系統
```

### Q3: API Key 輪換與融合向量的關係？
**A**: `generate_fused_embeddings.py` 會自動使用 API Key 輪換機制，遇到速率限制時自動切換

### Q4: 融合向量如何影響評分規則？
**A**: 不影響。評分規則（字數、完成度等）保持不變，融合向量只改進語意相似度的計算

## 📋 實現清單

- [x] 融合文本生成函數
- [x] VectorStore 融合向量支持
- [x] 離線上傳腳本
- [x] HybridEngine 參數化
- [x] API Key 輪換整合
- [x] 測試套件
- [x] 文檔

## 🔗 相關文件

| 文件 | 用途 |
|------|------|
| `src/core/vector_store.py` | 融合向量核心 |
| `src/core/engine.py` | 搜尋引擎 |
| `scripts/generate_fused_embeddings.py` | 離線生成腳本 |
| `test_fused_embeddings.py` | 測試套件 |
| `API_KEY_ROTATION_SUMMARY.md` | API Key 輪換文檔 |

---

**實現日期**: 2026-03-07  
**狀態**: ✅ 完成且測試就緒  
**下一步**: 執行 `python scripts/generate_fused_embeddings.py` 生成融合向量
