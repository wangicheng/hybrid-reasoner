# 🎯 Constraint Fidelity Violation 快速參考

## 問題概述

**現象**: 搜尋結果有 45-50% 的@1 和 79-92% 的@10 違反約束條件

```
查詢: "青春歡樂向小說，不要 NTR"
期望 required_tags: ["青春", "歡樂向"]
期望 blocked_tags: ["NTR"]

當前結果 (前 3):
1. 異世界冒險 ❌ (沒有 required_tags)
2. NTR 故事 ❌ (有 blocked_tag)
3. 後宮戀愛 ❌ (沒有 required_tags)

✅ 應該返回:
1. 青春歡樂向戀愛
2. 歡樂向校園
3. 青春溫暖向
```

---

## 根本原因 (5分鐘速度理解)

### 原因 1: Required_tags 沒有硬過濾 (94.6% 的違反)

**流程圖**:
```
搜尋
  ↓
語義相似度 (好) ← required_tags 不在這裡檢查！
  ↓
計分 (標籤只用於評分) ← required_tags 被軟化
  ↓
後過濾 ← ❌ 沒有檢查 required_tags！
  ↓
返回結果 ← 可能缺少 required_tags
```

**代碼證據**:
```python
# src/core/engine.py 第 730 行
def _post_filter(self, scored_items, criteria_list, negative_tag_terms):
    # ✅ 檢查: negative_tags, status, author, words
    # ❌ 缺少: required_tags 檢查！
```

**修復**: 在 _post_filter() 中添加 required_tags 檢查

---

### 原因 2: Blocked_tags 匹配不精確 (4.5-7.8% 的違反)

**問題 1**: 子字符串匹配
```python
# 當前邏輯
if negative_term in book_tag or book_tag in negative_term:
    excluded = True

# 問題:
"NTR" in "NTRG" → True (誤殺)  ❌
"NTR" in "中二" → False (漏掉)  ✓ 這個沒有誤殺
```

**問題 2**: 負面標籤映射不完整
```
LLM 提取 "NTR"
  ↓
映射到 tag_collection (limit=1, threshold=0.7)
  ↓
可能失敗或不完整
```

**修復**: 
1. 使用精確匹配而不是子字符串
2. 增加映射的 limit 和降低 threshold

---

## 快速修復 (適合立即實施)

### 修復 A: 添加 Required_tags 檢查

**位置**: `src/core/engine.py` → `_post_filter()` 方法

**添加這行代碼** (在檢查 negative_tags 後):
```python
# 在 for result in scored_items 循環中添加:
if not excluded and required_tags:
    for req_tag in required_tags:
        book_tags = self._normalize_tags(item.get("tags", []))
        if not any(req_tag == tag for tag in book_tags):
            excluded = True
            break
```

**效果**: Required_tags 違反從 94.6% → ~5%

---

### 修復 B: 改進 Blocked_tags 匹配

**位置**: `src/core/engine.py` → `_post_filter()` 方法

**當前**:
```python
for negative_term in negative_tag_terms:
    if any(
        negative_term in book_tag or book_tag in negative_term
        for book_tag in book_tags
    ):
        excluded = True
```

**改為**:
```python
for negative_term in negative_tag_terms:
    if any(
        negative_term == book_tag  # 精確匹配
        for book_tag in book_tags
    ):
        excluded = True
```

**效果**: Blocked_tags 違反從 5-7% → ~1%

---

## 預期改善

| 指標 | 修復前 | 修復後 | 改善 |
|------|--------|--------|------|
| Required_tags 違反 | 94.6% | ~5% | ⬇ 89% |
| Blocked_tags 違反 | 5-7% | ~1% | ⬇ 4-6% |
| Clean@10 | 20% | ~94% | ⬆ 74% |
| 整體 Viol@10 | 79-92% | ~6% | ⬇ 73-86% |

---

## 相關文件

| 文件 | 內容 |
|------|------|
| [CONSTRAINT_VIOLATION_ANALYSIS.md](CONSTRAINT_VIOLATION_ANALYSIS.md) | 完整技術分析 |
| [CONSTRAINT_VIOLATION_REMEDIATION.md](CONSTRAINT_VIOLATION_REMEDIATION.md) | 詳細修復計劃 |
| [diagnose_violations.py](diagnose_violations.py) | 診斷腳本 |

---

## 驗證方法

### 方法 1: 直接測試

```bash
# 運行診斷腳本
python3 diagnose_violations.py

# 檢查輸出是否顯示:
# - Required_tags 違反大幅減少
# - Blocked_tags 違反接近 0%
```

### 方法 2: 使用測試用例

```bash
# 運行現有測試
python3 -m pytest tests/test_violations.py -v

# 或手動測試
python3 << 'EOF'
from src.core.engine import HybridEngine

engine = HybridEngine()
result = await engine.search("青春歡樂向，不要 NTR")

# 驗證所有結果都沒有違反
assert all("NTR" not in r.get("tags", []) for r in result["results"])
print("✅ 測試通過")
EOF
```

### 方法 3: 運行實驗

```bash
# 生成新的運行結果
python3 -m src.eval.generate_run \
  --queries data/experiments/queries.json \
  --output-dir data/experiments/runs/test_fix \
  --engine HybridEngine

# 分析結果
python3 -m src.eval.ir_metrics \
  --experiment-dir data/experiments/runs/test_fix \
  --ks 1 10
```

---

## FAQ

**Q: 為什麼只有 required_tags 有問題?**  
A: 因為 `_post_filter()` 只檢查 negative_tags, status, author, words，漏掉了 required_tags。

**Q: 修復後無結果查詢會增加嗎?**  
A: 可能會。建議添加降級邏輯來處理無結果情況。

**Q: 這會影響性能嗎?**  
A: 額外的過濾會增加延遲，但應該可以接受 (< 100ms)。

**Q: 為什麼所有引擎都有相同問題?**  
A: 因為這是搜尋引擎的根本設計問題，不是配置問題。

---

## 實施路徑

```
1. 理解問題 (5 分鐘)
   ↓ 讀這份文檔

2. 準備 (5 分鐘)
   ↓ 打開 src/core/engine.py

3. 修復 (10 分鐘)
   ↓ 添加 2 個小改動

4. 測試 (10 分鐘)
   ↓ 運行診斷腳本

5. 驗證 (10 分鐘)
   ↓ 檢查指標改善

總計: ~40 分鐘
```

---

**最後更新**: 2026-05-10  
**狀態**: 準備實施  
**優先級**: 🔴 立即修復
