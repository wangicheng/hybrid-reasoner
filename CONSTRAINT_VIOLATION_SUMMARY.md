# 📊 Constraint Fidelity Violation 分析完成報告

## 執行摘要

### 問題描述
BM25 Hybrid Reasoner 搜尋引擎在滿足用戶約束條件時表現不佳：
- **Required_tags 違反**: 94.6-95.5% (非常嚴重)
- **Blocked_tags 違反**: 4.5-7.8% (中等)
- **Clean@10**: 只有 8-20% 的結果完全滿足約束

### 根本原因
1. **Required_tags 沒有硬過濾**: 在 `_post_filter()` 中缺少檢查
2. **Blocked_tags 匹配不精確**: 子字符串匹配 + 映射不完整

### 修復難度
- **優先級 1 (立即)**: 2 個小改動，~3 小時，預期效果 93% 改善
- **優先級 2 (中期)**: 標籤映射和別名，~4 小時
- **優先級 3 (長期)**: 分層設計，~8 小時

---

## 交付物清單

本分析生成了 **4 份主要文檔** 和 **1 個診斷腳本**:

### 📄 文檔 (4 份)

#### 1. [CONSTRAINT_VIOLATION_ANALYSIS.md](CONSTRAINT_VIOLATION_ANALYSIS.md)
**內容**: 完整的技術分析報告  
**長度**: ~500 行  
**包含**:
- 現狀快照和數據分析
- 根本原因詳細分析 (2 個主要問題)
- 代碼位置引用
- 深層原因根源分析
- 直接原因確認
- 修復建議 (立即/中期/長期)

**適用於**: 技術領導、開發者、架構師

---

#### 2. [CONSTRAINT_VIOLATION_REMEDIATION.md](CONSTRAINT_VIOLATION_REMEDIATION.md)
**內容**: 詳細的修復實施計劃  
**長度**: ~600 行  
**包含**:
- 5 項修復 (優先級排序)
- 具體代碼改動示例
- 完整的實施步驟 (Step by Step)
- 測試計劃和驗收標準
- 時間表和風險評估
- 檢查清單

**適用於**: 開發者、QA、項目經理

---

#### 3. [CONSTRAINT_VIOLATION_QUICK_REFERENCE.md](CONSTRAINT_VIOLATION_QUICK_REFERENCE.md)
**內容**: 快速理解指南  
**長度**: ~200 行  
**包含**:
- 5 分鐘問題理解
- 根本原因簡化解釋
- 快速修復 (2 個立即修復)
- 預期改善表格
- 驗證方法
- FAQ

**適用於**: 所有人 (尤其是首次接觸)

---

#### 4. [CONSTRAINT_VIOLATION_QUICK_SUMMARY.txt] (本文檔)
**內容**: 分析完成報告  
**長度**: ~300 行  
**包含**:
- 整體摘要
- 交付物清單
- 後續建議
- 相關鏈接

**適用於**: 管理層、決策者

---

### 🐍 診斷腳本 (1 個)

#### [diagnose_violations.py](diagnose_violations.py)
**功能**: 自動診斷約束違反原因  
**特性**:
- 加載最新運行結果
- 逐個查詢分析違反
- 分類統計違反類型
- 生成詳細診斷報告

**使用**:
```bash
python3 diagnose_violations.py
```

---

## 🎯 關鍵發現

### 發現 1: Required_tags 沒有被硬過濾

**代碼位置**: `src/core/engine.py` 第 730 行

**當前流程**:
```
1. VectorStore 搜尋 (無標籤約束)
2. calculate_score() (required_tags 只用於評分)
3. _post_filter() ← 缺少 required_tags 檢查！
4. 返回結果 ← 可能有違反
```

**影響**: 94.6-95.5% 的查詢有違反

**修復**: 在 _post_filter() 中添加 3 行代碼

---

### 發現 2: Blocked_tags 匹配邏輯不精確

**代碼位置**: `src/core/engine.py` 第 712-718 行

**問題**:
1. 子字符串匹配過於寬鬆
2. 負面標籤映射不完整 (limit=1, threshold=0.7 過於嚴格)

**影響**: 4.5-7.8% 的查詢有違反

**修復**: 改進匹配邏輯 + 調整映射參數

---

### 發現 3: 所有引擎都有相同問題

**統計**:
```
auto_t2_dynamic_routing_sameparse:    Viol@10=79.17%
baseline_weighted_ws35_wa65_sameparse: Viol@10=79.17%
weighted_ws35_wa65_sameparse:          Viol@10=79.17%
weighted_ws60_wa40_sameparse:          Viol@10=79.17%
auto_t3_dynamic_routing_sameparse:     Viol@10=83.33%
rrf_k60_with_bm25_sameparse:           Viol@10=91.67%
auto_llm_routing_sameparse:            Viol@10=91.67%
rrf_k60_no_bm25_sameparse:             Viol@10=91.67%
```

**結論**: 這是**系統性問題** (搜尋引擎設計)，不是配置問題

---

## 📈 預期改善

| 指標 | 當前 | 修復後 | 改善 |
|------|------|--------|------|
| Required_tags 違反 | 94.6% | ~5% | ⬇ 89% |
| Blocked_tags 違反 | 5-7% | ~1% | ⬇ 4-6% |
| 整體 Viol@10 | 79-92% | ~6% | ⬇ 73-86% |
| Clean@10 | 8-20% | ~94% | ⬆ 74% |

---

## 🚀 後續行動

### 立即 (這週)

- [ ] **審查分析報告** (30 分鐘)
  - 閱讀 [CONSTRAINT_VIOLATION_QUICK_REFERENCE.md](CONSTRAINT_VIOLATION_QUICK_REFERENCE.md)
  - 檢查根本原因是否準確

- [ ] **準備修復** (1 小時)
  - 備份 `src/core/engine.py`
  - 創建特性分支 `fix/constraint-fidelity`

- [ ] **實施修復 1.1** (1 小時)
  - 在 _post_filter() 添加 required_tags 檢查
  - 運行 diagnose_violations.py 驗證

- [ ] **實施修復 1.2** (30 分鐘)
  - 改進 blocked_tags 匹配邏輯
  - 運行基本測試

### 本週 (完成優先級 1)

- [ ] **完整測試** (1 小時)
  - 運行 ir_metrics 分析
  - 驗證 Viol@10 < 10%

- [ ] **性能檢查** (30 分鐘)
  - 確認延遲增長 < 100ms
  - 檢查無結果查詢比例

- [ ] **代碼審查** (1 小時)
  - 提交 PR
  - 獲得批准

### 下週 (優先級 2)

- [ ] **改進負面標籤映射** (1 小時)
  - 調整 limit 和 threshold
  - 測試映射準確性

- [ ] **添加標籤別名字典** (2 小時)
  - 創建 tag_aliases.py
  - 集成到匹配邏輯

---

## 💡 建議

### 建議 1: 添加降級邏輯

修復後，某些嚴格的查詢可能無結果。建議添加自動降級:

```
如果 required_tags 搜尋無結果 (嘗試 1):
  → 降級到 required_tags 部分匹配 (嘗試 2)
  → 降級到只檢查 blocked_tags (嘗試 3)
```

這與已有的 _apply_degradation_step() 配合。

---

### 建議 2: 建立標籤數據質量檢查

建議定期檢查 tag_collection 的質量：

```bash
# 檢查缺失的標籤
python3 << 'EOF'
from src.core.vector_store import VectorStore
vs = VectorStore()
all_tags = vs.get_all_tags()
# 檢查是否包含所有 queries.json 中使用的標籤
EOF
```

---

### 建議 3: 建立持續監控

建議添加指標監控:

```python
# 在每次運行後計算
metrics = {
    "required_tags_violation_rate": ...,
    "blocked_tags_violation_rate": ...,
    "clean_at_10": ...,
    "avg_degradation_attempts": ...,
}
```

---

## 📚 相關資源

### 現有代碼

- [src/core/engine.py](src/core/engine.py) - 主搜尋引擎 (第 230-730 行)
- [src/core/vector_store.py](src/core/vector_store.py) - 向量存儲
- [src/eval/tag_rules.py](src/eval/tag_rules.py) - 標籤規則
- [src/eval/analyze_violations.py](src/eval/analyze_violations.py) - 違反分析

### 數據

- [data/experiments/queries.json](data/experiments/queries.json) - 測試查詢
- [data/all_tags.json](data/all_tags.json) - 標籤集合

### 配置

- [data/experiments/runs/batch_*/results.json](data/experiments/runs) - 運行結果

---

## 📞 支援

### 問題排查

| 問題 | 解決方案 |
|------|---------|
| "如何驗證根本原因準確性?" | 運行 diagnose_violations.py |
| "修復後哪些測試需要通過?" | 查看 CONSTRAINT_VIOLATION_REMEDIATION.md 的測試計劃 |
| "為什麼修復後無結果增加?" | 這是預期行為，使用降級邏輯 |
| "如何測試性能影響?" | 比較修復前後的平均延遲 |

### 額外資源

- 技術分析: [CONSTRAINT_VIOLATION_ANALYSIS.md](CONSTRAINT_VIOLATION_ANALYSIS.md)
- 實施指南: [CONSTRAINT_VIOLATION_REMEDIATION.md](CONSTRAINT_VIOLATION_REMEDIATION.md)
- 快速參考: [CONSTRAINT_VIOLATION_QUICK_REFERENCE.md](CONSTRAINT_VIOLATION_QUICK_REFERENCE.md)

---

## 📋 檢查清單

### 分析階段 ✅
- [x] 識別根本原因
- [x] 分析代碼流程
- [x] 確認系統性問題
- [x] 生成診斷報告

### 規劃階段
- [ ] 審批修復計劃
- [ ] 評估時間和資源
- [ ] 計劃測試策略
- [ ] 準備回滾方案

### 實施階段
- [ ] 實施修復 1.1
- [ ] 實施修復 1.2
- [ ] 運行本地測試
- [ ] 提交代碼審查

### 驗證階段
- [ ] 指標驗證 (Viol@10 < 10%)
- [ ] 性能測試 (延遲 < 500ms)
- [ ] 無結果查詢檢查
- [ ] 回歸測試

### 發佈階段
- [ ] 發佈到測試環境
- [ ] 發佈到生產環境
- [ ] 監控指標
- [ ] 文檔更新

---

## 📅 時間軸

```
Day 1 (審批和準備)
├─ 上午: 分析審批 (1h)
├─ 下午: 環境準備 (1h)
└─ 晚上: 根因確認 (1h)

Day 2-3 (優先級 1 實施)
├─ 修復 1.1 實施 (2h)
├─ 修復 1.2 實施 (1h)
├─ 單元測試 (1h)
└─ 集成測試 (2h)

Day 4 (驗證和發佈)
├─ 指標驗證 (1h)
├─ 性能測試 (1h)
├─ 代碼審查 (1h)
└─ 發佈 (1h)

Day 5 (優先級 2 準備)
├─ 修復 2.1 計劃 (1h)
└─ 修復 2.2 計劃 (1h)
```

---

## 🎊 總結

本分析識別並文檔化了 BM25 Hybrid Reasoner 的 **Constraint Fidelity 違反問題**的根本原因。

### 關鍵成果
- ✅ 確定根本原因 (2 個主要問題)
- ✅ 提供詳細的代碼分析
- ✅ 制定修復計劃 (5 項修復)
- ✅ 估算預期效果 (93% 改善)
- ✅ 準備實施資源 (4 份文檔 + 1 個腳本)

### 下一步
1. 審批分析報告
2. 實施優先級 1 修復
3. 驗證指標改善
4. 考慮優先級 2-3 改進

---

**分析完成日期**: 2026-05-10  
**分析工作量**: ~4 小時  
**預期實施工作量**: ~7 小時  
**預期ROI**: 93% 約束違反率改善

---

📧 **如有問題，請參考相關文檔或運行診斷腳本**
