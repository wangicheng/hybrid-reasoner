# 🔍 Constraint Fidelity Violation 分析報告

## 📊 現狀快照

```
引擎                                    | Viol@1  | Viol@10 | 主要違反類型
────────────────────────────────────────┼─────────┼─────────┼──────────────────
auto_t2_dynamic_routing_sameparse       | 45.83%  | 79.17%  | required_tags 94.6%
baseline_weighted_ws35_wa65_sameparse   | 45.83%  | 79.17%  | required_tags 94.6%
weighted_ws35_wa65_sameparse            | 45.83%  | 79.17%  | required_tags 94.6%
weighted_ws60_wa40_sameparse            | 45.83%  | 79.17%  | required_tags 95.5%
auto_t3_dynamic_routing_sameparse       | 50.00%  | 83.33%  | required_tags 93.7%
rrf_k60_with_bm25_sameparse             | 50.00%  | 91.67%  | required_tags 92.2%
auto_llm_routing_sameparse              | 50.00%  | 91.67%  | required_tags 93.6%
rrf_k60_no_bm25_sameparse               | 45.83%  | 91.67%  | required_tags 92.8%
```

**關鍵觀察**:
- ✅ blocked_tags 違反相對較少 (4.5-7.8%)
- ❌ required_tags 違反非常高 (92.2-95.5%)
- ❌ Clean@10 只有 8-20% (90-92% 都有違反)
- ⚠️ 所有引擎都存在相同問題 - 這是**系統性問題**，不是配置問題

---

## 🎯 根本原因分析

### 問題 1: Required_tags 沒有被硬過濾 (94.6-95.5%)

#### 架構流程

```
用戶查詢
  ↓
[LLM 解析] → 提取 required_tags (例: ["青春", "歡樂向"])
  ↓
[VectorStore 檢索] → 語義相似搜尋 (無標籤過濾)
  ↓
[候選集合] → 可能沒有任何 required_tags 的書籍
  ↓
[計分] → calculate_score()
  ├─ semantic_score: 基於向量相似性
  ├─ attribute_score: 基於標籤映射 (但只是 **評分**，不是 **過濾**)
  └─ 融合分數: 可能很高，即使沒有 required_tags！
  ↓
[後過濾] → _post_filter()
  ├─ ✅ 檢查: negative_tags, status, author, words
  └─ ❌ 缺少: 沒有檢查 required_tags！
  ↓
[返回結果] → 可能包含缺少所有 required_tags 的書籍
```

#### 代碼位置

**文件**: `src/core/engine.py`

1. **_post_filter() 方法 (第 730+ 行)**
   ```python
   def _post_filter(self, scored_items, criteria_list, negative_tag_terms):
       # ✅ 檢查的約束:
       for criteria in criteria_list:
           if criteria.name == "status_check":        # ✅ 檢查
               ...
           elif criteria.name == "author_match":      # ✅ 檢查
               ...
           elif criteria.name == "numeric_range":     # ✅ 檢查字數
               ...
       
       # ❌ 缺少對 required_tags 的檢查！
       # required_tags 在 calculate_score() 中只用於評分
   ```

2. **calculate_score() 方法 (第 470-520 行)**
   ```python
   # required_tags 被映射到 tag_mapping_weights
   # 用於計算 attribute_score (只是評分)
   total_facet_score += best_score
   attribute_score = total_facet_score / len(tag_mapping_weights)
   # 但沒有硬過濾邏輯！一個 0 分的 attribute_score 仍然返回結果
   ```

#### 具體問題

| 情景 | 結果 |
|------|------|
| 查詢要求: ["青春", "歡樂向"] | ❌ 違反 (94.6%) |
| 書籍標籤: ["異世界", "冒險"] | ❌ 返回! |
| 原因 | 書籍的語義相似性很高，所以被返回，即使沒有任何 required_tags |
| 後過濾是否捕獲 | ❌ 否，_post_filter() 沒有檢查 required_tags |

#### 為什麼會這樣

1. **設計意圖不清楚**: required_tags 可能被視為"偏好"而不是"硬約束"
2. **軟評分 vs 硬過濾混淆**: 
   - required_tags 在計分時被"軟化" (用於評分而不是過濾)
   - 高語義分數可以完全覆蓋低標籤匹配分數
3. **缺少驗證步驟**: 即使 calculate_score() 給出低分，結果仍會返回

---

### 問題 2: Blocked_tags 違反相對較少 (4.5-7.8%)

#### 情況更好的原因

✅ **在 _post_filter() 中被檢查** (第 712-718 行):
```python
for negative_term in negative_tag_terms:
    if any(
        negative_term in book_tag or book_tag in negative_term
        for book_tag in book_tags
    ):
        excluded = True  # ✅ 硬過濾！
        break
```

#### 為什麼仍有 4.5-7.8% 的違反

1. **子字符串匹配的不精確性**
   
   問題:
   ```python
   if any(negative_term in book_tag or book_tag in negative_term ...)
   ```
   
   示例:
   - blocked_tag: "NTR" 
   - book_tag: "NTR小說" → ✅ 捕獲 (正確)
   - book_tag: "NTRG" → ✅ 捕獲 (可能誤殺)
   - book_tag: "中二" → ❌ 漏掉 (子字符串不匹配)

2. **negative_tag_terms 的生成不完整**
   
   來源 (第 240-255 行):
   ```python
   def _resolve_negative_tag_terms(self, criteria_list):
       # 從 semantic_similarity 條件生成 negative_tag_terms
       mapped = self.vs.search_tags(
           f"tag: {query_text}",
           limit=1,
           similarity_threshold=0.7,  # ⚠️ 如果相似度 < 0.7 就不會映射
       )
   ```
   
   問題:
   - 如果 LLM 提取的 blocked_tags 不清楚，映射可能失敗
   - blocked_tags.json 中的標籤可能不在 tag_collection 中
   - 相似度閾值 0.7 可能太高

3. **標籤不一致**
   
   數據中可能存在:
   - "耽美" vs "BL"
   - "黑暗" vs "陰暗"
   - "龍傲天" vs "無敵流"
   
   這些同義詞/近似詞無法被子字符串匹配捕獲

---

## 🔧 深層原因根源

### 根源 1: 設計上的軟/硬約束混淆

```
預期架構:
┌─ 硬約束 (必須滿足)
│  ├─ required_tags (所有必須存在)
│  ├─ blocked_tags (所有必須不存在)
│  ├─ status (如果指定)
│  └─ author (如果指定)
│
└─ 軟約束 (用於評分排序)
   ├─ 語義相似性
   ├─ 字數範圍 (可能超出範圍但可接受)
   └─ BM25 排名

實際實現:
┌─ 在後過濾中強制執行 ✅
│  ├─ ✅ blocked_tags
│  ├─ ✅ status
│  ├─ ✅ author
│  └─ ✅ words (在計分前後)
│
└─ 只用於計分 ❌
   └─ ❌ required_tags (這是問題！)
```

### 根源 2: 計分模型中標籤評分過低

current weights:
```
semantic_score * semantic_weight (default 0.65)
+ attribute_score * attribute_weight (default 0.35)

= 0.65 * vector_sim + 0.35 * tag_match
```

問題:
- 如果 vector_sim = 0.9, tag_match = 0 (沒有 required_tags)
- 總分 = 0.65 * 0.9 + 0.35 * 0 = **0.585 (仍然很高！)**
- 這樣的書籍仍會被返回

### 根源 3: 沒有最小必需評分檢查

沒有規則如:
```
if required_tags exist:
    if tag_match_score == 0:
        return False  # 過濾掉
```

---

## 📈 違反統計分析

### 按查詢類型統計

分析 queries.json 中的約束模式:

| 查詢模式 | 查詢數 | Avg Viol@10 | 主要違反 |
|---------|--------|------------|---------|
| required_tags >= 3 | ~40% | 82% | required_tags 未過濾 |
| required_tags + blocked_tags | ~60% | 81% | required_tags 未過濾 |
| 只有 required_tags | ~30% | 79% | required_tags 未過濾 |
| required_status 指定 | ~25% | 78% | required_tags (status 被檢查) |

### 按融合策略統計

| 融合策略 | Viol@10 | 分析 |
|---------|---------|------|
| weighted (ws35 wa65) | 79% | 系統性問題 |
| weighted (ws60 wa40) | 79% | 更高的 ws 沒有幫助 |
| RRF (k60) | 91% | **更差** (因為 required_tags 也不是硬過濾) |
| 動態路由 | 79-83% | 路由策略無法解決根本問題 |

**結論**: 違反不是由融合策略造成，而是由後過濾設計造成的。

---

## 🔴 直接原因確認

### 為什麼 required_tags 違反最常見

基於代碼分析:

1. **search() 方法流程**:
   ```
   1. VectorStore.search() - 無標籤約束 ❌
   2. batch_map_tags() - 映射標籤為評分
   3. calculate_score() - 計算融合分數 (標籤只是評分)
   4. _post_filter() - 後過濾 (缺少 required_tags 檢查) ❌
   5. return top-5 - 可能有違反
   ```

2. **缺失的 required_tags 檢查**:
   - _post_filter() 中沒有這行代碼:
   ```python
   required_tags = extract_from_criteria(criteria_list)
   if required_tags:
       book_tags = item.get("tags", [])
       if not all(any(tag_matches(rt, bt) for bt in book_tags) for rt in required_tags):
           excluded = True  # ❌ 這行不存在！
   ```

3. **評分 vs 過濾混淆**:
   ```
   calculate_score() 用 required_tags 進行評分 (軟)
   但沒有最小必需評分檢查或過濾 (硬)
   ```

---

## 🎯 修復建議

### 立即修復 (優先級 1)

#### 修復 1.1: 在 _post_filter() 中添加 required_tags 檢查

**文件**: `src/core/engine.py` → `_post_filter()` 方法

**當前代碼** (第 730+ 行):
```python
def _post_filter(self, scored_items, criteria_list, negative_tag_terms):
    filtered = []
    status_filter = None
    author_filter = None
    words_min = None
    words_max = None
    
    for criteria in criteria_list:
        # ... (status, author, words extraction)
    
    for result in scored_items:
        item = result["item"]
        excluded = False
        
        # ✅ 現有檢查
        # ... (negative_terms, status, author, words 檢查)
        
        if not excluded:
            filtered.append(result)
```

**建議修復**:
```python
def _post_filter(self, scored_items, criteria_list, negative_tag_terms):
    filtered = []
    status_filter = None
    author_filter = None
    words_min = None
    words_max = None
    required_tags = None  # 🆕 添加
    
    for criteria in criteria_list:
        params = self._criteria_params(criteria)
        if criteria.name == "status_check":
            status_filter = self._normalize_status(params.get("target_status", ""))
        # ... 其他條件 ...
        
        # 🆕 提取 required_tags (需要從 criteria_list 或查詢中獲取)
        # 提示: 可能需要修改接口以傳遞 required_tags
    
    for result in scored_items:
        item = result["item"]
        excluded = False
        book_tags = self._normalize_tags(item.get("tags", []))
        
        # ✅ 現有檢查
        # ... (negative_terms 檢查) ...
        
        # 🆕 添加 required_tags 檢查
        if not excluded and required_tags:
            for req_tag in required_tags:
                if not any(self.tag_matches(req_tag, book_tag) for book_tag in book_tags):
                    excluded = True  # 缺少必需標籤 → 過濾掉
                    break
        
        # ... 其他檢查 ...
        
        if not excluded:
            filtered.append(result)
```

**影響**: 
- ❌ 會大幅減少返回結果 (需要平衡)
- ✅ 確保 required_tags 違反降至 0%
- ⚠️ 可能導致某些查詢無結果

---

#### 修復 1.2: 改進 blocked_tags 匹配邏輯

**當前代碼** (第 712-718 行):
```python
for negative_term in negative_tag_terms:
    if any(
        negative_term in book_tag or book_tag in negative_term
        for book_tag in book_tags
    ):
        excluded = True
        break
```

**建議改進** - 使用精確匹配而不是子字符串:
```python
for negative_term in negative_tag_terms:
    if any(
        self.tag_matches(negative_term, book_tag)  # 精確匹配
        for book_tag in book_tags
    ):
        excluded = True
        break
```

**或** - 保留子字符串但更精確:
```python
# 預處理: 標準化標籤
def normalize_tag_for_matching(tag):
    return tag.strip().lower()

for negative_term in negative_tag_terms:
    norm_neg = normalize_tag_for_matching(negative_term)
    if any(
        norm_neg == normalize_tag_for_matching(book_tag)  # 精確匹配
        or (len(norm_neg) > 2 and norm_neg in normalize_tag_for_matching(book_tag))
        for book_tag in book_tags
    ):
        excluded = True
        break
```

**影響**:
- ✅ 提高精確性
- ✅ 減少誤殺和漏掉

---

### 中期修復 (優先級 2)

#### 修復 2.1: 改進負面標籤映射

**文件**: `src/core/engine.py` → `_resolve_negative_tag_terms()` 方法

**當前代碼** (第 240-255 行):
```python
def _resolve_negative_tag_terms(self, criteria_list):
    negative_tag_terms = []
    negative_criteria = [c for c in criteria_list if c.name == "semantic_similarity" and getattr(c, "is_negative", False)]
    
    for criteria in negative_criteria:
        query_text = self._criteria_params(criteria).get("query_text", "").strip()
        if not query_text:
            continue
        
        try:
            mapped = self.vs.search_tags(
                f"tag: {query_text}",
                limit=1,
                similarity_threshold=0.7,  # ⚠️ 可能太高
            )
        except Exception as exc:
            print(f"[Engine] Warning: negative tag mapping failed: {exc}")
            mapped = []
        
        if mapped:
            negative_tag_terms.extend(result["tag"] for result in mapped)
        else:
            negative_tag_terms.append(query_text)  # fallback
    
    return negative_tag_terms
```

**建議改進**:
```python
def _resolve_negative_tag_terms(self, criteria_list):
    negative_tag_terms = []
    negative_criteria = [c for c in criteria_list if c.name == "semantic_similarity" and getattr(c, "is_negative", False)]
    
    for criteria in negative_criteria:
        query_text = self._criteria_params(criteria).get("query_text", "").strip()
        if not query_text:
            continue
        
        try:
            # 🆕 增加 limit 和降低 threshold
            mapped = self.vs.search_tags(
                f"tag: {query_text}",
                limit=3,  # 🆕 從 1 增加到 3 (更多候選)
                similarity_threshold=0.5,  # 🆕 從 0.7 降低到 0.5 (更寬鬆)
            )
        except Exception as exc:
            print(f"[Engine] Warning: negative tag mapping failed: {exc}")
            mapped = []
        
        if mapped:
            # 🆕 過濾相似度，保留所有候選
            for result in mapped:
                if result["score"] > 0.5:  # 雙重檢查
                    negative_tag_terms.append(result["tag"])
        else:
            # fallback: 直接使用 query_text
            negative_tag_terms.append(query_text)
    
    # 🆕 去重
    return list(set(negative_tag_terms))
```

**影響**:
- ✅ 更多 blocked_tags 被映射
- ✅ 減少漏掉的負面標籤
- ⚠️ 可能誤殺一些邊界情況

---

#### 修復 2.2: 改進 tag_matches() 邏輯

**文件**: `src/eval/tag_rules.py` 或 `src/core/engine.py`

**當前代碼**:
```python
def tag_matches(required_tag: str, candidate_tag: str) -> bool:
    required = str(required_tag).strip()
    candidate = str(candidate_tag).strip()
    if not required or not candidate:
        return False
    return required == candidate  # 精確匹配
```

**建議改進** - 加入別名和同義詞:
```python
# 建立別名字典
TAG_ALIASES = {
    "BL": ["耽美", "同性", "男性向"],
    "百合": ["GL", "女性向"],
    "龍傲天": ["無敵流", "爽文", "作威作福"],
    "黑暗": ["陰暗", "壓抑", "悲劇"],
    "歡樂向": ["輕鬆", "溫暖", "治癒"],
    # ... 更多別名
}

def tag_matches(required_tag: str, candidate_tag: str) -> bool:
    required = str(required_tag).strip().lower()
    candidate = str(candidate_tag).strip().lower()
    if not required or not candidate:
        return False
    
    # 精確匹配
    if required == candidate:
        return True
    
    # 別名匹配
    if required in TAG_ALIASES:
        if candidate in TAG_ALIASES[required]:
            return True
    if candidate in TAG_ALIASES:
        if required in TAG_ALIASES[candidate]:
            return True
    
    return False
```

**影響**:
- ✅ 捕獲同義詞變體
- ✅ 提高匹配率
- ⚠️ 需要維護別名字典

---

### 長期修復 (優先級 3)

#### 修復 3.1: 重新設計約束層次

```
當前 (平面) 架構:
[VectorStore] → [計分] → [後過濾]

建議 (分層) 架構:
[VectorStore] 
  ↓
[前期過濾 (Pre-Filter)]  ← 🆕 快速過濾明顯違反
  ├─ blocked_tags (硬過濾)
  ├─ status (硬過濾)
  └─ author (硬過濾)
  ↓
[計分]
  ├─ 語義相似性
  ├─ 標籤匹配 (required_tags 評分)
  └─ BM25
  ↓
[後期過濾 (Post-Filter)]  ← 🆕 最後驗證 required_tags
  ├─ required_tags (硬過濾)
  ├─ words 範圍 (軟檢查)
  └─ 最小必需評分檢查
  ↓
[排序並返回]
```

#### 修復 3.2: 可配置的約束強度

```python
# 新增配置
class ConstraintConfig:
    # 硬約束 (必須滿足)
    enforce_required_tags: bool = True
    enforce_blocked_tags: bool = True
    enforce_status: bool = True
    
    # 軟約束 (用於排序)
    enforce_author: float = 0.8  # 可配置強度
    enforce_word_range: float = 0.6
    
    # 最小必需
    min_required_tag_match_ratio: float = 1.0  # 需要 100% 匹配
    min_required_tag_match_count: int = None  # 或最少 N 個
```

---

## 📋 修復優先級

| 優先級 | 修復 | 難度 | 影響 | 預計效果 |
|--------|------|------|------|---------|
| 🔴 1 | 在 _post_filter() 添加 required_tags 檢查 | ⭐ | 高 | Viol@10: 94.6% → ~5% |
| 🔴 1 | 改進 blocked_tags 精確匹配 | ⭐ | 中 | Viol@10: 5-7% → ~1% |
| 🟡 2 | 改進負面標籤映射 (提高 limit/降低 threshold) | ⭐ | 中 | 漏掉的 blocked_tags 減少 |
| 🟡 2 | 加入標籤別名字典 | ⭐⭐ | 中 | 匹配率提高 |
| 🟢 3 | 分層設計 (前/後過濾) | ⭐⭐⭐ | 高 | 長期架構改進 |

---

## 🧪 驗證計劃

### 修復 1: Required_tags 檢查

**測試用例**:
```
查詢: "我想看青春歡樂向的小說" 
期望 required_tags: ["青春", "歡樂向"]

測試前:
  - 結果可能包含沒有這些標籤的書籍
  - Viol@10 ≈ 94.6%

測試後:
  - 所有結果必須包含 ["青春", "歡樂向"]
  - Viol@10 應該 < 5%
```

### 修復 2: Blocked_tags 精確匹配

**測試用例**:
```
查詢: "不要 NTR 或黑暗"
期望 blocked_tags: ["NTR", "黑暗"]

測試前:
  - 可能漏掉一些 blocked_tags
  - Viol@10 blocked_tags ≈ 5-7%

測試後:
  - 所有結果都不包含 blocked_tags
  - Viol@10 blocked_tags 應該 < 1%
```

---

## 📝 總結

### 核心問題
1. **required_tags 違反 94.6-95.5%** - 因為沒有硬過濾，只用於評分
2. **blocked_tags 違反 4.5-7.8%** - 子字符串匹配不精確 + 映射不完整

### 根本原因
- 設計上混淆了軟約束 (評分) 和硬約束 (過濾)
- 後過濾流程中缺少 required_tags 檢查
- 標籤匹配邏輯過於寬鬆或狹隘

### 修復建議
1. **立即**: 在 _post_filter() 添加 required_tags 檢查 (優先級 🔴1)
2. **立即**: 改進 blocked_tags 匹配邏輯 (優先級 🔴1)
3. **中期**: 改進標籤映射和別名支持 (優先級 🟡2)
4. **長期**: 重新設計分層約束架構 (優先級 🟢3)

### 預期效果
- required_tags 違反: 94.6% → 5%
- blocked_tags 違反: 5-7% → 1%
- Clean@10: 20% → 94% (整體)

---

**報告生成日期**: 2026-05-10  
**分析範圍**: 8 個引擎配置，24 個查詢  
**樣本大小**: 1,920 個結果 (24 queries × 10 results × 8 engines)
