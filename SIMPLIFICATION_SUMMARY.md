# 混合搜索引擎简化改造总结

## 📅 改造日期
2026-03-10

## 🎯 改造目标
将混合搜索引擎从"多维度规则评分"模式简化为"纯语义搜索 + 硬过滤"模式

---

## ✅ 已完成的修改

### 1. **LLM Prompt 简化** (`src/core/llm.py`)

#### 修改内容：
- ✅ 更新 `full_system_instruction`，明确只支持 3 种过滤函数
- ✅ 简化 `manual_schema`，移除废弃的 keyword_match、numeric_ranking  
- ✅ 更新 Fallback 逻辑，不再生成 keyword_match

#### 新的评分函数体系：
| 函数名 | 用途 | 说明 |
|--------|------|------|
| `semantic_similarity` | ✅ 主要搜索方法 | 正向/负向语义查询 |
| `status_check` | ✅ 硬过滤 | 完结/连载状态 |
| `author_match` | ✅ 硬过滤 | 作者名称（模糊匹配） |
| `numeric_range` | ✅ 硬过滤 | 仅支持 words_total |
| ~~`keyword_match`~~ | ❌ 已废弃 | 由语义搜索替代 |
| ~~`numeric_ranking`~~ | ❌ 已废弃 | 软排名功能移除 |

---

### 2. **引擎核心重构** (`src/core/engine.py`)

#### A. 新增功能：

##### `_build_qdrant_filter()` 方法
- 根据 criteria 构建 Qdrant `rest.Filter` 对象
- 支持的过滤条件：
  - **状态过滤**：`publish_status = "完結"` or `"連載"`
  - **作者过滤**：`author` 包含指定名称（使用 `MatchText` 模糊匹配）
  - **字数过滤**：`words_total` 范围查询（支持 min/max/range）

##### 示例：
```python
# 用户："找一本完結的科幻小說，50萬字以上"
qdrant_filter = rest.Filter(
    must=[
        rest.FieldCondition(key="publish_status", match=rest.MatchValue(value="完結")),
        rest.FieldCondition(key="words_total", range=rest.Range(gte=500000))
    ]
)
```

#### B. 负向语义计算（使用 numpy 优化）

**实现流程：**
1. 正向检索获取候选集（带硬过滤）
2. 对每个负向 `semantic_similarity` criteria：
   - 使用其 `query_text` 做向量嵌入
   - 对候选集进行二次向量查询
   - 归一化负向分数并累加
3. 最终分数 = 正向语义分数 - Σ(负向语义分数)

#### C. 评分逻辑简化

**旧模式（多维度加权）：**
```python
total_score = semantic_score + Σ(rule_score × weight)
# 包含：关键字匹配、数值范围、状态检查、作者匹配等
```

**新模式（纯语义）：**
```python
total_score = positive_semantic_score - Σ(negative_semantic_scores)
# 过滤条件不参与评分，在 Qdrant 层面处理
```

#### D. 移除的功能
- ❌ 标题模糊匹配数据库查询（原 L246-254）
- ❌ 作者数据库查询召回（原 L257-270）
- ❌ 所有规则评分函数调用

---

### 3. **Breakdown 信息格式更新**

#### 新的评分明细结构：
```json
{
  "breakdown": [
    {
      "criteria": "semantic_similarity",
      "label": "語意相似度 (文本×0.7 + 標籤×0.3)",
      "raw_score": 0.52,
      "normalized_score": 0.82,
      "weighted_score": 0.82,
      "is_filter": false,
      "reason": "多向量融合分數"
    },
    {
      "criteria": "semantic_similarity",
      "label": "[排除] 負向語意",
      "raw_score": 0.15,
      "normalized_score": 0.15,
      "weighted_score": -0.15,
      "is_negative": true,
      "is_filter": false,
      "reason": "排除內容相似度: 0.15"
    },
    {
      "criteria": "status_check",
      "label": "[過濾] 狀態: 完結",
      "matched": true,
      "is_filter": true,
      "reason": "已在檢索層過濾（Qdrant Filter）"
    }
  ],
  "final_score": 0.67
}
```

---

## 🔄 工作流程对比

### 旧流程（复杂）
```
用户查询 
  → LLM 解析（生成多种 criteria）
  → 向量检索（无过滤）
  → 标题/作者数据库召回
  → 多维度规则评分
  → 软加权排序
```

### 新流程（简化）
```
用户查询 
  → LLM 解析（只生成semantic + 3种过滤）
  → 构建 Qdrant Filter
  → 向量检索（带硬过滤）
  → 负向语义计算
  →纯语义评分
  → 排序
```

---

## 📊 参数配置

### 多向量权重（不变）
```python
text_weight = 0.7    # 文本语意（书名 + 简介）
tag_weight = 0.3     # 标签语意
```

### 归一化阈值（不变）
```python
min_threshold = 0.35  # 低于此视为 0
max_threshold = 0.65  # 高于此视为 1
```

### 检索限制（优化）
```python
retrieval_limit = 100  # 从 10000 降低到 100（因为有硬过滤）
```

---

## 🎯 优势

1. **性能提升**
   - Qdrant 层面过滤，减少候选集规模
   - 减少 Python 层面的规则计算
   - 检索限制从 10000 降低到 100

2. **逻辑清晰**
   - 主次分明：语义搜索为主，过滤为辅
   - 代码简化：移除复杂的规则评分系统
   - 易于调试：明确区分"评分"和"过滤"

3. **灵活性**
   - 负向语义支持排除需求
   - 硬过滤精确满足约束条件
   - 语义搜索处理所有模糊需求

---

## ⚠️ 注意事项

1. **Qdrant Payload 字段依赖**
   - 必须确保 payload 包含：`publish_status`, `author`, `words_total`
   - 状态字段取值：`"完結"` 或 `"連載"`（中文）

2. **负向语义计算成本**
   - 每个负向条件需要一次额外的向量查询
   - 建议限制负向条件数量（1-2 个）

3. **作者匹配**
   - 使用 `MatchText` 实现模糊匹配
   - "猫腻" 可以匹配 "猫腻"、"猫腻（著）"等

---

## 🧪 测试建议

### 测试用例：
1. ✅ 纯语义搜索："找一本科幻小说"
2. ✅ 状态过滤："找一本完結的GL小說"
3. ✅ 字数范围："找一本20-50萬字的小說"
4. ✅ 字数最小值："找一本100萬字以上的長篇小說"
5. ✅ 负向语义："找一本奇幻小說，不要龍傲天"
6. ✅ 组合过滤："找一本完結的科幻小說，50萬字以上"

### 验证点：
- [ ] LLM 是否正确解析过滤条件
- [ ] Qdrant filter 是否正确构建
- [ ] 负向语义是否生效
- [ ] Breakdown 信息是否清晰
- [ ] 性能是否提升

---

## 📝 后续优化方向

1. **动态权重调整**
   ```python
   # 根据查询类型动态调整 text_weight 和 tag_weight
   if query_type == "type_focused":
       text_weight, tag_weight = 0.6, 0.4
   ```

2. **负向语义优化**
   - 使用向量缓存减少重复计算
   - 批量计算多个负向查询

3. **过滤条件扩展**
   - 支持评分范围过滤（rating_score）
   - 支持收藏数过滤（bookmark_count）

---

## 📚 相关文件

- `src/core/llm.py` - LLM查询解析
- `src/core/engine.py` - 混合引擎核心
- `src/core/vector_store.py` - 向量存储（未修改）
- `src/logic/scoring_functions.py` - 评分函数（保留但不使用）
- `test_simplified_search.py` - 测试脚本

---

## ✨ 总结

此次改造成功将混合搜索引擎从复杂的"规则评分"模式简化为"语义搜索 + 硬过滤"模式，核心优势为：

- **主要依赖语义向量**（准确度高）
- **硬过滤精确执行约束**（性能好）
- **负向语义支持排除**（灵活性强）

改造后的系统更加符合现代向量搜索的最佳实践！
