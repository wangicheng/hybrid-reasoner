# Tag Template Embedding 模型比較

## 實驗設定

- 評估集：40 個 evaluation labels、408 個 mapping queries；候選池為 `data/all_tags.json` 中的 96 個 candidate tags。需注意這 96 個 tag 含有部分文庫、平台與來源類 metadata tag，不能直接視為 96 個小說內容標籤。
- 固定 query template：`tag: {label}`
- query task type：`RETRIEVAL_QUERY`
- candidate task type：`RETRIEVAL_DOCUMENT`
- 比較模型：`gemini-embedding-001`、`models/gemini-embedding-2-preview`

## 結論摘要

`models/gemini-embedding-2-preview` 仍然優於 `gemini-embedding-001`，但這次差距比舊版結果更小。以最佳 template 計算，Top-1 Accuracy 由 0.8284 提升到 0.8431，增加 0.0147；Top-5 Accuracy 由 0.9534 提升到 0.9632，增加 0.0098；MRR 由 0.8874 提升到 0.8951，增加 0.0077；Macro-F1 則由 0.6107 提升到 0.6738，增加 0.0631。

這次新版評估集下，兩個模型的高分 template 都明顯集中在帶有「題材／設定／內容走向」語境的描述句，而不是裸 `label`。其中 `gemini-embedding-001` 的最佳 template 是 `這部小說採用了大量的{label}題材`，`models/gemini-embedding-2-preview` 的最佳 template 則變成 `這部小說的世界觀與設定帶有{label}色彩`。

## 最佳 Template 比較

| 模型 | 最佳 template name | 實際 template 字串 | Top-1 | Top-5 | MRR | Macro-F1 | Micro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemini-embedding-001` | `novel_material` | `這部小說採用了大量的{label}題材` | 0.8284 | 0.9534 | 0.8874 | 0.6107 | 0.8284 |
| `models/gemini-embedding-2-preview` | `novel_setting` | `這部小說的世界觀與設定帶有{label}色彩` | 0.8431 | 0.9632 | 0.8951 | 0.6738 | 0.8431 |

## 前五名 Template

### `gemini-embedding-001`

| Rank | Template name | 實際 template 字串 | Top-1 | Top-5 | MRR |
| --- | --- | --- | --- | --- | --- |
| 1 | `novel_material` | `這部小說採用了大量的{label}題材` | 0.8284 | 0.9534 | 0.8874 |
| 2 | `novel_genre` | `這部作品的類型偏向{label}` | 0.8211 | 0.9657 | 0.8796 |
| 3 | `novel_content_direction` | `這部作品的內容走向偏向{label}` | 0.8211 | 0.9485 | 0.8786 |
| 4 | `novel_style` | `這是一部帶有{label}元素的小說` | 0.8113 | 0.9461 | 0.8737 |
| 5 | `novel_core_element` | `這部作品的核心元素之一是{label}` | 0.8113 | 0.9461 | 0.8729 |

### `models/gemini-embedding-2-preview`

| Rank | Template name | 實際 template 字串 | Top-1 | Top-5 | MRR |
| --- | --- | --- | --- | --- | --- |
| 1 | `novel_setting` | `這部小說的世界觀與設定帶有{label}色彩` | 0.8431 | 0.9632 | 0.8951 |
| 2 | `novel_genre` | `這部作品的類型偏向{label}` | 0.8382 | 0.9755 | 0.8959 |
| 3 | `novel_color` | `這部作品整體帶有濃厚的{label}色彩` | 0.8333 | 0.9730 | 0.8912 |
| 4 | `novel_material` | `這部小說採用了大量的{label}題材` | 0.8309 | 0.9632 | 0.8913 |
| 5 | `novel_content_direction` | `這部作品的內容走向偏向{label}` | 0.8284 | 0.9730 | 0.8904 |

## 觀察

- `models/gemini-embedding-2-preview` 的最佳表現仍然全面領先，但優勢主要集中在 `Macro-F1`，表示它對各標籤的整體平衡性改善比純 Top-1 更明顯。
- `gemini-embedding-001` 的最佳 template 相對 `raw_label`，Top-1 提升 +0.0490、Top-5 提升 +0.0515、MRR 提升 +0.0515。
- `models/gemini-embedding-2-preview` 的最佳 template 相對 `raw_label`，Top-1 提升 +0.1667、Top-5 提升 +0.1348、MRR 提升 +0.1471。
- `models/gemini-embedding-2-preview` 對 template wording 仍然更敏感：它的 `raw_label` 基線只有 0.6765，低於 `gemini-embedding-001` 的 0.7794，但加上合適語境後提升幅度更大。
- 兩個模型都把 `novel_genre`、`novel_material`、`novel_content_direction` 放在前段，表示「作品類型／題材／內容走向」這種描述方式對新版標籤集仍是穩定有效的 template 家族。
