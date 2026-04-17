# Tag Template Mapping Experiment

- evaluation labels: 40
- mapping queries: 408
- candidate canonical tags: 96
- templates evaluated: 20
- embedding model: `gemini-embedding-001`
- query task type: `RETRIEVAL_QUERY`
- candidate task type: `RETRIEVAL_DOCUMENT`
- fixed query template: `tag: {label}`

## Ranking

| Rank | Name | Template | Top-1 | Top-5 | MRR | Macro-F1 | Micro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | novel_material | `這部小說採用了大量的{label}題材` | 0.8284 (+0.0490 vs raw) | 0.9534 | 0.8874 | 0.6107 | 0.8284 |
| 2 | novel_genre | `這部作品的類型偏向{label}` | 0.8211 (+0.0417 vs raw) | 0.9657 | 0.8796 | 0.5925 | 0.8211 |
| 3 | novel_content_direction | `這部作品的內容走向偏向{label}` | 0.8211 (+0.0417 vs raw) | 0.9485 | 0.8786 | 0.5755 | 0.8211 |
| 4 | novel_style | `這是一部帶有{label}元素的小說` | 0.8113 (+0.0319 vs raw) | 0.9461 | 0.8737 | 0.5922 | 0.8113 |
| 5 | novel_core_element | `這部作品的核心元素之一是{label}` | 0.8113 (+0.0319 vs raw) | 0.9461 | 0.8729 | 0.5807 | 0.8113 |
| 6 | light_novel_tag | `這本輕小說的標籤是{label}` | 0.8088 (+0.0294 vs raw) | 0.9608 | 0.8740 | 0.5743 | 0.8088 |
| 7 | novel_theme | `這部小說的核心題材是{label}` | 0.8039 (+0.0245 vs raw) | 0.9436 | 0.8623 | 0.5408 | 0.8039 |
| 8 | novel_contains | `這部小說包含明顯的{label}要素` | 0.7966 (+0.0172 vs raw) | 0.9608 | 0.8707 | 0.5547 | 0.7966 |
| 9 | novel_tag_cn | `小說標籤：{label}` | 0.7966 (+0.0172 vs raw) | 0.9485 | 0.8641 | 0.5532 | 0.7966 |
| 10 | novel_trait | `這部小說最鮮明的特色是{label}` | 0.7892 (+0.0098 vs raw) | 0.9412 | 0.8581 | 0.5595 | 0.7892 |
| 11 | novel_focus | `這部小說的主要看點之一是{label}` | 0.7892 (+0.0098 vs raw) | 0.9363 | 0.8572 | 0.5698 | 0.7892 |
| 12 | novel_subject | `從題材上看，這部小說屬於{label}` | 0.7892 (+0.0098 vs raw) | 0.9363 | 0.8568 | 0.5386 | 0.7892 |
| 13 | novel_color | `這部作品整體帶有濃厚的{label}色彩` | 0.7868 (+0.0074 vs raw) | 0.9534 | 0.8618 | 0.5633 | 0.7868 |
| 14 | novel_main_axis | `這部小說的主軸之一圍繞著{label}` | 0.7819 (+0.0025 vs raw) | 0.9265 | 0.8481 | 0.5062 | 0.7819 |
| 15 | raw_label | `{label}` | 0.7794 (+0.0000 vs raw) | 0.9020 | 0.8360 | 0.5375 | 0.7794 |
| 16 | novel_element | `這部小說具有鮮明的{label}元素` | 0.7770 (-0.0025 vs raw) | 0.9461 | 0.8515 | 0.5342 | 0.7770 |
| 17 | novel_impression | `這部小說給人的主要印象之一是{label}` | 0.7770 (-0.0025 vs raw) | 0.9338 | 0.8496 | 0.5611 | 0.7770 |
| 18 | novel_feature | `這部小說展現出明確的{label}特徵` | 0.7745 (-0.0049 vs raw) | 0.9461 | 0.8505 | 0.5241 | 0.7745 |
| 19 | novel_setting | `這部小說的世界觀與設定帶有{label}色彩` | 0.7745 (-0.0049 vs raw) | 0.9412 | 0.8439 | 0.5495 | 0.7745 |
| 20 | novel_property | `這部小說具備明顯的{label}屬性` | 0.7696 (-0.0098 vs raw) | 0.9461 | 0.8498 | 0.5279 | 0.7696 |

## Diagnostics

### novel_material - `這部小說採用了大量的{label}題材`

Best labels:
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000
- 妹妹: top1=1.0000, top5=1.0000, mrr=1.0000
- 機戰: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 宅文化: top1=0.5000, top5=0.5000, mrr=0.5599, confusions=輕改漫畫(2), 二次元(1), 蘿莉(1)
- 龍傲天: top1=0.5000, top5=0.7000, mrr=0.6083, confusions=熱血(1), NTR(1), 大逃殺(1)
- 遊戲: top1=0.5000, top5=0.9000, mrr=0.6843, confusions=遊戲劇本(2), 機戰(1), 穿越(1)
- 後宮: top1=0.5455, top5=0.7273, mrr=0.6476, confusions=經營(1), 戰鬥(1), 戀愛(1)
- 戰爭: top1=0.5455, top5=1.0000, mrr=0.7727, confusions=戰鬥(5)

### novel_genre - `這部作品的類型偏向{label}`

Best labels:
- JK: top1=1.0000, top5=1.0000, mrr=1.0000
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.0000, top5=0.8000, mrr=0.3205, confusions=搞笑吐槽(6), 歡樂(3), 腦洞(1)
- 遊戲: top1=0.4000, top5=1.0000, mrr=0.6150, confusions=遊戲劇本(3), novelengine(1), 旅行(1)
- 驚悚: top1=0.4000, top5=1.0000, mrr=0.6667, confusions=獵奇(4), 黑暗(2)
- 後宮: top1=0.4545, top5=0.7273, mrr=0.5918, confusions=鬥智(1), other(1), 戰鬥(1)
- 宅文化: top1=0.5000, top5=0.8000, mrr=0.5845, confusions=輕改漫畫(1), 二次元(1), 蘿莉(1)

### novel_content_direction - `這部作品的內容走向偏向{label}`

Best labels:
- JC: top1=1.0000, top5=1.0000, mrr=1.0000
- JK: top1=1.0000, top5=1.0000, mrr=1.0000
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.0000, top5=0.8000, mrr=0.3681, confusions=搞笑吐槽(6), 歡樂(3), 腦洞(1)
- 龍傲天: top1=0.4000, top5=0.7000, mrr=0.5359, confusions=大逃殺(2), 格鬥(1), 大小姐(1)
- 後宮: top1=0.4545, top5=0.7273, mrr=0.5513, confusions=鬥智(1), other(1), 戰鬥(1)
- 宅文化: top1=0.5000, top5=0.6000, mrr=0.5684, confusions=輕改漫畫(1), 二次元(1), 蘿莉(1)
- 遊戲: top1=0.5000, top5=0.9000, mrr=0.6617, confusions=遊戲劇本(2), novelengine(1), 旅行(1)
