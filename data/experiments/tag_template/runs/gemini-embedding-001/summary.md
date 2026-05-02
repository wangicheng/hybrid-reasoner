# Tag Template Mapping Experiment

- evaluation labels: 40
- mapping queries: 408
- candidate canonical tags: 96
- templates evaluated: 20
- embedding model: `gemini-embedding-001`
- query task type: `RETRIEVAL_QUERY`
- candidate task type: `RETRIEVAL_DOCUMENT`
- fixed query template: `tag: {label}`
- symmetric templates: `True`

## Ranking

| Rank | Name | Template | Top-1 | Top-5 | MRR | Macro-F1 | Micro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | novel_genre | `這部作品的類型偏向{label}` | 0.8309 (+0.0539 vs raw) | 0.9559 | 0.8880 | 0.6287 | 0.8309 |
| 2 | light_novel_tag | `這本輕小說的標籤是{label}` | 0.8284 (+0.0515 vs raw) | 0.9706 | 0.8891 | 0.5676 | 0.8284 |
| 3 | novel_property | `這部小說具備明顯的{label}屬性` | 0.8162 (+0.0392 vs raw) | 0.9436 | 0.8775 | 0.5684 | 0.8162 |
| 4 | novel_tag_cn | `小說標籤：{label}` | 0.8162 (+0.0392 vs raw) | 0.9559 | 0.8759 | 0.5838 | 0.8162 |
| 5 | novel_style | `這是一部帶有{label}元素的小說` | 0.8137 (+0.0368 vs raw) | 0.9461 | 0.8760 | 0.5865 | 0.8137 |
| 6 | novel_content_direction | `這部作品的內容走向偏向{label}` | 0.8088 (+0.0319 vs raw) | 0.9583 | 0.8774 | 0.5894 | 0.8088 |
| 7 | novel_color | `這部作品整體帶有濃厚的{label}色彩` | 0.8088 (+0.0319 vs raw) | 0.9583 | 0.8763 | 0.5888 | 0.8088 |
| 8 | novel_material | `這部小說採用了大量的{label}題材` | 0.8088 (+0.0319 vs raw) | 0.9559 | 0.8756 | 0.5738 | 0.8088 |
| 9 | novel_theme | `這部小說的核心題材是{label}` | 0.8064 (+0.0294 vs raw) | 0.9436 | 0.8702 | 0.5617 | 0.8064 |
| 10 | novel_element | `這部小說具有鮮明的{label}元素` | 0.7966 (+0.0196 vs raw) | 0.9387 | 0.8638 | 0.5662 | 0.7966 |
| 11 | novel_core_element | `這部作品的核心元素之一是{label}` | 0.7966 (+0.0196 vs raw) | 0.9485 | 0.8636 | 0.5584 | 0.7966 |
| 12 | novel_subject | `從題材上看，這部小說屬於{label}` | 0.7941 (+0.0172 vs raw) | 0.9412 | 0.8603 | 0.5814 | 0.7941 |
| 13 | novel_focus | `這部小說的主要看點之一是{label}` | 0.7917 (+0.0147 vs raw) | 0.9461 | 0.8610 | 0.5865 | 0.7917 |
| 14 | novel_contains | `這部小說包含明顯的{label}要素` | 0.7892 (+0.0123 vs raw) | 0.9583 | 0.8655 | 0.5491 | 0.7892 |
| 15 | novel_trait | `這部小說最鮮明的特色是{label}` | 0.7868 (+0.0098 vs raw) | 0.9510 | 0.8573 | 0.5350 | 0.7868 |
| 16 | novel_impression | `這部小說給人的主要印象之一是{label}` | 0.7819 (+0.0049 vs raw) | 0.9289 | 0.8476 | 0.5393 | 0.7819 |
| 17 | raw_label | `{label}` | 0.7770 (+0.0000 vs raw) | 0.9142 | 0.8396 | 0.5425 | 0.7770 |
| 18 | novel_setting | `這部小說的世界觀與設定帶有{label}色彩` | 0.7721 (-0.0049 vs raw) | 0.9436 | 0.8468 | 0.5420 | 0.7721 |
| 19 | novel_feature | `這部小說展現出明確的{label}特徵` | 0.7721 (-0.0049 vs raw) | 0.9387 | 0.8457 | 0.5459 | 0.7721 |
| 20 | novel_main_axis | `這部小說的主軸之一圍繞著{label}` | 0.7696 (-0.0074 vs raw) | 0.9338 | 0.8446 | 0.5081 | 0.7696 |

## Diagnostics

### novel_genre - `這部作品的類型偏向{label}`

Best labels:
- 人外: top1=1.0000, top5=1.0000, mrr=1.0000
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.0000, top5=1.0000, mrr=0.4400, confusions=歡樂(8), 腦洞(1), 搞笑吐槽(1)
- 宅文化: top1=0.5000, top5=0.7000, mrr=0.5637, confusions=二次元(2), 妹妹(1), 搞笑吐槽(1)
- 龍傲天: top1=0.5000, top5=0.8000, mrr=0.6306, confusions=戰鬥(2), 大小姐(1), 奇幻(1)
- NTR: top1=0.5455, top5=0.7273, mrr=0.6073, confusions=戀愛(3), 惡役(1), 復仇(1)
- 後宮: top1=0.5455, top5=0.7273, mrr=0.6182, confusions=職場(1), 黑暗(1), 戀愛(1)

### light_novel_tag - `這本輕小說的標籤是{label}`

Best labels:
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000
- 妹妹: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.0000, top5=1.0000, mrr=0.3917, confusions=歡樂(6), 搞笑吐槽(3), 腦洞(1)
- 後宮: top1=0.4545, top5=0.7273, mrr=0.5859, confusions=群像(2), 戰爭(1), 經營(1)
- 龍傲天: top1=0.5000, top5=1.0000, mrr=0.6317, confusions=webnovel(1), 穿越(1), 戰鬥(1)
- 宅文化: top1=0.6000, top5=1.0000, mrr=0.7033, confusions=二次元(1), 蘿莉(1), 搞笑吐槽(1)
- 遊戲: top1=0.6000, top5=1.0000, mrr=0.7667, confusions=遊戲劇本(2), novelengine(1), syosetu(1)

### novel_property - `這部小說具備明顯的{label}屬性`

Best labels:
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000
- 妹妹: top1=1.0000, top5=1.0000, mrr=1.0000
- 性轉: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.0000, top5=1.0000, mrr=0.4333, confusions=歡樂(7), 搞笑吐槽(2), 腦洞(1)
- 後宮: top1=0.4545, top5=0.6364, mrr=0.5665, confusions=競技(1), 經營(1), 黑暗(1)
- 宅文化: top1=0.5000, top5=0.9000, mrr=0.6393, confusions=二次元(2), 蘿莉(1), 同人(1)
- 龍傲天: top1=0.5000, top5=0.8000, mrr=0.6414, confusions=大小姐(1), novelengine(1), 戰鬥(1)
- 治癒: top1=0.5000, top5=1.0000, mrr=0.7500, confusions=溫馨(5)
