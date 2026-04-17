# Tag Template Mapping Experiment

- evaluation labels: 40
- mapping queries: 408
- candidate canonical tags: 96
- templates evaluated: 20
- embedding model: `models/gemini-embedding-2-preview`
- query task type: `RETRIEVAL_QUERY`
- candidate task type: `RETRIEVAL_DOCUMENT`
- fixed query template: `tag: {label}`

## Ranking

| Rank | Name | Template | Top-1 | Top-5 | MRR | Macro-F1 | Micro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | novel_setting | `這部小說的世界觀與設定帶有{label}色彩` | 0.8431 (+0.1667 vs raw) | 0.9632 | 0.8951 | 0.6738 | 0.8431 |
| 2 | novel_genre | `這部作品的類型偏向{label}` | 0.8382 (+0.1618 vs raw) | 0.9755 | 0.8959 | 0.5751 | 0.8382 |
| 3 | novel_color | `這部作品整體帶有濃厚的{label}色彩` | 0.8333 (+0.1569 vs raw) | 0.9730 | 0.8912 | 0.6126 | 0.8333 |
| 4 | novel_material | `這部小說採用了大量的{label}題材` | 0.8309 (+0.1544 vs raw) | 0.9632 | 0.8913 | 0.6173 | 0.8309 |
| 5 | novel_content_direction | `這部作品的內容走向偏向{label}` | 0.8284 (+0.1520 vs raw) | 0.9730 | 0.8904 | 0.6090 | 0.8284 |
| 6 | novel_property | `這部小說具備明顯的{label}屬性` | 0.8186 (+0.1422 vs raw) | 0.9657 | 0.8855 | 0.5889 | 0.8186 |
| 7 | novel_focus | `這部小說的主要看點之一是{label}` | 0.8186 (+0.1422 vs raw) | 0.9730 | 0.8842 | 0.6309 | 0.8186 |
| 8 | novel_contains | `這部小說包含明顯的{label}要素` | 0.8162 (+0.1397 vs raw) | 0.9657 | 0.8846 | 0.6315 | 0.8162 |
| 9 | novel_core_element | `這部作品的核心元素之一是{label}` | 0.8162 (+0.1397 vs raw) | 0.9681 | 0.8830 | 0.6177 | 0.8162 |
| 10 | novel_main_axis | `這部小說的主軸之一圍繞著{label}` | 0.8162 (+0.1397 vs raw) | 0.9510 | 0.8759 | 0.6200 | 0.8162 |
| 11 | novel_element | `這部小說具有鮮明的{label}元素` | 0.8137 (+0.1373 vs raw) | 0.9632 | 0.8803 | 0.6325 | 0.8137 |
| 12 | novel_theme | `這部小說的核心題材是{label}` | 0.8137 (+0.1373 vs raw) | 0.9559 | 0.8745 | 0.5634 | 0.8137 |
| 13 | novel_feature | `這部小說展現出明確的{label}特徵` | 0.8113 (+0.1348 vs raw) | 0.9583 | 0.8784 | 0.5982 | 0.8113 |
| 14 | novel_style | `這是一部帶有{label}元素的小說` | 0.8113 (+0.1348 vs raw) | 0.9510 | 0.8744 | 0.6198 | 0.8113 |
| 15 | novel_trait | `這部小說最鮮明的特色是{label}` | 0.8088 (+0.1324 vs raw) | 0.9461 | 0.8698 | 0.5882 | 0.8088 |
| 16 | novel_tag_cn | `小說標籤：{label}` | 0.8064 (+0.1299 vs raw) | 0.9510 | 0.8743 | 0.6170 | 0.8064 |
| 17 | novel_subject | `從題材上看，這部小說屬於{label}` | 0.8015 (+0.1250 vs raw) | 0.9436 | 0.8658 | 0.5749 | 0.8015 |
| 18 | light_novel_tag | `這本輕小說的標籤是{label}` | 0.7721 (+0.0956 vs raw) | 0.9314 | 0.8423 | 0.6146 | 0.7721 |
| 19 | novel_impression | `這部小說給人的主要印象之一是{label}` | 0.7623 (+0.0858 vs raw) | 0.9289 | 0.8351 | 0.5738 | 0.7623 |
| 20 | raw_label | `{label}` | 0.6765 (+0.0000 vs raw) | 0.8284 | 0.7480 | 0.5021 | 0.6765 |

## Diagnostics

### novel_setting - `這部小說的世界觀與設定帶有{label}色彩`

Best labels:
- NTR: top1=1.0000, top5=1.0000, mrr=1.0000
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.1000, top5=1.0000, mrr=0.5000, confusions=搞笑吐槽(8), 歡樂(1)
- 龍傲天: top1=0.3000, top5=0.8000, mrr=0.5200, confusions=webnovel(4), novelpia(2), 智斗(1)
- 後宮: top1=0.4545, top5=0.7273, mrr=0.5702, confusions=NTR(2), 智斗(1), webnovel(1)
- 腦洞: top1=0.5000, top5=1.0000, mrr=0.6833, confusions=奇幻(2), 獵奇(2), novelpia(1)
- 宅文化: top1=0.6000, top5=0.8000, mrr=0.6805, confusions=二次元(2), 蘿莉(1), 搞笑吐槽(1)

### novel_genre - `這部作品的類型偏向{label}`

Best labels:
- 偽娘: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000
- 妹妹: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.1000, top5=1.0000, mrr=0.4333, confusions=搞笑吐槽(8), 歡樂(1)
- 遊戲: top1=0.5000, top5=0.7000, mrr=0.6343, confusions=webnovel(1), 遊戲劇本(1), 穿越(1)
- 宅文化: top1=0.5000, top5=0.9000, mrr=0.6350, confusions=搞笑吐槽(2), JC(1), 二次元(1)
- 腦洞: top1=0.5000, top5=1.0000, mrr=0.7083, confusions=龍傲天(1), 奇幻(1), 鬥智(1)
- 後宮: top1=0.6364, top5=0.9091, mrr=0.7269, confusions=智斗(1), 經營(1), syosetu(1)

### novel_color - `這部作品整體帶有濃厚的{label}色彩`

Best labels:
- NTR: top1=1.0000, top5=1.0000, mrr=1.0000
- 大小姐: top1=1.0000, top5=1.0000, mrr=1.0000
- 女兒: top1=1.0000, top5=1.0000, mrr=1.0000
- 女性視角: top1=1.0000, top5=1.0000, mrr=1.0000
- 妹妹: top1=1.0000, top5=1.0000, mrr=1.0000

Worst labels:
- 歡樂向: top1=0.1000, top5=1.0000, mrr=0.5000, confusions=搞笑吐槽(8), 歡樂(1)
- 青春: top1=0.4000, top5=1.0000, mrr=0.6500, confusions=青梅竹馬(2), 少年漫畫(2), 青春日常(1)
- 後宮: top1=0.4545, top5=0.8182, mrr=0.5895, confusions=NTR(2), 智斗(1), webnovel(1)
- 宅文化: top1=0.5000, top5=0.8000, mrr=0.6134, confusions=二次元(2), 搞笑吐槽(2), 蘿莉(1)
- 遊戲: top1=0.5000, top5=0.8000, mrr=0.6452, confusions=webnovel(2), 遊戲劇本(2), 經營(1)
