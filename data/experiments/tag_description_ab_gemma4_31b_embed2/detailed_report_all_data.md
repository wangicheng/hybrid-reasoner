# Tag Extraction Detailed Report (All Data)

- generated_at: 2026-04-20T18:10:24
- extraction_model: models/gemma-4-31b-it
- embedding_model: models/gemini-embedding-2-preview
- total_queries: 24

## Aggregate Metrics (Comparison)

| Metric | No Description | With Description | Delta |
| --- | ---: | ---: | ---: |
| parse_success_rate | 1.000000 | 1.000000 | +0.000000 |
| required_exact_cover_rate | 0.928571 | 1.000000 | +0.071429 |
| blocked_clean_rate | 1.000000 | 1.000000 | +0.000000 |
| required_micro_f1 | 0.452381 | 0.449438 | -0.002943 |
| required_macro_f1 | 0.061508 | 0.072619 | +0.011111 |
| required_exact_match_rate | 0.071429 | 0.000000 | -0.071429 |
| raw_outside_taxonomy_rate | 0.018519 | 0.000000 | -0.018519 |
| avg_pred_tag_count | 4.500000 | 4.875000 | +0.375000 |

## Aggregate Metrics (Raw Reports)

| Key | No Description | With Description |
| --- | --- | --- |
| avg_pred_tag_count | 4.500000 | 4.875000 |
| blocked_clean_hits | 10 | 10 |
| blocked_clean_rate | 1.000000 | 1.000000 |
| blocked_query_count | 10 | 10 |
| descriptions_path |  | C:\Users\USER\Desktop\code\Hybrid Reasoner\hybrid-reasoner\data\tag_descriptions.json |
| embedding_model_id | models/gemini-embedding-2-preview | models/gemini-embedding-2-preview |
| extraction_model_id | models/gemma-4-31b-it | models/gemma-4-31b-it |
| parse_success | 24 | 24 |
| parse_success_rate | 1.000000 | 1.000000 |
| prediction_key | mapped_pred_tags | mapped_pred_tags |
| raw_outside_taxonomy_rate | 0.018519 | 0.000000 |
| raw_outside_taxonomy_tag_count | 2 | 0 |
| raw_pred_tag_count | 108 | 117 |
| required_exact_cover_hits | 13 | 14 |
| required_exact_cover_rate | 0.928571 | 1.000000 |
| required_exact_match_rate | 0.071429 | 0.000000 |
| required_macro_f1 | 0.061508 | 0.072619 |
| required_macro_precision | 0.048065 | 0.059350 |
| required_macro_recall | 0.093750 | 0.104167 |
| required_micro_f1 | 0.452381 | 0.449438 |
| required_micro_precision | 0.296875 | 0.289855 |
| required_micro_recall | 0.950000 | 1.000000 |
| required_query_count | 14 | 14 |
| total_queries | 24 | 24 |
| use_tag_descriptions | False | True |

## Per-Query Detailed Results

### q1

- query: 現在對於小說的要求越來越高了， / 而且慢慢的發現現在除了狗糧基本上都看不進去， / 連以很喜歡的異世界主題都看不進去了， / 好想念從前那個什麼都可以看得很快樂的自己 /  / 說回主題，希望可以找到沒看過的狗糧系小說， / 可以接受小刀，但一定要無黃毛無牛頭人， / 但背景自帶沈重的例如「戀愛光譜」「刮鬍」這一類的也不喜歡， / 黨爭就...看情況吧但其實我也一般般
- required_tags: ["戀愛"]
- blocked_tags: ["NTR"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["戀愛", "溫馨", "歡樂", "治癒"] | ["戀愛", "溫馨", "異世界", "歡樂", "治癒"] |
| raw_extracted_tags | ["戀愛", "溫馨", "歡樂", "治癒"] | ["戀愛", "溫馨", "異世界", "歡樂", "治癒"] |
| generated_keywords | ["純愛", "撒糖", "甜寵", "心動", "幸福感", "日常戀愛", "輕快"] | ["純愛", "甜寵", "單女主", "溫馨日常", "幸福感", "心跳", "甜蜜互動"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["戀愛", "溫馨", "歡樂", "治癒"] | ["戀愛", "溫馨", "異世界", "歡樂", "治癒"] |
| outside_tag_mapping | [] | [] |

### q2

- query: 不好意思,最近異世界需求大了,很想看異世界的故事不一定要轉生類但劍與魔法是必需的.. 已有蜘蛛子了,目前在找一些已經有結局的異世界小說 原本想看哥布林殺手的但是還沒完結啊.....超不想看到一半被中斷了 至少至少...像刀劍那種完了SAO篇再換ALO的方式我還可以接受 請問還有什麼戰鬥類已完結的異世界小說可以推介呢?
- required_tags: ["異世界", "戰鬥"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["異世界", "魔法", "冒險", "戰鬥", "奇幻"] | ["異世界", "魔法", "戰鬥", "冒險", "奇幻"] |
| raw_extracted_tags | ["異世界", "魔法", "冒險", "戰鬥", "奇幻"] | ["異世界", "魔法", "戰鬥", "冒險", "奇幻"] |
| generated_keywords | ["勇者", "魔王", "魔法騎士", "異世界冒險", "劍術", "奇幻世界", "史詩戰鬥", "完結作品"] | ["劍士", "地下城", "冒險者", "魔王", "西方奇幻", "技能體系", "討伐"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["異世界", "魔法", "冒險", "戰鬥", "奇幻"] | ["異世界", "魔法", "戰鬥", "冒險", "奇幻"] |
| outside_tag_mapping | [] | [] |

### q3

- query: 可以幫忙推薦幾部跟這部風格類似的輕小說 / 這部吸引我的地方有幾個: / 1.男女主角純愛堅定不移 / 2.各種甜蜜的打情罵俏劇情 / 3.異世界轉生戰鬥、魔法 / 4.中間雖然各種虐，但最後還是排除萬難再一起
- required_tags: ["異世界", "轉生", "魔法"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["異世界", "轉生", "魔法", "戰鬥", "戀愛", "溫馨"] | ["轉生", "異世界", "魔法", "戰鬥", "戀愛", "溫馨"] |
| raw_extracted_tags | ["異世界", "轉生", "魔法", "戰鬥", "戀愛", "溫馨"] | ["轉生", "異世界", "魔法", "戰鬥", "戀愛", "溫馨"] |
| generated_keywords | ["純愛", "打情罵俏", "宿命", "救贖", "忠誠", "甜蜜互動", "排除萬難", "圓滿結局"] | ["純愛", "命中注定", "虐心後圓滿", "甜蜜互動", "堅定不移", "強強聯手", "生死相依"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["異世界", "轉生", "魔法", "戰鬥", "戀愛", "溫馨"] | ["轉生", "異世界", "魔法", "戰鬥", "戀愛", "溫馨"] |
| outside_tag_mapping | [] | [] |

### q4

- query: 由於自己的胃口比較獨特的原因，喜歡男主被拋置於各種不利的情節或者感情中，也是因為這樣，喜歡看那些讓人看完后心緒難以平靜的小說，就我看過的來說，無職，re0，妹妹人生，異世界迷宮最深等，在某種程度上來說都有慘烈的一部分（感情上或者情節上都是），也沒有一定要轉生異世界，現實日常要是能描繪出這些來也很好了。 / 尤其那種男主能不斷被放置在痛苦或者恐懼之中，一邊流露出作者思考的文字的小說，再好不過。
- required_tags: []
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["黑暗", "異世界", "穿越"] | ["黑暗", "轉生", "異世界", "青春日常", "驚悚"] |
| raw_extracted_tags | ["黑暗", "異世界", "穿越"] | ["黑暗", "轉生", "異世界", "青春日常", "驚悚"] |
| generated_keywords | ["虐心", "精神折磨", "絕望感", "心理掙扎", "悲劇色彩", "沉重氛圍", "生存壓力", "情感崩潰", "人性反思"] | ["精神折磨", "絕望感", "人性探討", "虐心", "心理創傷", "悲劇", "沉重氛圍", "自我救贖"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["黑暗", "異世界", "穿越"] | ["黑暗", "轉生", "異世界", "青春日常", "驚悚"] |
| outside_tag_mapping | [] | [] |

### q5

- query: 如題，想找像是overlord、無職這種，有龐大世界，多個國家部落，文化特色分明，歷史背景完整，沉浸感和史詩感強的小說，請問各位有沒有推薦的？希望是劍與魔法的世界，有多不同種族、魔物的那種（其實我不太確定這類型叫什麼，異世界感覺不太對，但叫奇幻又好像太廣泛了？ /  / 有很多web小說我都有看過，像是轉生成魔劍、轉生龍蛋、哥布林，他們雖然世界都足夠龐大，但是都有個web小說的通病，就是讀起來會有割裂感，無法讓人帶入那個世界中 / 而轉生蜘蛛雖然沒有那種割裂感，但是世界觀卻不太夠了，就只簡單分成教皇、妖精、魔王而已，希望能看到更多人類國家間的關係，還有更多魔物亞人種族的故事設定
- required_tags: ["奇幻"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["奇幻", "異世界", "冒險", "轉生", "原生幻想"] | ["異世界", "奇幻", "冒險", "魔法", "原生幻想", "人外"] |
| raw_extracted_tags | ["奇幻", "異世界", "冒險", "轉生", "原生幻想"] | ["異世界", "奇幻", "冒險", "魔法", "原生幻想", "人外"] |
| generated_keywords | ["地緣政治", "種族衝突", "世界觀構築", "史詩奇幻", "文化差異", "亞人設定", "宏大敘事", "文明演進"] | ["世界觀設定", "地緣政治", "種族衝突", "史詩奇幻", "文明發展", "中世紀奇幻", "詳細設定", "宏大敘事"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["奇幻", "異世界", "冒險", "轉生", "原生幻想"] | ["異世界", "奇幻", "冒險", "魔法", "原生幻想", "人外"] |
| outside_tag_mapping | [] | [] |

### q6

- query: 很喜歡看戰爭或戰記類型的小說，不要有魔法一發收拾或莫名其妙的神器，也別機械降神，主角可以強但不要把對手變白癡，暑假到了求書來啃，斷尾或沒完結也沒關係。
- required_tags: ["戰爭"]
- blocked_tags: ["龍傲天"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["戰爭", "戰鬥", "智斗"] | ["戰爭", "智斗", "鬥智", "戰鬥", "原生幻想"] |
| raw_extracted_tags | ["戰爭", "戰鬥", "智斗"] | ["戰爭", "智斗", "鬥智", "戰鬥", "原生幻想"] |
| generated_keywords | ["軍事策略", "兵法", "攻防戰", "地緣政治", "後勤", "軍事鬥爭", "寫實戰爭", "陣地戰"] | ["軍事策略", "兵法", "權謀", "大軍團戰", "地緣政治", "寫實戰爭", "戰術分析"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["戰爭", "戰鬥", "智斗"] | ["戰爭", "智斗", "鬥智", "戰鬥", "原生幻想"] |
| outside_tag_mapping | [] | [] |

### q7

- query: 想請大家推薦一些書單希望符合以下條件 / 1.已經完結 / 2.以感情為主 / 希望是讀的時候可以隨著角色情緒有所起伏。並且不要轉生開無雙。
- required_tags: []
- blocked_tags: ["龍傲天", "轉生"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["戀愛", "青春", "治癒", "溫馨"] | ["戀愛", "青春", "治癒"] |
| raw_extracted_tags | ["戀愛", "青春", "治癒", "溫馨"] | ["戀愛", "青春", "治癒"] |
| generated_keywords | ["情感糾葛", "角色成長", "心路歷程", "深刻", "悲歡離合", "純愛", "情感共鳴"] | ["情感糾葛", "心理描寫", "成長物語", "深刻", "心碎", "純愛", "情感共鳴", "角色發展"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["戀愛", "青春", "治癒", "溫馨"] | ["戀愛", "青春", "治癒"] |
| outside_tag_mapping | [] | [] |

### q8

- query: 希望作品是 / 1.背景為現代 / 有槍砲炸藥之類的比較有代入感，對武俠/奇幻沒興趣，現代但有些微科幻/超能力要素沒關係 / 2.主角不是從0開始練等 / 漏漏長的成長過程會讓我想睡，大概是從看完鬥破蒼穹那公式化的遇強敵→隊友幫扛→遇到機運→變強反殺→前往下個城鎮→遇強敵，之後再也不想看這種了 / 3.主角非中二/死屁孩 / 不要太嚴重就好，隱殺主角偶爾也會屁孩一下但是是那種大人逗小鬼玩的感覺就不會反感
- required_tags: []
- blocked_tags: ["奇幻"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["科幻", "異能", "戰鬥"] | ["科幻", "異能", "戰鬥", "龍傲天"] |
| raw_extracted_tags | ["科幻", "異能", "戰鬥"] | ["科幻", "異能", "戰鬥", "龍傲天"] |
| generated_keywords | ["現代都市", "特種部隊", "雇傭兵", "強者無敵", "都市異能", "戰術作戰", "快節奏", "成熟心智"] | ["現代都市", "軍事戰術", "能力者", "強勢主角", "快節奏", "特工", "成熟心智"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["科幻", "異能", "戰鬥"] | ["科幻", "異能", "戰鬥", "龍傲天"] |
| outside_tag_mapping | [] | [] |

### q9

- query: 想請各位大大推薦一下這類的輕小說或小說，大陸網文也可以（我就是看了某部大陸網路輕小說才想找一些這類型的故事來看 /  / 世界觀再細一點的話 / 就是那種奇幻元素並不普及於所有人，只有少部份人擁有特殊能力，或是一些非人種族不會輕易在人前現身，隱藏在人類社會中那種都市奇幻類型
- required_tags: ["奇幻"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["奇幻", "異能"] | ["奇幻", "異能", "人外"] |
| raw_extracted_tags | ["奇幻", "異能"] | ["奇幻", "異能", "人外"] |
| generated_keywords | ["都市傳說", "現代異能", "秘密組織", "隱藏種族", "異類共存", "超能力者", "現代社會奇幻", " masquerade"] | ["都市傳說", "現代魔術", "秘密結社", "隱藏身分", "超能力", "種族共存", "現代幻想"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["奇幻", "異能"] | ["奇幻", "異能", "人外"] |
| outside_tag_mapping | [] | [] |

### q10

- query: 像是果青友少這種的 / 年份的話最近三到五年為主 / 背景是校園戀愛喜劇為主 / 有複數女角讓男主多選一 / 啊不要開場就有女朋友鎖死的 / 那種我覺得其他女角只是浪費時間而已
- required_tags: ["後宮", "校園"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["校園", "戀愛", "後宮", "青春", "搞笑吐槽", "日本輕小說"] | ["校園", "戀愛", "搞笑吐槽", "後宮", "青春", "歡樂"] |
| raw_extracted_tags | ["校園", "戀愛", "後宮", "青春", "搞笑吐槽", "日本輕小說"] | ["校園", "戀愛", "搞笑吐槽", "後宮", "青春", "歡樂"] |
| generated_keywords | ["青春喜劇", "修羅場", "多角關係", "校園生活", "戀愛競爭", "多位女主角", "後宮爭霸", "純情"] | ["修羅場", "多角關係", "純愛", "校園生活", "甜寵", "多女主"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["校園", "戀愛", "後宮", "青春", "搞笑吐槽", "日本輕小說"] | ["校園", "戀愛", "搞笑吐槽", "後宮", "青春", "歡樂"] |
| outside_tag_mapping | [] | [] |

### q11

- query: 最近人比較負面， / 想看一些有共鳴的小說。 / 有沒有那種「我一個人爛就好了別搞其他人」的主角的輕小說？以前看大老師自暴自棄滿戳中我的。 / 宅宅現實找不到人取暖，希望能找輕小說取。
- required_tags: []
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["校園", "青春", "治癒", "日本輕小說"] | ["青春", "校園", "治癒"] |
| raw_extracted_tags | ["校園", "青春", "治癒", "日本輕小說"] | ["青春", "校園", "治癒"] |
| generated_keywords | ["邊緣人", "自我犧牲", "孤立", "心理描寫", "救贖", "自卑", "社會恐懼", "扭曲的價值觀"] | ["社交恐懼", "自我厭惡", "孤獨感", "救贖", "邊緣人", "內心獨白", "情感共鳴"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["校園", "青春", "治癒", "日本輕小說"] | ["青春", "校園", "治癒"] |
| outside_tag_mapping | [] | [] |

### q12

- query: 想找一些日常系小說類型，希望劇情節奏安排和對話有趣些。 / 其它元素能接受百合、戀愛、生活、學校類型，有談話間交流劇情。
- required_tags: []
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["青春日常", "校園", "戀愛", "百合", "溫馨", "歡樂"] | ["青春日常", "校園", "百合", "戀愛", "溫馨", "歡樂"] |
| raw_extracted_tags | ["青春日常", "校園", "戀愛", "百合", "溫馨", "歡樂"] | ["青春日常", "校園", "百合", "戀愛", "溫馨", "歡樂"] |
| generated_keywords | ["聊天", "輕鬆", "生活化", "互動", "吐槽", "青春", "氛圍感", "日常趣事"] | ["日常", "吐槽", "輕快", "校園生活", "少女情懷", "生活瑣事", "聊天", "溫馨"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["青春日常", "校園", "戀愛", "百合", "溫馨", "歡樂"] | ["青春日常", "校園", "百合", "戀愛", "溫馨", "歡樂"] |
| outside_tag_mapping | [] | [] |

### q13

- query: 我很喜歡搞笑類，誤會系剛好能戳到我的笑點， / 看了會有種「這群人的腦迴路到底怎麼長的」的感覺，（像電玩咖這種的） / 當然不需要全程搞笑，後面進入主線開始嚴肅起來我還是能接受的（素晴、腦礙這種的）
- required_tags: ["歡樂向"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | True |
| blocked_clean | True | True |
| pred_tags | ["搞笑吐槽", "腦洞", "歡樂", "異世界", "冒險"] | ["搞笑吐槽", "歡樂", "歡樂向", "腦洞"] |
| raw_extracted_tags | ["搞笑吐槽", "腦洞", "歡樂", "異世界", "冒險"] | ["搞笑吐槽", "歡樂", "歡樂向", "腦洞"] |
| generated_keywords | ["誤會系", "沙雕", "腦補", "錯位", "反轉", "荒謬邏輯", "反差萌", "認知偏差"] | ["誤會系", "沙雕", "反差", "天然呆", "反轉", "邏輯崩壞", "爆笑", "腦迴路"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["搞笑吐槽", "腦洞", "歡樂", "異世界", "冒險"] | ["搞笑吐槽", "歡樂", "歡樂向", "腦洞"] |
| outside_tag_mapping | [] | [] |

### q14

- query: 想求各位大大們，有沒有推薦以百合、女生友情、男女愛情這類的日本輕小說呢? /  / 不太想要有奇幻、科幻、后宮、異世界這元素，有點太過氾濫不太想看 / 只要日常、青春的就可以了，有發糖的和胃痛我都可以~
- required_tags: []
- blocked_tags: ["後宮", "奇幻", "異世界", "科幻"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["百合", "日本輕小說", "青春", "戀愛", "青春日常", "溫馨"] | ["百合", "日本輕小說", "青春", "青春日常", "戀愛", "溫馨"] |
| raw_extracted_tags | ["百合", "日本輕小說", "青春", "戀愛", "青春日常", "溫馨"] | ["百合", "日本輕小說", "青春", "青春日常", "戀愛", "溫馨"] |
| generated_keywords | ["純愛", "校園生活", "少女心情", "友情", "情感糾葛", "生活瑣事", "青春悸動", "胃痛"] | ["純愛", "校園生活", "少女情誼", "心動", "青春疼痛", "情感糾葛", "日常瑣事", "純愛物語"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["百合", "日本輕小說", "青春", "戀愛", "青春日常", "溫馨"] | ["百合", "日本輕小說", "青春", "青春日常", "戀愛", "溫馨"] |
| outside_tag_mapping | [] | [] |

### q15

- query: 喜歡的類型 是異世界 轉生 重生 穿越 校園異能 幻想世界 劍與魔法 主角最強 多後宮 戰鬥要夠中二 都行 / 夠套路 / 要男主 一定要有戰鬥 後宮 沒後宮也沒關係但要有女主 如果主角拿的是劍我會更開心 / 不接受有沒男主的 例如蜘蛛又怎樣 能力值平均 怕痛的我 / 不接受虐主的 除了re 0 是因為動畫入坑的 除非前面虐後面很爽的這樣可以例如平職 / 需要裝逼讓我看的會高潮的 所以無職轉生我看不下去因為太真實太哭逼了
- required_tags: ["戰鬥"]
- blocked_tags: ["女性視角", "百合"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["異世界", "轉生", "穿越", "校園", "異能", "魔法", "奇幻", "後宮", "戰鬥", "龍傲天"] | ["異世界", "轉生", "穿越", "校園", "異能", "魔法", "後宮", "龍傲天", "戰鬥", "奇幻"] |
| raw_extracted_tags | ["異世界", "轉生", "穿越", "校園", "異能", "魔法", "奇幻", "後宮", "戰鬥", "龍傲天"] | ["異世界", "轉生", "穿越", "校園", "異能", "魔法", "後宮", "龍傲天", "戰鬥", "奇幻"] |
| generated_keywords | ["無敵流", "打臉", "爽文", "聖劍", "魔法學校", "最強", "後宮", "中二病", "異世界冒險"] | ["最強", "無敵", "劍客", "打臉", "中二", "爽文", "後宮", "幻想世界", "反殺", "扮豬吃虎"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["異世界", "轉生", "穿越", "校園", "異能", "魔法", "奇幻", "後宮", "戰鬥", "龍傲天"] | ["異世界", "轉生", "穿越", "校園", "異能", "魔法", "後宮", "龍傲天", "戰鬥", "奇幻"] |
| outside_tag_mapping | [] | [] |

### q16

- query: 如題想看校園類型小說，不知道有無推薦 / 純現實的不要有奇幻元素，集數不多的佳
- required_tags: ["校園"]
- blocked_tags: ["異世界", "奇幻", "轉生", "科幻"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["校園", "青春", "青春日常"] | ["校園", "青春", "青春日常"] |
| raw_extracted_tags | ["校園", "青春", "青春日常"] | ["校園", "青春", "青春日常"] |
| generated_keywords | ["高中生活", "寫實", "初戀", "學生時代", "純愛", "成長", "日常瑣事"] | ["學生", "教室", "社團", "成長", "純愛", "校園生活", "現實主義"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["校園", "青春", "青春日常"] | ["校園", "青春", "青春日常"] |
| outside_tag_mapping | [] | [] |

### q17

- query: 求推薦世界觀建構完整 / 主角很清楚當下該做甚麼 / 是不太希望主角輕鬆拿到一堆外掛這種的 / 然後也蠻喜歡主角大意、失誤或忘記重要信息導致危機感這種狀況發生 / 非戰鬥類主角也可以，感覺這類主角穿越的比較多? / 但是如果是一個高中生通十八般武藝心理素質又特好的就不要
- required_tags: []
- blocked_tags: ["龍傲天"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["穿越", "異世界", "冒險", "奇幻"] | ["穿越", "智斗", "原生幻想", "冒險", "腦洞"] |
| raw_extracted_tags | ["穿越", "智鬥", "異世界", "冒險", "奇幻"] | ["穿越", "智斗", "原生幻想", "冒險", "腦洞"] |
| generated_keywords | ["嚴謹世界觀", "理性思考", "人性弱點", "策略博弈", "心理壓力", "邏輯推理", "危機處理"] | ["緻密世界觀", "邏輯嚴謹", "合理成長", "人性弱點", "危機處理", "非龍傲天", "策略佈局"] |
| outside_tags_raw | ["智鬥"] | [] |
| mapped_pred_tags | ["穿越", "異世界", "冒險", "奇幻", "鬥智"] | ["穿越", "智斗", "原生幻想", "冒險", "腦洞"] |
| outside_tag_mapping | [{"source": "智鬥", "mapped_label": "鬥智", "similarity": 0.9190746115842447, "accepted": true}] | [] |

### q18

- query: 我最近剛看了龍王的工作的小說 / 雖然不懂將棋，但覺得對局的刻劃很棒 / 能看到將棋解說 / 也會側面描寫一些職業棋士、女流棋士的差別和情況 /  / 最喜歡的部分是銀子挑戰獎勵會那邊 / 書裡描寫的女性棋士會被小看的部分 / 那種低人一等、逆境求生的感覺做得很真實(?不會形容，總之很有代入感 / 銀子加油，不要輸給將棋星人! /  / 希望各位能分享一些好看的將棋小說!!
- required_tags: []
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["遊戲", "熱血", "日本輕小說", "青春"] | ["遊戲", "競技", "智斗", "日本輕小說", "青春"] |
| raw_extracted_tags | ["遊戲", "熱血", "智鬥", "日本輕小說", "青春"] | ["遊戲", "競技", "智斗", "日本輕小說", "青春"] |
| generated_keywords | ["將棋", "棋道", "對局", "女流棋士", "職業段位", "殘局", "棋譜", "逆境", "成長", "競技"] | ["將棋", "棋譜", "對局分析", "職業棋士", "女流棋士", "逆境", "成長", "心理戰", "日本棋道"] |
| outside_tags_raw | ["智鬥"] | [] |
| mapped_pred_tags | ["遊戲", "熱血", "日本輕小說", "青春", "鬥智"] | ["遊戲", "競技", "智斗", "日本輕小說", "青春"] |
| outside_tag_mapping | [{"source": "智鬥", "mapped_label": "鬥智", "similarity": 0.9190746115842447, "accepted": true}] | [] |

### q19

- query: 如題 最近看的書越來越多 總覺得自己也越來越挑了 希望有人能推薦一些好書 / 以下條件： / 1. 女主是（外表）蘿莉（可愛的少御也是可以啦） / 2.有戀愛的元素（後宮可以 但拜託不要無腦倒貼） / 3.希望是異世界（不一定要穿越） / 4.有傲嬌或無口或吃貨之類的屬性更好 / 拜託各位了
- required_tags: ["異世界", "戀愛"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["異世界", "蘿莉", "戀愛", "後宮", "奇幻", "歡樂"] | ["異世界", "蘿莉", "戀愛", "美食", "後宮", "奇幻"] |
| raw_extracted_tags | ["異世界", "蘿莉", "戀愛", "後宮", "奇幻", "歡樂"] | ["異世界", "蘿莉", "戀愛", "美食", "後宮", "奇幻"] |
| generated_keywords | ["萌屬性", "傲嬌", "無口", "吃貨", "少御", "異世界冒險", "純愛", "養成", "性格反差"] | ["傲嬌", "無口", "少御", "反差萌", "異世界生活", "純愛", "美食家", "胃口好"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["異世界", "蘿莉", "戀愛", "後宮", "奇幻", "歡樂"] | ["異世界", "蘿莉", "戀愛", "美食", "後宮", "奇幻"] |
| outside_tag_mapping | [] | [] |

### q20

- query: 本人喜歡偏重人物心理描寫、人物互動和角色變化的作品，較偏執在人物塑造和刻劃上。劇情不要求設計得有多精妙，不影響觀感就好，就算是平鋪直述的也行。節奏快慢無妨，掌握恰當即可。本人對於自己的閱讀能力還算有信心，文字量稍微過多也沒差。大忌是拿膚淺粗俗爛造的刻板僵化人物作為行銷點。
- required_tags: []
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["群像", "輕文學"] | ["群像", "輕文學", "青春", "治癒"] |
| raw_extracted_tags | ["群像", "輕文學"] | ["群像", "輕文學", "青春", "治癒"] |
| generated_keywords | ["人物弧光", "內心獨白", "情感細膩", "角色深度", "心理博弈", "人性刻畫"] | ["心理描寫", "角色成長", "人性探討", "內心獨白", "深層刻劃", "情感互動", "人物弧光"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["群像", "輕文學"] | ["群像", "輕文學", "青春", "治癒"] |
| outside_tag_mapping | [] | [] |

### q21

- query: 最近剛看完三角距離無限為零 / 雖然胃在痛不過還是很喜歡三角戀這個要素 / 有人可以推薦一下類似的作品嗎? / 當然不一定要三角戀 更多也行 只是想看這種描述男女主如何在戀愛中痛苦掙扎的小說 / 有糖更好 因為之後胃會更痛
- required_tags: ["戀愛"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["戀愛", "青春", "other"] | ["戀愛", "青春", "黑暗"] |
| raw_extracted_tags | ["戀愛", "青春", "other"] | ["戀愛", "青春", "黑暗"] |
| generated_keywords | ["虐心", "情感糾葛", "三角關係", "心碎", "酸甜", "愛而不得", "純愛", "心理掙扎"] | ["三角關係", "虐戀", "感情糾葛", "心碎", "苦戀", "情感拉扯", "多角關係", "酸甜"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["戀愛", "青春", "other"] | ["戀愛", "青春", "黑暗"] |
| outside_tag_mapping | [] | [] |

### q22

- query: 就是不後宮不NTR純純的戀愛作品 虐也行 比較希望校園類 其他類型也可以介紹看看 最近看一堆異世界後宮戰鬥有點膩了...... / 還有兄妹的作品 滿喜歡兄妹類型的 隨便丟一張圖
- required_tags: ["戀愛"]
- blocked_tags: ["後宮", "NTR"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["戀愛", "校園", "青春", "妹妹"] | ["戀愛", "校園", "妹妹", "青春"] |
| raw_extracted_tags | ["戀愛", "校園", "青春", "妹妹"] | ["戀愛", "校園", "妹妹", "青春"] |
| generated_keywords | ["單純戀愛", "純情", "禁斷之戀", "兄妹情", "校園日常", "虐心", "純愛", "單一女主角"] | ["純愛", "單純戀情", "禁斷之戀", "校園戀曲", "虐戀", "兄妹情", "純情", "青梅竹馬"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["戀愛", "校園", "青春", "妹妹"] | ["戀愛", "校園", "妹妹", "青春"] |
| outside_tag_mapping | [] | [] |

### q23

- query: 最近不知道要看什麼，想請大家推薦一些好看的作品 /  / 裝逼開掛類的 / 黑暗懸疑的 / 或是一些風格冷門的都還算喜歡 / 只有一個大前提，不要後宮，不要賣肉（好像變2個大前提了？ / 不喜歡那種明明男主沒幹什麼但每個女角都喜歡他，非得要的話2個是極限，而且劇情希望能別注重在他們愛來愛去互刷好感度卻又不決定人選的地方，如果是冒險類作品增加同伴的時候，可以的話別清一色都女的偶而也增加點男性同伴也好（都女的最後8成都變後宮 / 再來不要硬賣一些福利，例如某漫畫的黎斗摔，或者是洗澡時剛好開門，或者是組個樂團為了討老師歡心換了泳裝演奏……這種的不太能接受 /  / 喜歡偏向下剋上（不管是強者因未知原因被分到弱組還是靠努力爬上來把上位者打趴的都喜歡） / 描寫內心戲多的（黑掉壞掉，或是溫馨感動的都喜歡） / 不好意思這麼麻煩…因為總感覺現在大眾向的作品，開掛後宮福利都會有，希望能找到一些後宮福利少點的
- required_tags: []
- blocked_tags: ["後宮"]

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | False | False |
| blocked_clean | True | True |
| pred_tags | ["黑暗", "懸疑", "龍傲天", "冒險", "戰鬥", "智斗"] | ["龍傲天", "黑暗", "懸疑", "智斗", "冒險", "溫馨", "治癒"] |
| raw_extracted_tags | ["黑暗", "懸疑", "龍傲天", "冒險", "戰鬥", "智斗"] | ["龍傲天", "黑暗", "懸疑", "智斗", "冒險", "溫馨", "治癒"] |
| generated_keywords | ["實力碾壓", "反轉", "人性探討", "深層心理", "成長", "權力鬥爭", "冷酷主角", "非主流風格"] | ["逆襲", "心理博弈", "精神崩潰", "弱者反擊", "深度心理描寫", "冷門設定", "非後宮", "純愛", "階級跨越"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["黑暗", "懸疑", "龍傲天", "冒險", "戰鬥", "智斗"] | ["龍傲天", "黑暗", "懸疑", "智斗", "冒險", "溫馨", "治癒"] |
| outside_tag_mapping | [] | [] |

### q24

- query: 要求有點難找 / 希望推薦"戰爭+戀愛"要素的輕小說,不是戰"鬥"是戰"爭" / 要完結了的,GOOD ENDING最好
- required_tags: ["戀愛", "戰爭"]
- blocked_tags: []

| Field | No Description | With Description |
| --- | --- | --- |
| parse_success | True | True |
| parse_error |  |  |
| required_exact_cover | True | True |
| blocked_clean | True | True |
| pred_tags | ["戰爭", "戀愛"] | ["戰爭", "戀愛", "日本輕小說"] |
| raw_extracted_tags | ["戰爭", "戀愛"] | ["戰爭", "戀愛", "日本輕小說"] |
| generated_keywords | ["軍事", "戰略", "宏大敘事", "軍旅", "救贖", "和平", "陣地戰", "政治衝突"] | ["軍事", "戰略", "陣地戰", "戰地戀情", "終戰", "和平", "軍服", "政治衝突"] |
| outside_tags_raw | [] | [] |
| mapped_pred_tags | ["戰爭", "戀愛"] | ["戰爭", "戀愛", "日本輕小說"] |
| outside_tag_mapping | [] | [] |
