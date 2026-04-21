# Query Engine Parser Experiment Summary

整理日期：2026-04-21

本文件整理這輪針對查詢引擎 parser 流程的實驗、修復與結論。對照組是本分支原本的查詢 parser 主路徑：三段 LLM 呼叫皆以 JSON / response schema 為主要契約，並由 `generate_run` 產生 24 個查詢的實驗結果。

> 註：本機 Git for Windows 在執行 `git status` / `git diff` 時遇到 `sh.exe` 權限錯誤，無法直接產生完整 diff。因此本文以目前工作樹中的 source files、實驗 run 檔、metrics/analyze 腳本輸出，以及本輪每次實驗結果交叉核對。受影響的 Git 命令錯誤是環境層問題，不影響下列程式與 run JSON 的可讀性。

## 最終建議狀態

目前建議主路徑採用：

- `semantic_sections_v3_taglite`
- `structured` 保留固定四 key schema 與 raw text repair fallback
- retry 規則保留 `500/502/504` 等 transient provider errors
- local LLM timeout 預設保留 `180s`，必要時以 `HYBRID_REASONER_LLM_TIMEOUT_SECONDS=none` 關閉觀察長尾

不建議採用：

- `schema_off` prompt-only JSON：可解析性太差
- `deterministic` 作為主路徑：no-strict 較好，但 strict-only 較差，且 retry/latency 不穩定
- `deterministic_strict` 作為主路徑：tag 壓縮過度，strict-only 明顯退化

## 主要修改檔案

- `src/core/llm.py`
  - parser 主流程、marked sections parser、taglite、structured schema/repair、timeout。
- `src/core/engine.py`
  - `search()` 使用收斂後的預設 parser 主路徑。
- `src/core/api_utils.py`
  - retryable error 規則新增 `500/502/504` 與相關狀態字串。
- `src/eval/generate_run.py`
  - run metadata、單一預設 parser 實驗入口、query execution metadata。
- `src/eval/analyze_parser_runs.py`
  - 新增 branch telemetry 分析。

## 問題 1：原本無法看出三段 LLM 呼叫各自表現

### 原始問題

查詢 parser 有三次 LLM 呼叫：

1. `semantic_understanding`
2. `tag_projection`
3. `structured`

原本 run 結果只看最後書單與整體成功/失敗，無法判斷哪一段造成失敗、重試、長尾 latency 或 parse fallback。因此最初很難回答「response schema 是否使品質下降」，也無法知道三段 LLM 的特性是否不同。

### 解法

新增 branch-level telemetry：

- `success`
- `latency_ms`
- `request_count`
- `retry_count`
- `first_attempt_success`
- `used_response_schema`
- `parse_source`
- `recovered_from_raw_text`
- `last_retry_error`

同時在 query result 中記錄：

- `execution_metadata`
- `parse_metadata`
- `parser_variant`
- `parser_variant`

新增 `src/eval/analyze_parser_runs.py`，可直接分析每個 run JSON 的 query 成功率、branch 成功率、retry、latency、parse source。

### 效果

第一輪分析立刻確認三段 branch 表現差異很大：

- `semantic_understanding` 是最早期最不穩的 bottleneck。
- `tag_projection` 在 semantic 改穩後成為新的長尾瓶頸。
- `structured` 原本流程成功率高，但 SDK parsed 可靠性差，大量依賴 raw text fallback。

代表本輪後續優化能聚焦在正確位置，而不是只看整體 Avg@10。

## 問題 2：prompt-only JSON 不可用

### 原始問題

為了驗證 response schema 是否降低品質，做過 `schema_on` vs `schema_off`。`schema_off` 不使用 provider response schema，只靠 prompt 要求 JSON。

### 實驗結果

`batch_20260419_154338`：

| variant | query success | 主要失敗點 |
|---|---:|---|
| `gemma4_schema_on` | 19/24 | semantic/tag timeout 或 parse failure |
| `gemma4_schema_off` | 0/24 | semantic_understanding 全失敗 |

`schema_off` 的 24 筆都卡在 `semantic_understanding`，後續 branch 沒機會執行。

### 結論

prompt-only JSON 不是可用替代方案。模型內容常看起來像 JSON，但 SDK parsed payload 不存在，raw text 也不穩定，導致 machine-readable failure。這個方向停止。

## 問題 3：semantic_understanding 使用嚴格 JSON/schema 時不穩且成本高

### 原始問題

`semantic_understanding` 原本輸出 JSON，欄位包含：

- `semantic_query_text`
- `intent_summary`
- `positive_concepts`
- `negative_concepts`
- `ambiguities`

實驗顯示它是早期最不穩的 branch。`schema_on` 在 `batch_20260419_154338` 的 semantic branch：

- success：20/24
- avg request count：16.71
- avg retry count：15.71
- avg latency：約 24.7s

這代表 schema + JSON 對 semantic branch 造成很重的重試與長尾。

### 解法

把 semantic branch 改成 marked sections 輸出，不再用 JSON：

```text
[semantic_query_text]
...
[intent_summary]
...
[positive_concepts]
- ...
[negative_concepts]
- ...
```

並新增 section parser：

- `_parse_marked_section_output`
- `_parse_semantic_sections_text`
- `_parse_semantic_sections_v3_text`

最終採用 `semantic_sections_v3`：保留短版 `intent_summary`，避免 v2 因資訊太瘦讓 tag branch 變差。

### 效果

`batch_20260419_234359`：

| variant | query success | no-strict Avg@10 | strict Avg@10 |
|---|---:|---:|---:|
| `schema_on_v3` | 約 81.25% coverage-adjusted | 1.8771 | 1.7304 |
| `semantic_sections_v3` | 約 93.75% coverage-adjusted | 2.2021 | 2.0083 |

結論：

- marked sections 明確改善 semantic branch 穩定性。
- 補回短版 `intent_summary` 是必要的，因為純 3 欄版本會讓 tag branch 輸入不足。

## 問題 4：tag_projection 長尾與生成失控

### 原始問題

semantic branch 改穩後，`tag_projection` 開始暴露長尾問題。拿掉 45s timeout 後，部分 query 不是「再等一下就成功」，而是等待約 900 秒後回 malformed JSON 或重複垃圾。

典型案例：

- `q4`：tag branch 長時間後輸出重複 `龍傲天`
- `q21`：tag branch 長時間後亂噴 `病嬌`、`龍傲天`

### 解法

新增 `semantic_sections_v3_taglite`：

- semantic 繼續使用 marked sections v3。
- tag projection 改吃 compact semantic context。
- 只傳：
  - `semantic_query_text`
  - `intent_summary`
  - 最多 4 個 strongest positive concepts
  - 最多 3 個 strongest negative concepts
- tag prompt 改成保守投影：
  - positive terms 3-6
  - negative terms 0-4
  - 不要近義詞、重複詞、弱概念、example-derived concepts

### 效果

`batch_20260420_135431`：

| variant | query success | no-strict Avg@10 | strict Avg@10 | coverage-adjusted strict |
|---|---:|---:|---:|---:|
| `schema_on_v3` | 20/24 | 2.3700 | 2.0320 | 1.6933 |
| `semantic_sections_v3_taglite` | 24/24 | 2.2958 | 2.1619 | 2.1619 |

Branch latency：

| branch | `schema_on_v3` avg | `taglite` avg |
|---|---:|---:|
| semantic | 85.7s | 23.5s |
| tag | 86.8s | 2.5s |

結論：

- taglite 是本輪最重要的主流程修復。
- 它把 tag branch 長尾直接壓掉。
- 雖然 no-strict 單看成功 query 不一定最高，但 coverage-adjusted 品質明顯改善。

## 問題 5：移除 timeout 後會暴露 900 秒級長尾

### 原始問題

為了觀察模型實際花時間，我們一度把 45s local timeout 拿掉。結果發現部分失敗不是正常慢回，而是 provider/model stall：

- 約 900 秒後回空結果
- 或回 malformed JSON
- 或出現重複垃圾

這表示「完全無 timeout」不適合作為產品主路徑。

### 解法

新增可設定的 timeout：

- 預設 `180s`
- 可用 `HYBRID_REASONER_LLM_TIMEOUT_SECONDS=none` 關閉
- 也支援環境變數指定秒數

### 效果

在穩定版 `semantic_sections_v3_taglite` 下，正常 batch 的 branch latency 遠低於 180s：

`batch_20260420_162103`：

| branch | mean | p95 | max |
|---|---:|---:|---:|
| semantic | 23.1s | 38.6s | 43.9s |
| tag | 2.4s | 2.9s | 3.0s |
| structured | 2.6s | 2.9s | 3.0s |

結論：

- `180s` 足以保護主流程不進入 900 秒級 stall。
- 同時不會誤殺目前正常請求。

## 問題 6：structured 大量依賴 raw_text_json fallback

### 原始問題

`structured` branch 原本成功率看起來高，但 SDK parsed payload 很不穩，常常靠 raw text fallback。

例：

`batch_20260420_162103`：

- structured success：24/24
- `sdk_parsed_dict`：5/24
- `raw_text_json`：19/24

這代表 structured 的 response schema 對 SDK 來說不夠穩，雖然 fallback 讓結果能用，但長期維護風險高。

### 解法

修改 structured output contract：

- 固定回傳四個 top-level keys：
  - `target_status_candidate`
  - `author_name_candidate`
  - `words_min_candidate`
  - `words_max_candidate`
- 不再允許缺 key。
- 若沒有 constraint，回 empty candidate：

```json
{"value":"","evidence":"","is_explicit":false}
```

或 numeric：

```json
{"value":0,"evidence":"","is_explicit":false}
```

同時補兩層 parser fallback：

- `_parse_json_object_from_text()` 加入 `ast.literal_eval` 嘗試。
- `_repair_structured_payload_from_text()` 可從半殘 candidate fragment 裡修復 structured payload。

### 效果

修正後 `batch_20260420_223354`：

- structured success：24/24
- `sdk_parsed_dict`：24/24
- `raw_text_json`：0/24

品質幾乎持平：

| batch | no-strict Avg@10 | strict Avg@10 |
|---|---:|---:|
| before structured schema fix, `20260420_162103` | 2.3208 | 2.0358 |
| after structured schema fix, `20260420_223354` | 2.3125 | 2.0348 |

結論：

- structured 可解析性已解。
- 這個修復主要改善穩定性與可維護性，不是直接提升書單品質。

## 問題 7：Gemini response_schema 不支援 additionalProperties

### 原始問題

為了讓 structured schema 更嚴格，一度加入 `additionalProperties: False`。結果 Gemini API 不接受此欄位，造成整批 structured request 直接 `400 INVALID_ARGUMENT`。

`batch_20260420_214620`：

- query success：0/24
- semantic：24/24 成功
- tag：24/24 成功
- structured：0/24，全部 400

錯誤：

```text
Unknown name "additional_properties" at generation_config.response_schema
```

### 解法

移除所有 `additionalProperties`。

### 效果

下一批 `batch_20260420_223354` 回到：

- query success：24/24
- structured：24/24 `sdk_parsed_dict`

結論：

- Gemini response schema 子集不等同完整 JSON Schema。
- 後續不要在 provider schema 中使用未確認支援的 JSON Schema 欄位。

## 問題 8：structured prompt 中 JSON 範例造成 f-string format error

### 原始問題

`structured_instruction` 是 f-string，prompt 裡的 JSON example 沒跳脫大括號，導致 Python 在組 prompt 時報錯：

```text
Invalid format specifier '"","evidence":"","is_explicit":false' for object of type 'str'
```

### 解法

把 prompt 裡 JSON 範例的大括號改為 f-string escape：

```text
{{"value":"","evidence":"","is_explicit":false}}
{{"value":0,"evidence":"","is_explicit":false}}
```

### 效果

- `py_compile` 通過。
- `from src.core.llm import parse_query` 成功。
- 後續 run 不再出現此錯誤。

## 問題 9：provider 500 INTERNAL 沒被當成 retryable

### 原始問題

`batch_20260421_010711` 出現大量失敗，但錯誤幾乎都是：

```text
500 INTERNAL
```

發生在 semantic/tag/structured 各 branch，且多在 1-1.5s 內失敗。這是 provider transient error，不是 parser 語義失敗。

原本 `_is_retryable()` 只處理：

- 429
- 503
- 少數 Windows/network errors

導致 500 被當作 final failure。

### 解法

在 `src/core/api_utils.py` 將下列錯誤納入 retryable：

- `500` / `INTERNAL`
- `502` / `BAD_GATEWAY`
- `504` / `DEADLINE_EXCEEDED` / `GATEWAY_TIMEOUT`

### 效果

修正後 `batch_20260421_134156`：

- default：24/24 success
- deterministic：24/24 success

`batch_20260421_195116`：

- default：24/24 success
- deterministic：24/24 success
- deterministic_strict：24/24 success

結論：

- transient provider error 不再把整批實驗打穿。
- retry 後仍能成功完成所有 query。

## 問題 10：需要比較 parser profile 的 deterministic 實驗

### 目的

在停止 schema/format 改造後，測試是否能用低溫與 prompt 收斂降低跨 run 波動。

### 新增 profile

新增：

- `default`
- `deterministic`
- `deterministic_strict`

`deterministic`：

- semantic：`temperature=0.0`, `top_p=0.2`
- tag/structured：`temperature=0.0`, `top_p=0.1`
- prompt 加入少推測、固定排序、少輸出規則

`deterministic_strict`：

- semantic：`temperature=0.0`, `top_p=0.15`
- tag/structured：`temperature=0.0`, `top_p=0.1`
- tag prompt 要求 2-3 positive terms
- tag normalize 後最多保留 3 個 positive terms 與 3 個 negative terms

### 實驗結果

`batch_20260421_195116`：

| profile | query success | no-strict Avg@10 | strict Avg@10 |
|---|---:|---:|---:|
| default | 24/24 | 2.3208 | 2.2369 |
| deterministic | 24/24 | 2.4042 | 2.1098 |
| deterministic_strict | 24/24 | 2.3833 | 1.8469 |

平均 tag 數：

| profile | avg positive_count | avg negative_count |
|---|---:|---:|
| default | 3.50 | 1.71 |
| deterministic | 3.83 | 1.42 |
| deterministic_strict | 2.04 | 1.21 |

### 結論

`deterministic`：

- no-strict 最高。
- strict-only 低於 default。
- 較像 recall-oriented profile，不適合作為 strict 主路徑。

`deterministic_strict`：

- 成功壓低 positive tag 數。
- 但壓得太多，很多 query 只剩 1 個正向 tag。
- strict-only 明顯退化。
- 不建議採用。

失敗案例：

- `q13` 只剩 `歡樂向`，丟掉 `腦洞 / 誤會系`。
- `q17` 只剩 `危機感`，丟掉 `鬥智 / 經營 / 懸疑 / 異世界`。
- `q9` 只剩 `奇幻`，丟掉 `都市奇幻 / 異能 / 人外 / 懸疑`。
- `q6` 只剩 `戰爭`，丟掉 `鬥智`。
- `q1` 只剩 `戀愛`，狗糧/溫馨感下降。

因此最終不採用 deterministic profiles 作為主路徑。

## 最終效果總表

### 穩定性

| 階段 | query success | 主要問題 |
|---|---:|---|
| 原 schema_on baseline, `20260419_154338` | 19/24 | semantic retry/timeout, tag failure |
| prompt-only schema_off | 0/24 | semantic JSON parse 全失敗 |
| semantic_sections v3 | 約 93.75% coverage | tag branch 仍需減載 |
| semantic_sections_v3_taglite, `20260420_135431` | 24/24 | structured 仍大量 raw fallback |
| structured schema fix, `20260420_223354` | 24/24 | structured SDK parsed 已修 |
| retry 500 fix 後, `20260421_195116` | 24/24 across profiles | provider transient error 可恢復 |

### 品質

| run | variant/profile | no-strict Avg@10 | strict Avg@10 | coverage-adjusted strict |
|---|---|---:|---:|---:|
| `20260419_154338` | schema_on | 2.3632 | 2.0671 | 1.6365 |
| `20260420_135431` | schema_on_v3 | 2.3700 | 2.0320 | 1.6933 |
| `20260420_135431` | semantic_sections_v3_taglite | 2.2958 | 2.1619 | 2.1619 |
| `20260420_223354` | default parser | 2.3125 | 2.0348 | 2.0348 |
| `20260421_195116` | default profile | 2.3208 | 2.2369 | 2.2369 |
| `20260421_195116` | deterministic | 2.4042 | 2.1098 | 2.1098 |
| `20260421_195116` | deterministic_strict | 2.3833 | 1.8469 | 1.8469 |

解讀：

- 最終主流程在成功率與穩定性上明顯優於原 baseline。
- strict 品質受單次抽樣波動影響，但目前 default profile 是 strict-only 最穩的選擇。
- deterministic_strict 的策略被證明太 aggressive，應停止。

## 已移除的實驗邏輯

本輪 cleanup 已移除下列不建議採用且會增加主流程複雜度的實驗邏輯：

- `parser_profile = deterministic`
- `parser_profile = deterministic_strict`
- `generate_run.py` 的三組 profile 對照
- `engine.search()` / `parse_query()` 的 `parser_profile` 入口

目前程式已收斂為單一產品主線，只保留：

- default parser
- branch telemetry
- structured parser repair
- retryable 500/502/504
- `analyze_parser_runs.py`

## 未完全解決 / 後續可做但目前停止的方向

### 1. semantic precision 波動

仍存在跨 run 差異。例如 `q13`, `q9`, `q17`, `q6`, `q1` 等查詢，semantic/tag 詞的選擇會影響 strict-only。

但 deterministic profiles 沒有帶來更好的 strict 結果，因此目前不繼續投入。

### 2. metrics.py 會略過失敗 query

`metrics.py` 的官方輸出只看成功產生 candidates 的 query。這在失敗率高的早期實驗會高估品質。因此分析時需要搭配 coverage-adjusted metrics。

目前尚未把 coverage-adjusted 指標正式整合進 `metrics.py`。

### 3. Git diff 無法自動產生

因本機 Git for Windows 權限問題，未能直接產生對 HEAD 的完整 diff。若要做 PR 前整理，建議在 Git 可正常運作的環境中重跑：

```bash
git status --short
git diff --stat
git diff
```

## 最終結論

本輪真正解決的查詢引擎問題有四個核心：

1. Parser 可觀測性不足：已由 branch telemetry 與 `analyze_parser_runs.py` 解決。
2. semantic JSON/schema branch 不穩：已由 marked sections v3 解決。
3. tag_projection 長尾與生成失控：已由 compact context + taglite prompt 解決。
4. structured SDK parsed 不穩：已由固定四 key schema + repair fallback 解決。
5. provider transient 500 未重試：已由 retryable 規則修正。

目前最穩定的主路徑是 `semantic_sections_v3_taglite` 單一路徑。deterministic 系列實驗提供了重要反證：降低 sampling 與強制壓縮 tag 不一定提高 strict precision，甚至會因為刪掉必要 facet 而退化。因此本輪 cleanup 已移除 deterministic profile 入口，保留 default 行為作為主線。
