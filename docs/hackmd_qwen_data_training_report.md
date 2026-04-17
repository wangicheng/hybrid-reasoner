# Hybrid Reasoner 標籤抽取實驗報告（方法與原理版）

## 摘要

本實驗針對「小說簡介多標籤抽取」任務，使用固定 taxonomy，採用 `Qwen/Qwen2.5-7B-Instruct` 搭配 QLoRA 微調。核心設計重點是：

1. 先以嚴格規則清洗 JSONL 訓練資料，確保標籤合法且輸出格式可學習。
2. 以 4-bit 量化 + LoRA 進行高效微調，降低 GPU 記憶體需求。
3. 在同一驗證集上比較 tuned 與 base，使用統一指標判斷增益。

---

## 1. 研究問題與方法概觀

### 1.1 任務定義

輸入：小說簡介 `input_intro`。  
輸出：標籤集合 `final_tags`，且每個標籤必須屬於 `all_tags.json`。

本任務本質是「受限標籤空間的多標籤分類」，但透過生成式 LLM 以 JSON 形式輸出。

### 1.2 實驗假設

1. 若訓練資料強化「合法 JSON + 合法 taxonomy + 證據導向 thinking」，可提升可解析率與標籤一致性。
2. QLoRA 能在有限資源下取得相對 base model 的可觀增益。
3. tuned model 在 micro/macro F1 與 exact match 上應高於 base model。

### 1.3 Pipeline

本專案實際流程（`kaggle_train_validate_direct.py`）為：

1. 資料準備：`prepare_qwen_training_data.py`
2. 模型訓練：`train_qwen25_7b_qlora.py`
3. tuned 驗證：`validate_qwen_tagger.py --adapter-path ...`
4. base 驗證：`validate_qwen_tagger.py`（不帶 adapter）
5. 報告比較：輸出 `model_comparison.json`

---

## 2. 方法設計

## 2.1 資料方法

### 2.1.1 訓練樣本格式

每筆樣本包含：

- `input_intro`：模型判斷依據
- `thinking`：中間推理文字（限制 40-320 字）
- `final_tags`：最終標籤（list）
- `messages`：chat template 可直接訓練的 system/user/assistant 結構

### 2.1.2 清洗原則

`prepare_qwen_training_data.py` 的主要清洗策略：

1. taxonomy 約束：移除不在 `all_tags.json` 的標籤。
2. 去重：移除重複標籤。
3. thinking 規格化：可移除排除式語句（例如「不選哪些標籤」）。
4. 長度控制：截斷過長 thinking，丟棄過短或仍過長樣本。
5. split 一致化：固定 seed 與 `val_ratio` 進行 train/val 切分。

此設計目的在於降低 supervision noise，避免模型學到非法輸出模式。

## 2.2 模型方法

### 2.2.1 為何使用 QLoRA

QLoRA 透過「凍結基底權重 + 小量可訓練低秩矩陣」達成高效微調。實作上採：

- 4-bit 量化（NF4）減少顯存占用
- LoRA 套用在 attention 與 MLP 投影層（`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`）

原理上可視為權重更新：

$$
W' = W + \Delta W, \quad \Delta W = BA
$$

其中 $W$ 為凍結原始權重，$A, B$ 為低秩可訓練矩陣（rank = $r$）。

### 2.2.2 訓練目標

本實驗採自回歸語言模型目標，對 assistant token 進行 next-token learning。  
tokenization 後以 padding mask 將無效位置標成 `-100`，不計入 loss。

### 2.2.3 核心超參數（本專案 Kaggle 預設）

- base model: `Qwen/Qwen2.5-7B-Instruct`
- `max_seq_length=768`
- `num_train_epochs=4`
- `learning_rate=2e-4`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=24`
- `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- `load_in_4bit=True`, `gradient_checkpointing=True`

---

## 3. 評估方法與指標原理

## 3.1 評估流程

`validate_qwen_tagger.py` 對每筆驗證資料：

1. 生成模型輸出（JSON 文字）。
2. 解析 `final_tags`。
3. 依 taxonomy 正規化標籤。
4. 計算樣本級與整體級指標。

## 3.2 指標定義

### 3.2.1 格式與約束指標

- `parse_success_rate`：輸出可被 JSON 解析的比例
- `raw_outside_taxonomy_rate`：原始預測標籤中，落在 taxonomy 外的比例（越低越好）

### 3.2.2 多標籤品質指標

- `exact_match_rate`：預測集合與 gold 集合完全相同的比例
- `micro` Precision/Recall/F1：先全域累積 TP/FP/FN 再計算
- `macro` Precision/Recall/F1：對每個 label 各自計算後取平均

公式如下：

$$
P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}, \quad F1 = \frac{2PR}{P+R}
$$

micro 反映整體樣本分布下表現，macro 對長尾標籤更敏感，因此兩者需同時觀察。

---

## 4. 實驗執行設定（精簡）

- 資料切分：`val_ratio=0.1`, `seed=42`
- 解碼設定：`temperature=0.0`, `top_p=1.0`, `max_new_tokens=220`
- 比較策略：同一驗證集，同一解碼參數，比較 tuned vs base

主要輸出檔：

- `prep_report.json`（資料清洗統計）
- `train_metrics.json`（訓練過程）
- `validation_report.json`（單模型驗證）
- `model_comparison.json`（tuned-base 差值）

---

## 5. 結果解讀框架

若 `model_comparison.json` 中多數關鍵指標滿足下列方向，代表微調有效：

1. `delta(parse_success_rate) > 0`
2. `delta(raw_outside_taxonomy_rate) < 0`
3. `delta(micro_f1) > 0`
4. `delta(macro_f1) > 0`
5. `delta(exact_match_rate) > 0`

若 micro 提升但 macro 無明顯改善，通常表示模型主要學到高頻標籤，長尾標籤仍需增加資料覆蓋。

---

## 6. 結論

本實驗方法的核心價值在於：

1. 以資料規格化先控制輸出邊界（合法 JSON、合法標籤）。
2. 以 QLoRA 平衡訓練成本與性能增益。
3. 以 tuned/base 同場比較，避免僅看單一模型分數造成誤判。

在固定 taxonomy 的小說標籤任務中，此流程可作為可重現、可擴充的基準實驗設計。

---

## 附錄 A：最小執行指令（簡化）

```bash
!pip -q install --upgrade transformers datasets accelerate peft bitsandbytes trl sentencepiece

!python /kaggle/input/YOUR_DATASET/kaggle_train_validate_direct.py \
  --dataset-root /kaggle/input/YOUR_DATASET \
  --work-dir /kaggle/working/qwen_pipeline
```

以上指令會一次完成 prepare/train/validate/comparison。
