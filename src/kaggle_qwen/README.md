# Qwen2.5-7B (Kaggle) Training Kit

This folder contains a complete pipeline for training and validating a Qwen2.5-7B tag extractor on Kaggle.

## Files

- `prepare_qwen_training_data.py`: data cleaning, analysis, and split generation
- `train_qwen25_7b_qlora.py`: QLoRA training script for Qwen2.5-7B-Instruct
- `validate_qwen_tagger.py`: post-training validation script with metrics and taxonomy checks
- `run_full_qwen_pipeline.py`: one-command pipeline (prepare -> train -> validate)
- `kaggle_train_validate_direct.py`: robust Kaggle entrypoint with live logs and safe defaults

## Recommended Kaggle Setup

- Accelerator: GPU (T4, L4, or V100)
- Internet: ON
- Add your dataset files (at least):
  - `exercises_v2_gemini_intro_only.jsonl`
  - `all_tags.json`
  - these scripts

Install dependencies in the first notebook cell:

```bash
!pip -q install transformers datasets accelerate peft bitsandbytes trl sentencepiece
```

## Step 1: Prepare and Repair Training Data

This step performs analysis and fixes before training:

- removes out-of-taxonomy tags
- removes duplicate tags
- can remove exclusion-language sentences from thinking
- can truncate overlong thinking
- drops rows that remain invalid after fixes
- writes a full report for audit

```bash
!python src/kaggle_qwen/prepare_qwen_training_data.py \
  --input-jsonl /kaggle/input/YOUR_DATASET/exercises_v2_gemini_intro_only.jsonl \
  --tags-path /kaggle/input/YOUR_DATASET/all_tags.json \
  --output-dir /kaggle/working/qwen_data \
  --val-ratio 0.1 \
  --seed 42 \
  --min-thinking-chars 40 \
  --max-thinking-chars 320 \
  --fix-exclusion-language \
  --truncate-thinking
```

Outputs:

- `/kaggle/working/qwen_data/cleaned_all.jsonl`
- `/kaggle/working/qwen_data/train.jsonl`
- `/kaggle/working/qwen_data/val.jsonl`
- `/kaggle/working/qwen_data/prep_report.json`

## Step 2: Train Qwen2.5-7B with QLoRA

Default hyperparameters follow the recommended baseline:

- learning rate: 2e-4
- batch size: 1
- gradient accumulation: 16
- epochs: 4
- max sequence length: 1024
- optimizer: paged_adamw_8bit
- warmup ratio: 0.05

```bash
!python src/kaggle_qwen/train_qwen25_7b_qlora.py \
  --train-file /kaggle/working/qwen_data/train.jsonl \
  --val-file /kaggle/working/qwen_data/val.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir /kaggle/working/qwen25_7b_tag_lora \
  --max-seq-length 1024 \
  --num-train-epochs 4 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --logging-steps 10 \
  --eval-steps 50 \
  --save-steps 100 \
  --seed 42 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --load-in-4bit \
  --gradient-checkpointing
```

Training outputs:

- `/kaggle/working/qwen25_7b_tag_lora/` (LoRA adapter + tokenizer)
- `/kaggle/working/qwen25_7b_tag_lora/train_metrics.json`
- `/kaggle/working/qwen25_7b_tag_lora/run_config.json`

## Step 3: Validate Model Quality (Tuned Model)

This script generates predictions on the holdout validation set and reports:

- parse success rate
- out-of-taxonomy rate in raw model outputs
- exact match rate
- micro precision/recall/F1
- macro precision/recall/F1

```bash
!python src/kaggle_qwen/validate_qwen_tagger.py \
  --val-file /kaggle/working/qwen_data/val.jsonl \
  --tags-path /kaggle/input/YOUR_DATASET/all_tags.json \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-path /kaggle/working/qwen25_7b_tag_lora \
  --output-dir /kaggle/working/qwen25_7b_eval \
  --max-new-tokens 220 \
  --temperature 0.0 \
  --top-p 1.0 \
  --load-in-4bit
```

Validation outputs:

- `/kaggle/working/qwen25_7b_eval/predictions.jsonl`
- `/kaggle/working/qwen25_7b_eval/validation_report.json`

## Step 4: Validate Base Model (for Fair Comparison)

Run the same validation set with the original base model (without adapter):

```bash
!python src/kaggle_qwen/validate_qwen_tagger.py \
  --val-file /kaggle/working/qwen_data/val.jsonl \
  --tags-path /kaggle/input/YOUR_DATASET/all_tags.json \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir /kaggle/working/qwen25_7b_eval_base \
  --max-new-tokens 220 \
  --temperature 0.0 \
  --top-p 1.0 \
  --load-in-4bit
```

Then compare `validation_report.json` from:

- tuned model: `/kaggle/working/qwen25_7b_eval/validation_report.json`
- base model: `/kaggle/working/qwen25_7b_eval_base/validation_report.json`

## One-Command Full Pipeline

If you want to run everything in one shot:

```bash
!python src/kaggle_qwen/run_full_qwen_pipeline.py \
  --input-jsonl /kaggle/input/YOUR_DATASET/exercises_v2_gemini_intro_only.jsonl \
  --tags-path /kaggle/input/YOUR_DATASET/all_tags.json \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --work-dir /kaggle/working/qwen_pipeline
```

This command creates:

- `/kaggle/working/qwen_pipeline/data/` (prepared train/val + prep report)
- `/kaggle/working/qwen_pipeline/train_output/` (LoRA adapter + training metrics)
- `/kaggle/working/qwen_pipeline/eval_output/tuned/` (tuned predictions + validation report)
- `/kaggle/working/qwen_pipeline/eval_output/base/` (base predictions + validation report)
- `/kaggle/working/qwen_pipeline/eval_output/model_comparison.json` (base vs tuned metric deltas)

## Recommended Single Command on Kaggle

If you uploaded scripts and data into a Kaggle dataset folder, use this robust entrypoint:

```bash
!python /kaggle/input/YOUR_DATASET/kaggle_train_validate_direct.py
```

Optional explicit dataset path:

```bash
!python /kaggle/input/YOUR_DATASET/kaggle_train_validate_direct.py \
  --dataset-root /kaggle/input/YOUR_DATASET
```

If you want to skip base-model validation (faster but no comparison file):

```bash
!python /kaggle/input/YOUR_DATASET/kaggle_train_validate_direct.py \
  --dataset-root /kaggle/input/YOUR_DATASET \
  --skip-base-eval
```

## Analysis and Repair Report Guide

`prep_report.json` records what was fixed and what was dropped during preparation.

- `drop_reasons`: why rows were removed (for example empty intro or too-short thinking)
- `fix_counts`: how many fixes were applied (for example invalid tags removed)
- `kept_rows` / `input_rows`: effective retention ratio

If `drop_reasons` spikes or `kept_rows` is too low, revise upstream generation prompt and regenerate data.

## Notes

- If your current dataset size is small, this is still useful as a baseline run, but overfitting risk is high.
- Prefer continuing data generation and retraining after increasing sample size.
- Keep `all_tags.json` unchanged between preparation, training, and validation.
