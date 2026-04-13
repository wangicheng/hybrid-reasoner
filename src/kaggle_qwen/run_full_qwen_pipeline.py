import argparse
import json
import subprocess
import sys
from pathlib import Path


def gpu_supports_bf16() -> bool:
    try:
        import torch  # local import to keep this script lightweight
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        return bool(checker())
    return False


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison(base_report: dict, tuned_report: dict) -> dict:
    metrics = {
        "parse_success_rate": {
            "base": float(base_report.get("parse_success_rate", 0.0)),
            "tuned": float(tuned_report.get("parse_success_rate", 0.0)),
        },
        "raw_outside_taxonomy_rate": {
            "base": float(base_report.get("raw_outside_taxonomy_rate", 0.0)),
            "tuned": float(tuned_report.get("raw_outside_taxonomy_rate", 0.0)),
        },
        "exact_match_rate": {
            "base": float(base_report.get("exact_match_rate", 0.0)),
            "tuned": float(tuned_report.get("exact_match_rate", 0.0)),
        },
        "micro_f1": {
            "base": float((base_report.get("micro") or {}).get("f1", 0.0)),
            "tuned": float((tuned_report.get("micro") or {}).get("f1", 0.0)),
        },
        "macro_f1": {
            "base": float((base_report.get("macro") or {}).get("f1", 0.0)),
            "tuned": float((tuned_report.get("macro") or {}).get("f1", 0.0)),
        },
    }

    for item in metrics.values():
        item["delta_tuned_minus_base"] = item["tuned"] - item["base"]

    return {
        "total_samples": int(tuned_report.get("total_samples", 0)),
        "base_model": tuned_report.get("base_model", ""),
        "adapter_path": tuned_report.get("adapter_path", ""),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Qwen2.5-7B pipeline: prepare -> train -> validate.")

    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--tags-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/qwen_pipeline"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-thinking-chars", type=int, default=40)
    parser.add_argument("--max-thinking-chars", type=int, default=320)

    parser.add_argument("--disable-drop-empty-tags", action="store_true")
    parser.add_argument("--disable-fix-exclusion-language", action="store_true")
    parser.add_argument("--disable-truncate-thinking", action="store_true")

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--disable-bf16", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")

    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--skip-base-eval", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    prepare_script = script_dir / "prepare_qwen_training_data.py"
    train_script = script_dir / "train_qwen25_7b_qlora.py"
    validate_script = script_dir / "validate_qwen_tagger.py"

    data_dir = args.work_dir / "data"
    train_out_dir = args.work_dir / "train_output"
    eval_out_dir = args.work_dir / "eval_output"
    tuned_eval_out_dir = eval_out_dir / "tuned"
    base_eval_out_dir = eval_out_dir / "base"

    data_dir.mkdir(parents=True, exist_ok=True)
    train_out_dir.mkdir(parents=True, exist_ok=True)
    tuned_eval_out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_base_eval:
        base_eval_out_dir.mkdir(parents=True, exist_ok=True)

    prepare_cmd = [
        sys.executable,
        str(prepare_script),
        "--input-jsonl",
        str(args.input_jsonl),
        "--tags-path",
        str(args.tags_path),
        "--output-dir",
        str(data_dir),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--min-thinking-chars",
        str(args.min_thinking_chars),
        "--max-thinking-chars",
        str(args.max_thinking_chars),
    ]
    if not args.disable_drop_empty_tags:
        prepare_cmd.append("--drop-empty-tags")
    if not args.disable_fix_exclusion_language:
        prepare_cmd.append("--fix-exclusion-language")
    if not args.disable_truncate_thinking:
        prepare_cmd.append("--truncate-thinking")

    run_cmd(prepare_cmd)

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    train_cmd = [
        sys.executable,
        str(train_script),
        "--train-file",
        str(train_file),
        "--val-file",
        str(val_file),
        "--base-model",
        args.base_model,
        "--output-dir",
        str(train_out_dir),
        "--max-seq-length",
        str(args.max_seq_length),
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--seed",
        str(args.seed),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
    ]
    if not args.disable_4bit:
        train_cmd.append("--load-in-4bit")
    enable_bf16 = (not args.disable_bf16) and gpu_supports_bf16()
    if enable_bf16:
        train_cmd.append("--bf16")
    else:
        print("[info] bf16 disabled (unsupported GPU or flag disabled).")
    if not args.disable_gradient_checkpointing:
        train_cmd.append("--gradient-checkpointing")

    run_cmd(train_cmd)

    validate_cmd = [
        sys.executable,
        str(validate_script),
        "--val-file",
        str(val_file),
        "--tags-path",
        str(args.tags_path),
        "--base-model",
        args.base_model,
        "--adapter-path",
        str(train_out_dir),
        "--output-dir",
        str(tuned_eval_out_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-samples",
        str(args.max_val_samples),
    ]
    if not args.disable_4bit:
        validate_cmd.append("--load-in-4bit")
    if enable_bf16:
        validate_cmd.append("--bf16")

    run_cmd(validate_cmd)

    if not args.skip_base_eval:
        base_validate_cmd = [
            sys.executable,
            str(validate_script),
            "--val-file",
            str(val_file),
            "--tags-path",
            str(args.tags_path),
            "--base-model",
            args.base_model,
            "--output-dir",
            str(base_eval_out_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--max-samples",
            str(args.max_val_samples),
        ]
        if not args.disable_4bit:
            base_validate_cmd.append("--load-in-4bit")
        if enable_bf16:
            base_validate_cmd.append("--bf16")

        run_cmd(base_validate_cmd)

        tuned_report = load_report(tuned_eval_out_dir / "validation_report.json")
        base_report = load_report(base_eval_out_dir / "validation_report.json")
        comparison = build_comparison(base_report=base_report, tuned_report=tuned_report)
        comparison_path = eval_out_dir / "model_comparison.json"
        with comparison_path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        print(f"[done] model comparison written: {comparison_path}")

    print("[done] full pipeline completed")
    print(f"  data_dir:      {data_dir}")
    print(f"  train_output:  {train_out_dir}")
    print(f"  eval_output:   {eval_out_dir}")
    print(f"  tuned_eval_output: {tuned_eval_out_dir}")
    if not args.skip_base_eval:
        print(f"  base_eval_output:  {base_eval_out_dir}")


if __name__ == "__main__":
    main()
