import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = (
    "prepare_qwen_training_data.py",
    "train_qwen25_7b_qlora.py",
    "validate_qwen_tagger.py",
    "all_tags.json",
    "exercises_v2_gemini_intro_only.jsonl",
)


def run_live(cmd: list[str], title: str) -> None:
    print(f"\n===== {title} =====")
    print(" ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{title} failed with exit code {code}")


def find_dataset_root(dataset_hint: Path | None) -> Path:
    if dataset_hint is not None:
        missing = [name for name in REQUIRED_FILES if not (dataset_hint / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"dataset root {dataset_hint} is missing files: {', '.join(missing)}"
            )
        return dataset_hint

    base = Path("/kaggle/input")
    if not base.exists():
        raise FileNotFoundError("/kaggle/input not found. This script is intended for Kaggle.")

    for candidate in base.rglob("all_tags.json"):
        root = candidate.parent
        if all((root / name).exists() for name in REQUIRED_FILES):
            return root

    raise FileNotFoundError(
        "Could not locate a dataset folder containing all required files: "
        + ", ".join(REQUIRED_FILES)
    )


def install_dependencies(python_exe: str) -> None:
    run_live(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "bitsandbytes",
            "trl",
            "sentencepiece",
        ],
        "Install dependencies",
    )


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
    parser = argparse.ArgumentParser(
        description="Run prepare -> train -> validate on Kaggle with robust defaults."
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/qwen_pipeline"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=24)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    python_exe = sys.executable
    dataset_root = find_dataset_root(args.dataset_root)

    print(f"[info] python: {python_exe}")
    print(f"[info] dataset_root: {dataset_root}")

    if not args.skip_install:
        install_dependencies(python_exe)

    work_dir = args.work_dir
    data_dir = work_dir / "data"
    train_out = work_dir / "train_output"
    eval_out = work_dir / "eval_output"
    tuned_eval_out = eval_out / "tuned"
    base_eval_out = eval_out / "base"

    data_dir.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(parents=True, exist_ok=True)
    tuned_eval_out.mkdir(parents=True, exist_ok=True)
    if not args.skip_base_eval:
        base_eval_out.mkdir(parents=True, exist_ok=True)

    prepare_script = dataset_root / "prepare_qwen_training_data.py"
    train_script = dataset_root / "train_qwen25_7b_qlora.py"
    validate_script = dataset_root / "validate_qwen_tagger.py"

    input_jsonl = dataset_root / "exercises_v2_gemini_intro_only.jsonl"
    tags_path = dataset_root / "all_tags.json"

    run_live(
        [
            python_exe,
            str(prepare_script),
            "--input-jsonl",
            str(input_jsonl),
            "--tags-path",
            str(tags_path),
            "--output-dir",
            str(data_dir),
            "--val-ratio",
            "0.1",
            "--seed",
            str(args.seed),
            "--min-thinking-chars",
            "40",
            "--max-thinking-chars",
            "320",
            "--drop-empty-tags",
            "--fix-exclusion-language",
            "--truncate-thinking",
        ],
        "Prepare dataset",
    )

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    run_live(
        [
            python_exe,
            str(train_script),
            "--train-file",
            str(train_file),
            "--val-file",
            str(val_file),
            "--base-model",
            args.base_model,
            "--output-dir",
            str(train_out),
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
            "--load-in-4bit",
            "--gradient-checkpointing",
        ],
        "Train model",
    )

    run_live(
        [
            python_exe,
            str(validate_script),
            "--val-file",
            str(val_file),
            "--tags-path",
            str(tags_path),
            "--base-model",
            args.base_model,
            "--adapter-path",
            str(train_out),
            "--output-dir",
            str(tuned_eval_out),
            "--max-new-tokens",
            "220",
            "--temperature",
            "0.0",
            "--top-p",
            "1.0",
            "--load-in-4bit",
        ],
        "Validate tuned model",
    )

    if not args.skip_base_eval:
        run_live(
            [
                python_exe,
                str(validate_script),
                "--val-file",
                str(val_file),
                "--tags-path",
                str(tags_path),
                "--base-model",
                args.base_model,
                "--output-dir",
                str(base_eval_out),
                "--max-new-tokens",
                "220",
                "--temperature",
                "0.0",
                "--top-p",
                "1.0",
                "--load-in-4bit",
            ],
            "Validate base model",
        )

        tuned_report = load_report(tuned_eval_out / "validation_report.json")
        base_report = load_report(base_eval_out / "validation_report.json")
        comparison = build_comparison(base_report=base_report, tuned_report=tuned_report)
        comparison_path = eval_out / "model_comparison.json"
        with comparison_path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        print("[done] model comparison finished")
        print(f"  comparison: {comparison_path}")

    print("\n[done] pipeline finished")
    print(f"  data_dir: {data_dir}")
    print(f"  train_output: {train_out}")
    print(f"  eval_output: {eval_out}")
    print(f"  tuned_eval_output: {tuned_eval_out}")
    if not args.skip_base_eval:
        print(f"  base_eval_output: {base_eval_out}")


if __name__ == "__main__":
    main()
