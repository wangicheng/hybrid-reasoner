import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


EXPECTED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
        description="Run local Qwen base vs LoRA tuned evaluation on the same validation set."
    )

    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))

    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(
            "data/experiments/slm_tag_dataset_gemini_intro_only/"
            "exercises_v2_gemini_intro_only.jsonl"
        ),
        help="Optional source dataset used when --val-file does not exist.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/experiments/qwen_test_prepared/val.jsonl"),
        help="Validation JSONL for testing. If missing, script can auto-prepare it.",
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=Path,
        default=Path("data/experiments/qwen_test_prepared"),
    )

    parser.add_argument(
        "--prepare-script",
        type=Path,
        default=Path("scripts/qwen_prepare_training_data_local.py"),
    )
    parser.add_argument(
        "--prepare-from-queries-script",
        type=Path,
        default=Path("scripts/qwen_prepare_validation_from_queries_local.py"),
    )
    parser.add_argument(
        "--queries-source",
        type=Path,
        default=Path("queries.json"),
        help="Primary existing experiment queries source.",
    )
    parser.add_argument(
        "--queries-source-alt",
        type=Path,
        default=Path("data/raw_repo_data/experiments/queries.json"),
        help="Alternative existing experiment queries source.",
    )
    parser.add_argument(
        "--validate-script",
        type=Path,
        default=Path("scripts/qwen_validate_tagger_local.py"),
    )

    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("data/models/qwen25_7b_tag_lora"),
        help="Directory containing LoRA adapter files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/qwen_model_tests"),
    )

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-thinking-chars", type=int, default=40)
    parser.add_argument("--max-thinking-chars", type=int, default=320)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--allow-missing-adapter",
        action="store_true",
        help="If set, runs base-only when LoRA adapter files are missing.",
    )
    parser.add_argument(
        "--skip-data-prepare",
        action="store_true",
        help="If set, fails when --val-file is missing instead of preparing data.",
    )
    return parser.parse_args()


def ensure_validation_file(args: argparse.Namespace, python_exe: str) -> Path:
    if args.val_file.exists():
        return args.val_file

    if args.skip_data_prepare:
        raise FileNotFoundError(
            f"Validation file not found: {args.val_file}. "
            "Provide --val-file or remove --skip-data-prepare."
        )

    if not args.tags_path.exists():
        raise FileNotFoundError(f"Tags file not found: {args.tags_path}")

    args.prepared_data_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_script.exists() and args.input_jsonl.exists():
        cmd = [
            python_exe,
            str(args.prepare_script),
            "--input-jsonl",
            str(args.input_jsonl),
            "--tags-path",
            str(args.tags_path),
            "--output-dir",
            str(args.prepared_data_dir),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--min-thinking-chars",
            str(args.min_thinking_chars),
            "--max-thinking-chars",
            str(args.max_thinking_chars),
            "--drop-empty-tags",
            "--fix-exclusion-language",
            "--truncate-thinking",
        ]
        run_live(cmd, "Prepare validation data from exercises JSONL")
    elif args.prepare_from_queries_script.exists() and args.queries_source.exists():
        cmd = [
            python_exe,
            str(args.prepare_from_queries_script),
            "--queries-json",
            str(args.queries_source),
            "--tags-path",
            str(args.tags_path),
            "--output-dir",
            str(args.prepared_data_dir),
        ]
        run_live(cmd, "Prepare validation data from existing queries.json")
    elif args.prepare_from_queries_script.exists() and args.queries_source_alt.exists():
        cmd = [
            python_exe,
            str(args.prepare_from_queries_script),
            "--queries-json",
            str(args.queries_source_alt),
            "--tags-path",
            str(args.tags_path),
            "--output-dir",
            str(args.prepared_data_dir),
        ]
        run_live(cmd, "Prepare validation data from existing queries.json (alt)")
    else:
        raise FileNotFoundError(
            "Could not prepare validation data. Checked:\n"
            f"- input jsonl: {args.input_jsonl}\n"
            f"- queries source: {args.queries_source}\n"
            f"- queries source alt: {args.queries_source_alt}"
        )

    prepared_val = args.prepared_data_dir / "val.jsonl"
    if not prepared_val.exists():
        raise FileNotFoundError(f"Prepared validation file not found: {prepared_val}")

    return prepared_val


def build_validate_cmd(
    python_exe: str,
    validate_script: Path,
    val_file: Path,
    tags_path: Path,
    base_model: str,
    output_dir: Path,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_4bit: bool,
    use_bf16: bool,
    adapter_path: Path | None,
) -> list[str]:
    cmd = [
        python_exe,
        str(validate_script),
        "--val-file",
        str(val_file),
        "--tags-path",
        str(tags_path),
        "--base-model",
        base_model,
        "--output-dir",
        str(output_dir),
        "--max-samples",
        str(max_samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
    ]
    if use_4bit:
        cmd.append("--load-in-4bit")
    if use_bf16:
        cmd.append("--bf16")
    if adapter_path is not None:
        cmd.extend(["--adapter-path", str(adapter_path)])
    return cmd


def has_adapter_files(path: Path) -> bool:
    return all((path / name).exists() for name in EXPECTED_ADAPTER_FILES)


def resolve_adapter_path(adapter_path: Path) -> Path:
    if has_adapter_files(adapter_path):
        return adapter_path

    if not adapter_path.exists() or not adapter_path.is_dir():
        return adapter_path

    candidates = []
    for child in adapter_path.iterdir():
        if not child.is_dir() or not has_adapter_files(child):
            continue

        match = re.match(r"checkpoint-(\d+)$", child.name)
        step = int(match.group(1)) if match else -1
        candidates.append((step, child.stat().st_mtime, child))

    if not candidates:
        return adapter_path

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def pick_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    tags_path = (repo_root / args.tags_path).resolve() if not args.tags_path.is_absolute() else args.tags_path
    input_jsonl = (repo_root / args.input_jsonl).resolve() if not args.input_jsonl.is_absolute() else args.input_jsonl
    val_file = (repo_root / args.val_file).resolve() if not args.val_file.is_absolute() else args.val_file
    prepared_data_dir = (
        (repo_root / args.prepared_data_dir).resolve()
        if not args.prepared_data_dir.is_absolute()
        else args.prepared_data_dir
    )
    user_prepare_script = (
        (repo_root / args.prepare_script).resolve()
        if not args.prepare_script.is_absolute()
        else args.prepare_script
    )
    prepare_from_queries_script = (
        (repo_root / args.prepare_from_queries_script).resolve()
        if not args.prepare_from_queries_script.is_absolute()
        else args.prepare_from_queries_script
    )
    queries_source = (
        (repo_root / args.queries_source).resolve()
        if not args.queries_source.is_absolute()
        else args.queries_source
    )
    queries_source_alt = (
        (repo_root / args.queries_source_alt).resolve()
        if not args.queries_source_alt.is_absolute()
        else args.queries_source_alt
    )
    user_validate_script = (
        (repo_root / args.validate_script).resolve()
        if not args.validate_script.is_absolute()
        else args.validate_script
    )
    adapter_path = (
        (repo_root / args.adapter_path).resolve()
        if not args.adapter_path.is_absolute()
        else args.adapter_path
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
    )

    prepare_candidates = [
        user_prepare_script,
        (repo_root / "src/kaggle_qwen/prepare_qwen_training_data.py").resolve(),
        (repo_root / "scripts/qwen_prepare_training_data_local.py").resolve(),
    ]
    validate_candidates = [
        user_validate_script,
        (repo_root / "src/kaggle_qwen/validate_qwen_tagger.py").resolve(),
        (repo_root / "scripts/qwen_validate_tagger_local.py").resolve(),
    ]

    prepare_script = pick_existing_path(prepare_candidates)
    validate_script = pick_existing_path(validate_candidates)

    if validate_script is None:
        checked = "\n  - ".join(str(path) for path in validate_candidates)
        raise FileNotFoundError(f"Validate script not found. Checked:\n  - {checked}")
    if not tags_path.exists():
        raise FileNotFoundError(f"Tags file not found: {tags_path}")

    args.tags_path = tags_path
    args.input_jsonl = input_jsonl
    args.val_file = val_file
    args.prepared_data_dir = prepared_data_dir
    args.prepare_from_queries_script = prepare_from_queries_script
    args.queries_source = queries_source
    args.queries_source_alt = queries_source_alt
    if prepare_script is None:
        args.prepare_script = user_prepare_script
    else:
        args.prepare_script = prepare_script

    val_file = ensure_validation_file(args=args, python_exe=python_exe)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_out = output_dir / "base"
    tuned_out = output_dir / "tuned"
    base_out.mkdir(parents=True, exist_ok=True)
    tuned_out.mkdir(parents=True, exist_ok=True)

    use_4bit = not args.disable_4bit

    base_cmd = build_validate_cmd(
        python_exe=python_exe,
        validate_script=validate_script,
        val_file=val_file,
        tags_path=tags_path,
        base_model=args.base_model,
        output_dir=base_out,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_4bit=use_4bit,
        use_bf16=args.bf16,
        adapter_path=None,
    )
    run_live(base_cmd, "Validate base Qwen model")

    selected_adapter_path = resolve_adapter_path(adapter_path)
    if selected_adapter_path != adapter_path:
        print(f"\n[info] Auto-selected adapter checkpoint: {selected_adapter_path}")

    missing_adapter_files = [name for name in EXPECTED_ADAPTER_FILES if not (selected_adapter_path / name).exists()]

    if missing_adapter_files:
        adapter_path.mkdir(parents=True, exist_ok=True)
        if args.allow_missing_adapter:
            print("\n[warn] LoRA adapter files are missing; tuned evaluation skipped.")
            print(f"[warn] expected at: {selected_adapter_path}")
            print(f"[warn] missing files: {', '.join(missing_adapter_files)}")
            print(f"[done] base report: {base_out / 'validation_report.json'}")
            return
        raise FileNotFoundError(
            "LoRA adapter files missing. "
            f"Place them under {adapter_path} or its checkpoint-* subfolders. "
            f"Missing: {', '.join(missing_adapter_files)}"
        )

    tuned_cmd = build_validate_cmd(
        python_exe=python_exe,
        validate_script=validate_script,
        val_file=val_file,
        tags_path=tags_path,
        base_model=args.base_model,
        output_dir=tuned_out,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_4bit=use_4bit,
        use_bf16=args.bf16,
        adapter_path=selected_adapter_path,
    )
    run_live(tuned_cmd, "Validate Qwen + LoRA tuned model")

    base_report = load_json(base_out / "validation_report.json")
    tuned_report = load_json(tuned_out / "validation_report.json")
    comparison = build_comparison(base_report=base_report, tuned_report=tuned_report)
    comparison_path = output_dir / "model_comparison.json"
    dump_json(comparison_path, comparison)

    print("\n[done] Qwen base vs tuned evaluation complete")
    print(f"  val_file:    {val_file}")
    print(f"  base_report: {base_out / 'validation_report.json'}")
    print(f"  tuned_report:{tuned_out / 'validation_report.json'}")
    print(f"  comparison:  {comparison_path}")


if __name__ == "__main__":
    main()
