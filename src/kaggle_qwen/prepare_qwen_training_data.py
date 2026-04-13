import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

FORBIDDEN_EXCLUSION_KEYWORDS = [
    "本文未提及",
    "未提及",
    "沒有提及",
    "未出現",
    "不包含",
    "不選",
    "不選取",
    "排除",
    "不納入",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_tags(tags: Sequence[str], allowed_set: set[str]) -> Tuple[List[str], int]:
    out: List[str] = []
    seen = set()
    removed = 0

    for tag in tags:
        t = str(tag).strip()
        if not t:
            removed += 1
            continue
        if t not in allowed_set:
            removed += 1
            continue
        if t in seen:
            removed += 1
            continue
        seen.add(t)
        out.append(t)

    return out, removed


def remove_exclusion_sentences(text: str) -> Tuple[str, int]:
    if not text.strip():
        return text, 0

    segments = re.split(r"(?<=[。！？!?])\s+|\n+", text.strip())
    kept: List[str] = []
    removed_count = 0

    for segment in segments:
        s = segment.strip()
        if not s:
            continue
        if any(k in s for k in FORBIDDEN_EXCLUSION_KEYWORDS):
            removed_count += 1
            continue
        kept.append(s)

    if not kept:
        return "", removed_count

    cleaned = " ".join(kept)
    return cleaned, removed_count


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rstrip()
    sentence_break = max(clipped.rfind("。"), clipped.rfind("！"), clipped.rfind("？"))
    if sentence_break >= int(max_chars * 0.6):
        clipped = clipped[: sentence_break + 1]

    return clipped.strip()


def build_system_prompt() -> str:
    return (
        "你是繁體中文小說標籤提取器。"
        "你只能根據給定簡介與候選標籤做判斷，不得使用外部知識。"
        "輸出必須是 JSON，欄位為 thinking 與 final_tags。"
    )


def build_user_prompt(intro_text: str, allowed_tags: Sequence[str]) -> str:
    tag_block = "、".join(allowed_tags)
    return (
        "請根據以下簡介選出最符合的標籤。\\n"
        "硬性規則：\\n"
        "1. 只能從候選標籤中挑選。\\n"
        "2. 若證據不足可少標，不要亂標。\\n"
        "3. thinking 只說明你為何選這些標籤，禁止寫不選哪些標籤。\\n"
        "4. thinking 長度 40-320 字。\\n\\n"
        f"候選標籤：{tag_block}\\n\\n"
        "請輸出 JSON：\\n"
        "{\\n"
        "  \"thinking\": \"...\",\\n"
        "  \"final_tags\": [\"標籤1\", \"標籤2\"]\\n"
        "}\\n\\n"
        f"簡介：\\n{intro_text}"
    )


def build_assistant_json(thinking: str, tags: Sequence[str]) -> str:
    payload = {
        "thinking": thinking,
        "final_tags": list(tags),
    }
    return json.dumps(payload, ensure_ascii=False)


def preprocess_rows(
    rows: Sequence[dict],
    all_tags: Sequence[str],
    min_thinking_chars: int,
    max_thinking_chars: int,
    drop_empty_tags: bool,
    fix_exclusion_language: bool,
    truncate_thinking: bool,
) -> Tuple[List[dict], dict]:
    allowed_set = set(all_tags)
    cleaned: List[dict] = []

    drop_reasons: Counter = Counter()
    fix_counts: Counter = Counter()

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        book = row.get("book") or {}
        book_id = str(book.get("id", "")).strip()
        intro = str(row.get("input_intro", "")).strip()
        thinking = str(row.get("thinking", "")).strip()
        raw_tags = row.get("final_tags") or []

        if not intro:
            drop_reasons["empty_intro"] += 1
            continue

        if not isinstance(raw_tags, list):
            drop_reasons["final_tags_not_list"] += 1
            continue

        tags, removed_tag_count = normalize_tags(raw_tags, allowed_set)
        if removed_tag_count:
            fix_counts["removed_invalid_or_duplicate_tags"] += removed_tag_count

        if drop_empty_tags and not tags:
            drop_reasons["empty_tags_after_normalization"] += 1
            continue

        if fix_exclusion_language:
            cleaned_thinking, removed_sentences = remove_exclusion_sentences(thinking)
            if removed_sentences:
                fix_counts["removed_exclusion_sentences"] += removed_sentences
            thinking = cleaned_thinking

        if truncate_thinking:
            original_len = len(thinking)
            thinking = truncate_text(thinking, max_thinking_chars)
            if len(thinking) < original_len:
                fix_counts["truncated_thinking"] += 1

        thinking_len = len(thinking)
        if thinking_len < min_thinking_chars:
            drop_reasons["thinking_too_short_after_fix"] += 1
            continue

        if thinking_len > max_thinking_chars:
            drop_reasons["thinking_too_long_after_fix"] += 1
            continue

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(intro_text=intro, allowed_tags=all_tags)
        assistant_json = build_assistant_json(thinking=thinking, tags=tags)

        cleaned.append(
            {
                "id": sample_id,
                "book_id": book_id,
                "input_intro": intro,
                "gold_tags": tags,
                "thinking": thinking,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant": assistant_json,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_json},
                ],
            }
        )

    report = {
        "input_rows": len(rows),
        "kept_rows": len(cleaned),
        "dropped_rows": len(rows) - len(cleaned),
        "drop_reasons": dict(drop_reasons),
        "fix_counts": dict(fix_counts),
        "taxonomy_size": len(all_tags),
        "min_thinking_chars": min_thinking_chars,
        "max_thinking_chars": max_thinking_chars,
        "drop_empty_tags": drop_empty_tags,
        "fix_exclusion_language": fix_exclusion_language,
        "truncate_thinking": truncate_thinking,
    }

    return cleaned, report


def split_dataset(rows: Sequence[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    data = list(rows)
    rng = random.Random(seed)
    rng.shuffle(data)

    if len(data) <= 1:
        return data, []

    val_size = max(1, int(round(len(data) * val_ratio)))
    val_size = min(val_size, len(data) - 1)

    val_rows = data[:val_size]
    train_rows = data[val_size:]
    return train_rows, val_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned chat-format data for Qwen2.5-7B QLoRA training.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/experiments/slm_tag_dataset_gemini_intro_only/exercises_v2_gemini_intro_only.jsonl"),
    )
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/qwen_training"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-thinking-chars", type=int, default=40)
    parser.add_argument("--max-thinking-chars", type=int, default=320)
    parser.add_argument("--drop-empty-tags", action="store_true")
    parser.add_argument("--fix-exclusion-language", action="store_true")
    parser.add_argument("--truncate-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
    if not args.tags_path.exists():
        raise FileNotFoundError(f"Tags file not found: {args.tags_path}")
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")

    all_tags = load_json(args.tags_path)
    if not isinstance(all_tags, list) or not all(isinstance(t, str) for t in all_tags):
        raise ValueError("all_tags.json must be a JSON array of strings")

    rows = load_jsonl(args.input_jsonl)
    cleaned_rows, report = preprocess_rows(
        rows=rows,
        all_tags=all_tags,
        min_thinking_chars=args.min_thinking_chars,
        max_thinking_chars=args.max_thinking_chars,
        drop_empty_tags=args.drop_empty_tags,
        fix_exclusion_language=args.fix_exclusion_language,
        truncate_thinking=args.truncate_thinking,
    )

    train_rows, val_rows = split_dataset(cleaned_rows, val_ratio=args.val_ratio, seed=args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = out_dir / "cleaned_all.jsonl"
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    report_path = out_dir / "prep_report.json"

    write_jsonl(cleaned_path, cleaned_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    report.update(
        {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "seed": args.seed,
            "input_jsonl": str(args.input_jsonl),
            "tags_path": str(args.tags_path),
            "output_dir": str(out_dir),
            "outputs": {
                "cleaned_all": str(cleaned_path),
                "train": str(train_path),
                "val": str(val_path),
                "report": str(report_path),
            },
        }
    )

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[done] prepared dataset")
    print(f"  cleaned_all: {cleaned_path}")
    print(f"  train:       {train_path}")
    print(f"  val:         {val_path}")
    print(f"  report:      {report_path}")
    print(f"  kept_rows:   {report['kept_rows']} / {report['input_rows']}")


if __name__ == "__main__":
    main()
