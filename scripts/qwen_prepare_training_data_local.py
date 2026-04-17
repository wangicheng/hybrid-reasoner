import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


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


def build_system_prompt() -> str:
    return (
        "You are a Traditional Chinese novel tag extractor. "
        "Only use provided intro and allowed tags. "
        "Output JSON with keys thinking and final_tags."
    )


def build_user_prompt(intro_text: str, allowed_tags: Sequence[str]) -> str:
    tag_block = ", ".join(allowed_tags)
    return (
        "Choose the best tags from candidate tags based on the intro.\n"
        "Rules:\n"
        "1. Use only candidate tags.\n"
        "2. If evidence is weak, choose fewer tags.\n"
        "3. Return JSON with fields thinking and final_tags.\n\n"
        f"Candidate tags: {tag_block}\n\n"
        "Output JSON format:\n"
        "{\n"
        "  \"thinking\": \"...\",\n"
        "  \"final_tags\": [\"tag1\", \"tag2\"]\n"
        "}\n\n"
        f"Intro:\n{intro_text}"
    )


def preprocess_rows(
    rows: Sequence[dict],
    all_tags: Sequence[str],
    min_thinking_chars: int,
    max_thinking_chars: int,
    drop_empty_tags: bool,
) -> Tuple[List[dict], dict]:
    allowed_set = set(all_tags)
    cleaned: List[dict] = []

    dropped = 0

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        book = row.get("book") or {}
        book_id = str(book.get("id", "")).strip()
        intro = str(row.get("input_intro", "")).strip()
        thinking = str(row.get("thinking", "")).strip()
        raw_tags = row.get("final_tags") or []

        if not intro:
            dropped += 1
            continue

        if not isinstance(raw_tags, list):
            dropped += 1
            continue

        tags, _removed_tag_count = normalize_tags(raw_tags, allowed_set)

        if drop_empty_tags and not tags:
            dropped += 1
            continue

        if thinking and len(thinking) > max_thinking_chars:
            thinking = thinking[:max_thinking_chars].rstrip()

        if thinking and len(thinking) < min_thinking_chars:
            dropped += 1
            continue

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(intro_text=intro, allowed_tags=all_tags)

        cleaned.append(
            {
                "id": sample_id,
                "book_id": book_id,
                "input_intro": intro,
                "gold_tags": tags,
                "thinking": thinking,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )

    report = {
        "input_rows": len(rows),
        "kept_rows": len(cleaned),
        "dropped_rows": dropped,
        "taxonomy_size": len(all_tags),
        "min_thinking_chars": min_thinking_chars,
        "max_thinking_chars": max_thinking_chars,
        "drop_empty_tags": drop_empty_tags,
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
    parser = argparse.ArgumentParser(description="Prepare cleaned JSONL data for Qwen validation/testing.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--tags-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


if __name__ == "__main__":
    main()
