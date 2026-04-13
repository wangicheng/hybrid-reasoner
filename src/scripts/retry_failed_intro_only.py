import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry generation only for book_ids listed in failed_v2_gemini_intro_only.jsonl"
    )
    parser.add_argument("--books-path", type=Path, default=Path("data/books_crawled.json"))
    parser.add_argument(
        "--failed-path",
        type=Path,
        default=Path("data/experiments/slm_tag_dataset_gemini_intro_only/failed_v2_gemini_intro_only.jsonl"),
    )
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/slm_tag_dataset_gemini_intro_only"),
    )

    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-books", type=int, default=0, help="0 means all failed book_ids")
    parser.add_argument(
        "--reason-pattern",
        type=str,
        default="",
        help="Optional regex filter on failed reason (example: '429|No JSON object').",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.books_path.exists():
        raise FileNotFoundError(f"books file not found: {args.books_path}")
    if not args.failed_path.exists():
        raise FileNotFoundError(f"failed file not found: {args.failed_path}")
    if not args.tags_path.exists():
        raise FileNotFoundError(f"tags file not found: {args.tags_path}")

    books = load_json(args.books_path)
    if not isinstance(books, list):
        raise ValueError("books_crawled.json must be a list")

    failed_rows = load_jsonl(args.failed_path)

    reason_re = re.compile(args.reason_pattern) if args.reason_pattern else None

    book_map: Dict[str, dict] = {}
    for b in books:
        book_id = str((b or {}).get("id", "")).strip()
        if book_id:
            book_map[book_id] = b

    retry_ids: List[str] = []
    seen = set()

    for row in failed_rows:
        book_id = str((row or {}).get("book_id", "")).strip()
        if not book_id or book_id in seen:
            continue

        reason = str((row or {}).get("reason", ""))
        if reason_re and not reason_re.search(reason):
            continue

        seen.add(book_id)
        retry_ids.append(book_id)

    if args.max_books > 0:
        retry_ids = retry_ids[: args.max_books]

    retry_books = [book_map[bid] for bid in retry_ids if bid in book_map]
    missing_count = sum(1 for bid in retry_ids if bid not in book_map)

    if not retry_books:
        print("[info] no failed book_ids matched the filter; nothing to retry")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_books_path = args.output_dir / "_tmp_retry_failed_books.json"
    with temp_books_path.open("w", encoding="utf-8") as f:
        json.dump(retry_books, f, ensure_ascii=False, indent=2)

    print(f"[info] retry candidates in failed file: {len(retry_ids)}")
    print(f"[info] retry books found in books file: {len(retry_books)}")
    if missing_count:
        print(f"[warn] missing from books file: {missing_count}")

    cmd = [
        sys.executable,
        "-m",
        "src.scripts.generate_gemini_intro_only_dataset",
        "--books-path",
        str(temp_books_path),
        "--tags-path",
        str(args.tags_path),
        "--output-dir",
        str(args.output_dir),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--max-retries",
        str(args.max_retries),
    ]

    if args.skip_preflight:
        cmd.append("--skip-preflight")
    if args.dry_run:
        cmd.append("--dry-run")

    print("[run]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        if temp_books_path.exists() and not args.keep_temp:
            temp_books_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
