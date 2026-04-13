from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from google import genai
from google.genai import types

from src.core.api_utils import _is_retryable, get_api_key_rotator, get_current_api_key, get_rate_limiter


DEFAULT_OUTPUT_DIR = Path("data/experiments/slm_tag_dataset_gemini_intro_only")
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_existing_book_ids(jsonl_path: Path) -> Tuple[set[str], int]:
    ids: set[str] = set()
    total = 0
    if not jsonl_path.exists():
        return ids, total

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            book_id = str(row.get("book", {}).get("id", "")).strip()
            if book_id:
                ids.add(book_id)
            total += 1
    return ids, total


def initialize_output_files(
    exercises_path: Path,
    failed_path: Path,
    overwrite: bool,
) -> None:
    """Prepare output files for either fresh overwrite or append/resume mode."""
    exercises_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        with exercises_path.open("w", encoding="utf-8") as f:
            f.write("")
        with failed_path.open("w", encoding="utf-8") as f:
            f.write("")
    else:
        if not exercises_path.exists():
            with exercises_path.open("w", encoding="utf-8") as f:
                f.write("")
        if not failed_path.exists():
            with failed_path.open("w", encoding="utf-8") as f:
                f.write("")


def write_run_checkpoint(
    checkpoint_path: Path,
    *,
    status: str,
    created_at: str,
    model: str,
    processed: int,
    total_books_seen: int,
    appended_count: int,
    skipped_existing: int,
    failed_count: int,
    existing_record_count: int,
    records_total_after_run: int,
    last_index: int,
    last_book_id: str,
) -> None:
    checkpoint = {
        "schema_version": "slm_tag_train_v2_gemini_intro_only",
        "status": status,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "progress": {
            "processed": processed,
            "total_books_seen": total_books_seen,
            "appended_count": appended_count,
            "skipped_existing": skipped_existing,
            "failed_count": failed_count,
            "existing_record_count": existing_record_count,
            "records_total_after_run": records_total_after_run,
            "last_index": last_index,
            "last_book_id": last_book_id,
        },
    }
    save_json(checkpoint_path, checkpoint)


def clean_intro_to_chinese(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()
    keep_punct = "，。！？；：、（）「」『』《》〈〉【】—…．·0123456789０１２３４５６７８９"
    kept_chars: List[str] = []

    for ch in text:
        code = ord(ch)
        is_cjk = 0x4E00 <= code <= 0x9FFF
        is_space = ch == " "
        is_allowed = ch in keep_punct
        if is_cjk or is_space or is_allowed:
            kept_chars.append(ch)

    cleaned = "".join(kept_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else text


def extract_json_object(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Empty model output")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text[:300]}")
    return json.loads(match.group(0))


def normalize_tags(tags: Sequence[str], allowed_set: set[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in tags:
        t = str(tag).strip()
        if t in allowed_set and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def is_reasoning_low_quality(reasoning: str, allowed_tags: Sequence[str]) -> bool:
    """
    Detect low-quality reasoning patterns such as listing too many absent tags.
    """
    text = (reasoning or "").strip()
    if not text:
        return True

    mentioned_count = sum(1 for tag in allowed_tags if tag in text)
    forbidden_exclusion_keywords = [
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
    has_exclusion_language = any(k in text for k in forbidden_exclusion_keywords)

    # New policy: do not explain non-selected tags at all.
    if has_exclusion_language:
        return True

    # Keep reasoning concise and focused on selected evidence.
    if len(text) > 320:
        return True

    # Prevent listing too many tag names in a single reasoning block.
    if mentioned_count > 6:
        return True

    return False


def build_prompt(intro_text: str, allowed_tags: Sequence[str]) -> str:
    tag_block = "、".join(allowed_tags)
    return f"""
你是標籤提取器。你只能使用下方提供的「簡介文本」進行推理。

硬性規則：
1. 禁止使用任何外部知識。
2. 禁止使用書名、作者、網址、原始標籤或任何你沒有在簡介中看到的資訊。
3. 只能從候選標籤中選擇最符合的標籤。
4. 如果證據不足，寧可少標，不要亂標。
5. thinking 必須是逐步推理，至少 40 字，且只說明「你為何選了這些標籤」。
6. 禁止任何「不選哪些標籤」或「本文未提及哪些標籤」的敘述。
7. thinking 請聚焦文本證據，建議 2-5 句，避免冗長模板化廢話。

候選標籤（只能從這裡選）：
{tag_block}

請輸出嚴格 JSON 物件，且只能輸出 JSON，不要任何額外文字：
{{
  "thinking": "...",
  "final_tags": ["標籤1", "標籤2"]
}}

簡介文本：
{intro_text}
""".strip()


class GeminiIntroOnlyExtractor:
    def __init__(self, model: str, temperature: float, max_retries: int):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = self._new_client()

    @staticmethod
    def _new_client() -> genai.Client:
        key = get_current_api_key()
        return genai.Client(api_key=key)

    def _rotate_key_if_possible(self) -> None:
        try:
            rotator = get_api_key_rotator()
            rotator.on_rate_limit_error()
            self.client = self._new_client()
        except Exception:
            # Single key mode or no rotator available.
            pass

    def infer(self, prompt: str) -> str:
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                get_rate_limiter().wait()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                    ),
                )

                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Model returned empty text response")
                return text

            except Exception as exc:
                last_exc = exc
                retryable = _is_retryable(exc)
                if not retryable or attempt >= self.max_retries:
                    raise

                self._rotate_key_if_possible()
                time.sleep(min(5.0 * (attempt + 1), 30.0))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected infer loop termination")


def validate_row(row: dict, allowed_set: set[str]) -> None:
    required = {
        "schema_version",
        "sample_type",
        "sample_id",
        "book",
        "input_intro",
        "thinking",
        "final_tags",
        "metadata",
    }
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"Missing row fields: {sorted(missing)}")

    if row["schema_version"] != "slm_tag_train_v2_gemini_intro_only":
        raise ValueError("Invalid schema_version")

    if row["sample_type"] != "exercise_gemini_intro_only":
        raise ValueError("Invalid sample_type")

    if not isinstance(row["thinking"], str) or len(row["thinking"].strip()) < 40:
        raise ValueError("thinking is too short")

    if len(row["thinking"].strip()) > 320:
        raise ValueError("thinking is too long")

    forbidden_exclusion_keywords = [
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
    if any(k in row["thinking"] for k in forbidden_exclusion_keywords):
        raise ValueError("thinking contains forbidden exclusion language")

    tags = row["final_tags"]
    if not isinstance(tags, list):
        raise ValueError("final_tags must be a list")
    for t in tags:
        if t not in allowed_set:
            raise ValueError(f"Tag out of taxonomy: {t}")


def build_schema() -> dict:
    return {
        "title": "slm_tag_train_v2_gemini_intro_only",
        "type": "object",
        "required": [
            "schema_version",
            "sample_type",
            "sample_id",
            "book",
            "input_intro",
            "thinking",
            "final_tags",
            "metadata",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": "slm_tag_train_v2_gemini_intro_only"},
            "sample_type": {"type": "string", "const": "exercise_gemini_intro_only"},
            "sample_id": {"type": "string"},
            "book": {
                "type": "object",
                "required": ["id", "source", "name", "author", "url"],
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "name": {"type": "string"},
                    "author": {"type": "string"},
                    "url": {"type": "string"},
                },
            },
            "input_intro": {"type": "string"},
            "thinking": {"type": "string"},
            "final_tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "metadata": {"type": "object"},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate independent Gemini intro-only tag extraction dataset."
    )
    parser.add_argument("--books-path", type=Path, default=Path("data/books_crawled.json"))
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-books", type=int, default=0, help="0 means all books.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Gemini; produce empty placeholder tags.")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip startup API availability check (not recommended).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def resolve_manifest_path(base_manifest_path: Path, overwrite: bool) -> Path:
    if overwrite or not base_manifest_path.exists():
        return base_manifest_path

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base_manifest_path.with_name(f"{base_manifest_path.stem}_{ts}{base_manifest_path.suffix}")


def preflight_check_api(model: str, temperature: float) -> None:
    """
    Verify API availability before generation starts.
    Tries all configured API keys and leaves rotator on the first usable key.
    """
    rotator = get_api_key_rotator()
    keys = getattr(rotator, "api_keys", None)
    key_count = len(keys) if isinstance(keys, list) and keys else 1
    last_exc: Exception | None = None

    for _ in range(key_count):
        key = rotator.get_current_key()
        client = genai.Client(api_key=key)
        try:
            get_rate_limiter().wait()
            response = client.models.generate_content(
                model=model,
                contents="請只回覆 OK",
                config=types.GenerateContentConfig(
                    temperature=min(max(temperature, 0.0), 0.2),
                    response_mime_type="text/plain",
                ),
            )
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                raise ValueError("Empty response during API preflight check")
            return
        except Exception as exc:
            last_exc = exc
            try:
                rotator.rotate()
            except Exception:
                # If rotation is unavailable, continue to fail at the end.
                pass

    raise RuntimeError(
        "API preflight check failed for all configured keys. "
        f"Last error: {last_exc}"
    )


def main() -> None:
    args = parse_args()

    books = load_json(args.books_path)
    all_tags = load_json(args.tags_path)
    if not isinstance(books, list):
        raise ValueError("books_crawled.json must be a list")
    if not isinstance(all_tags, list):
        raise ValueError("all_tags.json must be a list")

    if args.max_books > 0:
        books = books[: args.max_books]

    created_at = datetime.now(timezone.utc).isoformat()
    allowed_set = set(all_tags)

    out_dir = args.output_dir
    exercises_path = out_dir / "exercises_v2_gemini_intro_only.jsonl"
    failed_path = out_dir / "failed_v2_gemini_intro_only.jsonl"
    checkpoint_path = out_dir / "run_state_v2_gemini_intro_only.json"
    schema_path = out_dir / "schema_v2_gemini_intro_only.json"
    manifest_path = resolve_manifest_path(out_dir / "manifest_v2_gemini_intro_only.json", args.overwrite)

    initialize_output_files(exercises_path=exercises_path, failed_path=failed_path, overwrite=args.overwrite)

    existing_book_ids: set[str] = set()
    existing_record_count = 0
    if not args.overwrite:
        existing_book_ids, existing_record_count = load_existing_book_ids(exercises_path)

    if not args.dry_run and not args.skip_preflight:
        print("[preflight] checking Gemini API availability...")
        preflight_check_api(model=args.model, temperature=args.temperature)
        print("[preflight] API is available. Starting/resuming generation now.")

    extractor = None if args.dry_run else GeminiIntroOnlyExtractor(
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

    failed_preview: List[dict] = []
    skipped_existing = 0
    appended_count = 0
    failed_count = 0
    processed_count = 0
    last_index = 0
    last_book_id = ""

    # Always keep schema available from startup, even if interrupted later.
    save_json(schema_path, build_schema())

    write_run_checkpoint(
        checkpoint_path,
        status="running",
        created_at=created_at,
        model=args.model,
        processed=processed_count,
        total_books_seen=len(books),
        appended_count=appended_count,
        skipped_existing=skipped_existing,
        failed_count=failed_count,
        existing_record_count=existing_record_count,
        records_total_after_run=existing_record_count + appended_count,
        last_index=last_index,
        last_book_id=last_book_id,
    )

    for idx, book in enumerate(books, start=1):
        book_id = str(book.get("id", "")).strip()
        last_index = idx
        last_book_id = book_id
        if book_id and book_id in existing_book_ids:
            skipped_existing += 1
            processed_count += 1
            if processed_count % 25 == 0:
                write_run_checkpoint(
                    checkpoint_path,
                    status="running",
                    created_at=created_at,
                    model=args.model,
                    processed=processed_count,
                    total_books_seen=len(books),
                    appended_count=appended_count,
                    skipped_existing=skipped_existing,
                    failed_count=failed_count,
                    existing_record_count=existing_record_count,
                    records_total_after_run=existing_record_count + appended_count,
                    last_index=last_index,
                    last_book_id=last_book_id,
                )
            continue

        intro_raw = (book.get("intro") or "").strip()
        intro_zh = clean_intro_to_chinese(intro_raw)

        if not intro_zh:
            fail_row = {
                "book_id": book.get("id", ""),
                "index": idx,
                "reason": "empty_intro",
            }
            append_jsonl_row(failed_path, fail_row)
            failed_count += 1
            if len(failed_preview) < 30:
                failed_preview.append(fail_row)
            processed_count += 1
            continue

        try:
            if args.dry_run:
                model_output = {
                    "thinking": "dry_run 模式：未呼叫 Gemini，此欄位僅用於本地流程驗證，正式資料會由模型生成完整推理。",
                    "final_tags": [],
                }
            else:
                prompt = build_prompt(intro_text=intro_zh, allowed_tags=all_tags)
                raw = extractor.infer(prompt)
                parsed = extract_json_object(raw)
                model_output = {
                    "thinking": str(parsed.get("thinking", "")).strip(),
                    "final_tags": normalize_tags(parsed.get("final_tags", []), allowed_set),
                }

                # One quality-correction retry if reasoning is boilerplate/laundry-list style.
                if is_reasoning_low_quality(model_output["thinking"], all_tags):
                    repair_prompt = f"""
你上一版推理不合格。

請重新輸出 JSON，並嚴格遵守：
1) reasoning 至少 40 字，最多 320 字。
2) 只能說明你選中標籤的正向證據。
3) 禁止任何不選標籤、排除標籤、未提及標籤的描述。
4) 聚焦簡介中的直接證據，避免模板化廢話。

候選標籤（只能從這裡選）：
{'、'.join(all_tags)}

輸出格式（只能 JSON）：
{{
  "thinking": "...",
  "final_tags": ["標籤1", "標籤2"]
}}

簡介文本：
{intro_zh}
""".strip()
                    raw_retry = extractor.infer(repair_prompt)
                    parsed_retry = extract_json_object(raw_retry)
                    model_output = {
                        "thinking": str(parsed_retry.get("thinking", "")).strip(),
                        "final_tags": normalize_tags(parsed_retry.get("final_tags", []), allowed_set),
                    }

            row = {
                "schema_version": "slm_tag_train_v2_gemini_intro_only",
                "sample_type": "exercise_gemini_intro_only",
                "sample_id": f"gio_{idx:06d}",
                "book": {
                    "id": book.get("id", ""),
                    "source": book.get("source", ""),
                    "name": book.get("name", ""),
                    "author": book.get("author", ""),
                    "url": book.get("url", ""),
                },
                "input_intro": intro_zh,
                "thinking": model_output["thinking"],
                "final_tags": model_output["final_tags"],
                "metadata": {
                    "created_at": created_at,
                    "model": args.model,
                    "intro_only_inference": True,
                    "dry_run": args.dry_run,
                    "source_tags_for_eval": book.get("tags") or [],
                    "intro_raw": intro_raw,
                },
            }

            validate_row(row, allowed_set)
            append_jsonl_row(exercises_path, row)
            appended_count += 1
            processed_count += 1
            if book_id:
                existing_book_ids.add(book_id)

            if idx % 50 == 0:
                print(
                    f"[progress] processed={idx}/{len(books)} new_valid={appended_count} "
                    f"failed={failed_count} skipped_existing={skipped_existing}"
                )

            if processed_count % 25 == 0:
                write_run_checkpoint(
                    checkpoint_path,
                    status="running",
                    created_at=created_at,
                    model=args.model,
                    processed=processed_count,
                    total_books_seen=len(books),
                    appended_count=appended_count,
                    skipped_existing=skipped_existing,
                    failed_count=failed_count,
                    existing_record_count=existing_record_count,
                    records_total_after_run=existing_record_count + appended_count,
                    last_index=last_index,
                    last_book_id=last_book_id,
                )

        except Exception as exc:
            fail_row = {
                "book_id": book.get("id", ""),
                "index": idx,
                "reason": str(exc),
            }
            append_jsonl_row(failed_path, fail_row)
            failed_count += 1
            if len(failed_preview) < 30:
                failed_preview.append(fail_row)
            processed_count += 1

    total_after = existing_record_count + appended_count

    write_run_checkpoint(
        checkpoint_path,
        status="completed",
        created_at=created_at,
        model=args.model,
        processed=processed_count,
        total_books_seen=len(books),
        appended_count=appended_count,
        skipped_existing=skipped_existing,
        failed_count=failed_count,
        existing_record_count=existing_record_count,
        records_total_after_run=total_after,
        last_index=last_index,
        last_book_id=last_book_id,
    )

    manifest = {
        "schema_version": "slm_tag_train_v2_gemini_intro_only",
        "generated_at": created_at,
        "source": {
            "books_path": str(args.books_path),
            "tags_path": str(args.tags_path),
            "total_books_seen": len(books),
            "allowed_tag_count": len(all_tags),
        },
        "inference": {
            "model": args.model,
            "temperature": args.temperature,
            "max_retries": args.max_retries,
            "intro_only": True,
            "dry_run": args.dry_run,
            "preflight_checked": (not args.dry_run and not args.skip_preflight),
        },
        "output": {
            "schema_path": str(schema_path),
            "exercises_path": str(exercises_path),
            "failed_path": str(failed_path),
            "checkpoint_path": str(checkpoint_path),
            "existing_records_before_run": existing_record_count,
            "records_appended": appended_count,
            "records_total_after_run": total_after,
            "records_skipped_existing": skipped_existing,
            "records_failed": failed_count,
        },
        "failed_examples": failed_preview,
        "rule": (
            "Gemini input includes intro text and label taxonomy only; no title/author/url/original tags are provided to the model. "
            "Without --overwrite, this script appends only new records and keeps existing data unchanged. "
            "Each generated row is written immediately, so interruption can resume from previously saved records."
        ),
    }
    save_json(manifest_path, manifest)

    print(f"[OK] exercises_v2_gemini_intro_only.jsonl appended = {appended_count}")
    print(f"[OK] exercises_v2_gemini_intro_only.jsonl total = {total_after}")
    print(f"[OK] skipped existing = {skipped_existing}")
    print(f"[OK] schema_v2_gemini_intro_only.json = {schema_path}")
    print(f"[OK] manifest_v2_gemini_intro_only.json = {manifest_path}")
    print(f"[OK] failed records = {len(failed)}")


if __name__ == "__main__":
    main()
