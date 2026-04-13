from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_OUTPUT_DIR = Path("data/experiments/slm_tag_dataset")


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


def first_sentence(text: str, max_len: int = 88) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip("，。！？；：、 ") + "…"


def build_tag_frequency(books: Sequence[dict], allowed_tags: Sequence[str]) -> Counter:
    allowed = set(allowed_tags)
    freq: Counter = Counter()
    for book in books:
        raw_tags = book.get("tags") or []
        uniq = []
        seen = set()
        for t in raw_tags:
            if t in allowed and t not in seen:
                uniq.append(t)
                seen.add(t)
        freq.update(uniq)
    return freq


def build_confusion_map(books: Sequence[dict], allowed_tags: Sequence[str], top_k: int = 3) -> Dict[str, List[str]]:
    allowed = set(allowed_tags)
    pair_counts: Counter = Counter()

    for book in books:
        tags = []
        seen = set()
        for t in (book.get("tags") or []):
            if t in allowed and t not in seen:
                tags.append(t)
                seen.add(t)

        tags.sort()
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                pair_counts[(tags[i], tags[j])] += 1

    neighbors: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (a, b), cnt in pair_counts.items():
        neighbors[a].append((b, cnt))
        neighbors[b].append((a, cnt))

    confusion_map: Dict[str, List[str]] = {}
    for tag in allowed_tags:
        ranked = sorted(neighbors.get(tag, []), key=lambda x: x[1], reverse=True)
        confusion_map[tag] = [x[0] for x in ranked[:top_k]]
    return confusion_map


@dataclass
class StrategyDecision:
    difficulty: str
    strategy: str


def decide_strategy(tags: Sequence[str], freq: Counter) -> StrategyDecision:
    if not tags:
        return StrategyDecision(difficulty="hard", strategy="recall_reason_generate")

    rare_count = sum(1 for t in tags if freq.get(t, 0) < 180)
    total = len(tags)

    if total >= 6 or rare_count >= 2:
        return StrategyDecision(difficulty="hard", strategy="recall_reason_generate")
    if total >= 4 or rare_count >= 1:
        return StrategyDecision(difficulty="medium", strategy="recall_reason_generate")
    return StrategyDecision(difficulty="easy", strategy="direct_answer")


def build_long_thinking(
    intro_zh: str,
    selected_tags: Sequence[str],
    tag_descriptions: Dict[str, str],
    confusion_map: Dict[str, List[str]],
) -> str:
    summary = first_sentence(intro_zh, max_len=110)
    lines: List[str] = []

    lines.append(f"第一步，我先抽取主題語義。這段簡介的核心訊號是：{summary}。")
    lines.append("第二步，我按照標籤定義逐一對齊文本線索，避免只看單一詞彙而誤判。")

    for idx, tag in enumerate(selected_tags, start=1):
        desc = tag_descriptions.get(tag, "此標籤反映了文本中的核心題材與敘事重點。")
        confusions = [x for x in confusion_map.get(tag, []) if x not in selected_tags][:2]
        confusion_note = ""
        if confusions:
            confusion_note = f" 同時與「{'、'.join(confusions)}」做邊界比較，確認語義主軸仍以「{tag}」更貼近。"
        lines.append(f"第三步-{idx}，標籤「{tag}」成立。依據：{desc}{confusion_note}")

    if selected_tags:
        lines.append(
            "第四步，我做多標籤一致性檢查：這些標籤之間可以共同解釋劇情、角色關係與世界觀，沒有互相衝突。"
        )
        lines.append(
            f"第五步，輸出最終多標籤集合：{'、'.join(selected_tags)}。"
        )
    else:
        lines.append("第四步，未出現足夠明確的目標標籤證據，因此保持空標籤並標記為高難度樣本。")

    return "\n".join(lines)


def extract_allowed_and_oov_tags(raw_tags: Sequence[str], allowed_set: set) -> Tuple[List[str], List[str]]:
    allowed: List[str] = []
    oov: List[str] = []
    seen_allowed = set()
    seen_oov = set()

    for tag in raw_tags:
        if tag in allowed_set:
            if tag not in seen_allowed:
                allowed.append(tag)
                seen_allowed.add(tag)
        else:
            if tag not in seen_oov:
                oov.append(tag)
                seen_oov.add(tag)
    return allowed, oov


def make_exercise_record(
    book: dict,
    index: int,
    allowed_set: set,
    tag_descriptions: Dict[str, str],
    freq: Counter,
    confusion_map: Dict[str, List[str]],
    created_at: str,
) -> dict:
    raw_tags = book.get("tags") or []
    final_tags, oov_tags = extract_allowed_and_oov_tags(raw_tags, allowed_set)

    intro_raw = (book.get("intro") or "").strip()
    intro_zh = clean_intro_to_chinese(intro_raw)
    decision = decide_strategy(final_tags, freq)
    thinking = build_long_thinking(intro_zh, final_tags, tag_descriptions, confusion_map)

    return {
        "schema_version": "slm_tag_train_v1",
        "sample_type": "exercise",
        "sample_id": f"ex_{index:06d}",
        "book": {
            "id": book.get("id", ""),
            "source": book.get("source", ""),
            "name": book.get("name", ""),
            "author": book.get("author", ""),
            "url": book.get("url", ""),
        },
        "input_intro": intro_zh,
        "thinking": thinking,
        "final_tags": final_tags,
        "difficulty": decision.difficulty,
        "strategy": decision.strategy,
        "metadata": {
            "original_tags": raw_tags,
            "oov_tags": oov_tags,
            "intro_raw": intro_raw,
            "created_at": created_at,
        },
    }


def build_textbook_records(
    all_tags: Sequence[str],
    tag_descriptions: Dict[str, str],
    confusion_map: Dict[str, List[str]],
    created_at: str,
) -> List[dict]:
    rows: List[dict] = []
    idx = 1

    for tag in all_tags:
        definition = tag_descriptions.get(tag, "此標籤代表特定題材與敘事重點。")
        confusion = confusion_map.get(tag, [])[:2]

        main_lesson = (
            f"標籤教材：{tag}。定義：{definition}"
            f" 判斷時要確認文本的核心敘事是否穩定對齊，不可只因單句提及就直接套標。"
        )
        think_main = (
            f"先回想「{tag}」的正式定義，再找文本中的主題線索、角色互動與劇情推進方式。"
            "若線索只出現一次且不是主軸，則不應標記。"
        )
        rows.append(
            {
                "schema_version": "slm_tag_train_v1",
                "sample_type": "textbook",
                "sample_id": f"tb_{idx:06d}",
                "input_intro": main_lesson,
                "thinking": think_main,
                "final_tags": [tag],
                "difficulty": "easy",
                "strategy": "recall_reason_generate",
                "metadata": {
                    "created_at": created_at,
                    "lesson_kind": "definition",
                    "tag": tag,
                },
            }
        )
        idx += 1

        if confusion:
            confusion_text = "、".join(confusion)
            compare_lesson = (
                f"邊界辨析：主標籤「{tag}」，易混淆標籤「{confusion_text}」。"
                "請先列出主軸語義，再比較角色關係、衝突來源、敘事目標，最後只保留最貼合的標籤。"
            )
            think_compare = (
                f"我先用定義判斷「{tag}」是否為主軸，再逐一排除「{confusion_text}」等近鄰標籤。"
                "只要缺少核心條件，就不能勉強標註。"
            )
            rows.append(
                {
                    "schema_version": "slm_tag_train_v1",
                    "sample_type": "textbook",
                    "sample_id": f"tb_{idx:06d}",
                    "input_intro": compare_lesson,
                    "thinking": think_compare,
                    "final_tags": [tag],
                    "difficulty": "medium",
                    "strategy": "recall_reason_generate",
                    "metadata": {
                        "created_at": created_at,
                        "lesson_kind": "boundary",
                        "tag": tag,
                        "confusions": confusion,
                    },
                }
            )
            idx += 1

    return rows


def build_schema() -> dict:
    return {
        "title": "slm_tag_train_v1",
        "type": "object",
        "required": [
            "schema_version",
            "sample_type",
            "sample_id",
            "input_intro",
            "thinking",
            "final_tags",
            "difficulty",
            "strategy",
            "metadata",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": "slm_tag_train_v1"},
            "sample_type": {"type": "string", "enum": ["textbook", "exercise"]},
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
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "strategy": {
                "type": "string",
                "enum": ["direct_answer", "recall_reason_generate"],
            },
            "metadata": {"type": "object"},
        },
    }


def validate_strict_json(records: Sequence[dict], all_tags: set) -> None:
    required_fields = {
        "schema_version",
        "sample_type",
        "sample_id",
        "input_intro",
        "thinking",
        "final_tags",
        "difficulty",
        "strategy",
        "metadata",
    }

    for i, row in enumerate(records, start=1):
        missing = required_fields - set(row.keys())
        if missing:
            raise ValueError(f"record#{i} missing fields: {sorted(missing)}")

        if row["schema_version"] != "slm_tag_train_v1":
            raise ValueError(f"record#{i} invalid schema_version")

        if row["sample_type"] not in {"textbook", "exercise"}:
            raise ValueError(f"record#{i} invalid sample_type")

        if row["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError(f"record#{i} invalid difficulty")

        if row["strategy"] not in {"direct_answer", "recall_reason_generate"}:
            raise ValueError(f"record#{i} invalid strategy")

        tags = row["final_tags"]
        if not isinstance(tags, list):
            raise ValueError(f"record#{i} final_tags must be a list")

        for tag in tags:
            if tag not in all_tags:
                raise ValueError(f"record#{i} contains out-of-taxonomy tag: {tag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full fixed JSONL datasets for SLM tag extraction.")
    parser.add_argument("--books-path", type=Path, default=Path("data/books_crawled.json"))
    parser.add_argument("--tags-path", type=Path, default=Path("data/all_tags.json"))
    parser.add_argument("--tag-desc-path", type=Path, default=Path("data/tag_descriptions.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def assert_output_paths(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        joined = "\n".join(existing)
        raise FileExistsError(f"Output files already exist. Use --overwrite to replace:\n{joined}")


def main() -> None:
    args = parse_args()

    books = load_json(args.books_path)
    all_tags = load_json(args.tags_path)
    tag_descriptions = load_json(args.tag_desc_path)

    if not isinstance(books, list):
        raise ValueError("books_crawled.json must be a list")
    if not isinstance(all_tags, list):
        raise ValueError("all_tags.json must be a list")
    if not isinstance(tag_descriptions, dict):
        raise ValueError("tag_descriptions.json must be an object")

    created_at = datetime.now(timezone.utc).isoformat()
    allowed_set = set(all_tags)
    freq = build_tag_frequency(books, all_tags)
    confusion_map = build_confusion_map(books, all_tags, top_k=3)

    exercise_rows = [
        make_exercise_record(
            book=book,
            index=i,
            allowed_set=allowed_set,
            tag_descriptions=tag_descriptions,
            freq=freq,
            confusion_map=confusion_map,
            created_at=created_at,
        )
        for i, book in enumerate(books, start=1)
    ]

    textbook_rows = build_textbook_records(
        all_tags=all_tags,
        tag_descriptions=tag_descriptions,
        confusion_map=confusion_map,
        created_at=created_at,
    )

    validate_strict_json(exercise_rows, allowed_set)
    validate_strict_json(textbook_rows, allowed_set)

    out_dir = args.output_dir
    schema_path = out_dir / "schema_v1.json"
    textbook_path = out_dir / "textbook_v1.jsonl"
    exercises_path = out_dir / "exercises_v1.jsonl"
    manifest_path = out_dir / "manifest_v1.json"

    assert_output_paths(
        [schema_path, textbook_path, exercises_path, manifest_path],
        overwrite=args.overwrite,
    )

    schema = build_schema()
    textbook_count = save_jsonl(textbook_path, textbook_rows)
    exercise_count = save_jsonl(exercises_path, exercise_rows)
    save_json(schema_path, schema)

    oov_count = 0
    no_tag_count = 0
    for row in exercise_rows:
        if row["metadata"]["oov_tags"]:
            oov_count += 1
        if not row["final_tags"]:
            no_tag_count += 1

    manifest = {
        "schema_version": "slm_tag_train_v1",
        "generated_at": created_at,
        "source": {
            "books_path": str(args.books_path),
            "tags_path": str(args.tags_path),
            "tag_desc_path": str(args.tag_desc_path),
            "total_books": len(books),
            "total_allowed_tags": len(all_tags),
        },
        "output": {
            "schema_path": str(schema_path),
            "textbook_path": str(textbook_path),
            "exercises_path": str(exercises_path),
            "textbook_count": textbook_count,
            "exercise_count": exercise_count,
        },
        "quality": {
            "records_with_oov_metadata": oov_count,
            "records_with_empty_final_tags": no_tag_count,
            "rule": "No source book removed; each input record maps to exactly one exercise sample.",
        },
    }
    save_json(manifest_path, manifest)

    print(f"[OK] textbook_v1.jsonl = {textbook_count}")
    print(f"[OK] exercises_v1.jsonl = {exercise_count}")
    print(f"[OK] schema_v1.json = {schema_path}")
    print(f"[OK] manifest_v1.json = {manifest_path}")


if __name__ == "__main__":
    main()
