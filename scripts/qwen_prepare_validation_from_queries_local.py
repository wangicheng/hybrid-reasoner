import argparse
import json
from pathlib import Path
from typing import Iterable, List


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-16") as f:
            return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Qwen-compatible validation JSONL from existing experiment queries.json"
    )
    parser.add_argument("--queries-json", type=Path, required=True)
    parser.add_argument("--tags-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_system_prompt() -> str:
    return (
        "你是繁體中文小說標籤提取器。"
        "你只能根據給定文字與候選標籤做判斷，不得使用外部知識。"
        "輸出必須是 JSON，欄位為 thinking 與 final_tags。"
    )


def build_user_prompt(query_text: str, allowed_tags: List[str]) -> str:
    tag_block = "、".join(allowed_tags)
    return (
        "請根據以下需求文字選出最符合的標籤。\n"
        "硬性規則：\n"
        "1. 只能從候選標籤中挑選。\n"
        "2. 若證據不足可少標，不要亂標。\n"
        "3. 輸出格式必須是 JSON。\n\n"
        f"候選標籤：{tag_block}\n\n"
        "請輸出 JSON：\n"
        "{\n"
        "  \"thinking\": \"...\",\n"
        "  \"final_tags\": [\"標籤1\", \"標籤2\"]\n"
        "}\n\n"
        f"需求文字：\n{query_text}"
    )


def normalize_tags(tags: List[str], allowed_set: set[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for tag in tags:
        t = str(tag).strip()
        if not t or t not in allowed_set or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def main() -> None:
    args = parse_args()

    if not args.queries_json.exists():
        raise FileNotFoundError(f"queries json not found: {args.queries_json}")
    if not args.tags_path.exists():
        raise FileNotFoundError(f"tags file not found: {args.tags_path}")

    queries = load_json(args.queries_json)
    all_tags = load_json(args.tags_path)

    if not isinstance(queries, list):
        raise ValueError("queries json must be a JSON array")
    if not isinstance(all_tags, list) or not all(isinstance(t, str) for t in all_tags):
        raise ValueError("tags file must be a JSON array of strings")

    allowed_set = set(all_tags)
    system_prompt = build_system_prompt()

    val_rows: List[dict] = []

    for idx, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            continue

        query_text = str(item.get("query", "")).strip()
        if not query_text:
            continue

        sample_id = str(item.get("id", f"q{idx}")).strip() or f"q{idx}"
        golden_rules = item.get("golden_rules") or {}
        required_tags = golden_rules.get("required_tags") or []
        gold_tags = normalize_tags(required_tags, allowed_set)

        user_prompt = build_user_prompt(query_text=query_text, allowed_tags=all_tags)

        val_rows.append(
            {
                "id": sample_id,
                "book_id": sample_id,
                "input_intro": query_text,
                "gold_tags": gold_tags,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )

    if not val_rows:
        raise ValueError("No valid rows generated from queries json")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = out_dir / "val.jsonl"
    report_path = out_dir / "prep_report.json"

    write_jsonl(val_path, val_rows)

    report = {
        "source": str(args.queries_json),
        "tags_path": str(args.tags_path),
        "total_rows": len(val_rows),
        "notes": "Validation set generated from existing queries.json required_tags.",
        "outputs": {
            "val": str(val_path),
            "report": str(report_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[done] prepared validation set from queries.json")
    print(f"  val:    {val_path}")
    print(f"  report: {report_path}")
    print(f"  rows:   {len(val_rows)}")


if __name__ == "__main__":
    main()
