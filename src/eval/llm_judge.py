import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from src.eval.paths import resolve_annotation_path
from src.eval.pool_data import build_annotation_rows, load_experiment_pool


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

JUDGE_MODELS = ["gemma-4-31b-it", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]
ANNOTATION_COLUMNS = [
    "Query ID",
    "Query",
    "Book ID",
    "Title",
    "Author",
    "Words (萬)",
    "Status",
    "Tags",
    "Intro",
    "Score (0-3)",
    "Comment",
]

SYSTEM_PROMPT = """\
You are an expert relevance assessor for a web novel recommendation system.
Your task is to judge how well a recommended book matches a user's search query.

### Scoring Rubric (0-3 scale):
- Score 0: The book is completely unrelated to the user's core intent.
- Score 1: The book shares only a superficial or tangential connection to the query.
- Score 2: The book satisfies some key requirements but misses important aspects.
- Score 3: The book is an excellent match for the user's core intent.

### Important Guidelines:
1. Focus on the user's intent, not just keyword overlap.
2. Ignore hard constraints like status or word count. Only judge semantic and genre relevance.
3. Books with empty or missing information should be scored 0.
4. Provide brief reasoning in Traditional Chinese.

### Output Format:
Return a JSON object with exactly two fields:
{
  "reasoning": "<brief explanation in Traditional Chinese>",
  "score": <integer 0-3>
}
"""


def _load_api_utils():
    from src.core.api_utils import (
        _is_retryable,
        get_api_key_rotator,
        get_current_api_key,
        get_rate_limiter,
        is_rate_limit_error,
    )

    return _is_retryable, get_api_key_rotator, get_current_api_key, get_rate_limiter, is_rate_limit_error


def build_user_prompt(query: str, title: str, tags: str, intro: str) -> str:
    return f"""\
### User Query:
{query}

### Recommended Book:
- Title: {title}
- Tags: {tags}
- Intro: {intro}

Please evaluate the relevance of this book to the user's query and return your judgment as JSON.
"""


class LLMJudge:
    def __init__(self, model_id: Optional[str] = None):
        if genai is None or types is None:
            raise ImportError("google-genai is required to run llm_judge.py")

        _, _, get_current_api_key, _, _ = _load_api_utils()
        api_key = get_current_api_key()
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_id = model_id or JUDGE_MODELS[0]
        self.models_to_try = [self.model_id]
        self.response_schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["reasoning", "score"],
        }

    @staticmethod
    def _retry_delay_seconds(
        attempt: int,
        base_delay: float = 5.0,
        max_delay: float = 120.0,
    ) -> float:
        _ = attempt, max_delay
        return base_delay

    def _rotate_api_key(self) -> None:
        _, get_api_key_rotator, _, _, _ = _load_api_utils()
        rotator = get_api_key_rotator()
        new_key = rotator.on_rate_limit_error()
        self.client = genai.Client(api_key=new_key)
        print(f"  [judge] API key rotated. Current index: {rotator.current_index}")

    def judge_single(self, query: str, title: str, tags: str, intro: str) -> Dict[str, Any]:
        if not title or title == "Unknown" or not title.strip():
            return {"score": 0, "reasoning": "作品資訊不足，無法判定與需求相關。"}
        if not intro or not intro.strip():
            return {"score": 0, "reasoning": "缺少作品簡介，無法判定與需求相關。"}

        user_prompt = build_user_prompt(query, title, tags, intro)
        last_exception = None

        for model_id in self.models_to_try:
            try:
                return self._call_llm(model_id, user_prompt)
            except Exception as exc:
                last_exception = exc
                print(f"  [judge] Model {model_id} failed: {exc}")

        return {"score": 0, "reasoning": f"LLM judge failed: {last_exception}"}

    def _call_llm(self, model_id: str, user_prompt: str) -> Dict[str, Any]:
        attempt = 0
        _is_retryable, _, _, get_rate_limiter, is_rate_limit_error = _load_api_utils()

        while True:
            try:
                get_rate_limiter().wait()
                is_gemma = "gemma" in model_id.lower()

                if is_gemma:
                    config_args: Dict[str, Any] = {}
                    contents = (
                        f"{SYSTEM_PROMPT}\n\n"
                        f"{user_prompt}\n\n"
                        "IMPORTANT: Output only valid JSON with keys 'reasoning' and 'score'."
                    )
                else:
                    config_args = {
                        "response_mime_type": "application/json",
                        "response_schema": self.response_schema,
                        "system_instruction": SYSTEM_PROMPT,
                    }
                    contents = user_prompt

                if is_gemma:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=contents,
                    )
                else:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=self.types.GenerateContentConfig(**config_args),
                    )

                if not response.text:
                    raise ValueError("Empty response from LLM")

                raw_text = response.text.strip()
                raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```\s*$", "", raw_text)
                parsed = json.loads(raw_text.strip())

                score = int(parsed.get("score", 0))
                score = max(0, min(3, score))
                return {
                    "score": score,
                    "reasoning": str(parsed.get("reasoning", "")),
                }
            except Exception as exc:
                if not _is_retryable(exc):
                    raise

                attempt += 1
                error_text = str(exc)
                if is_rate_limit_error(exc):
                    self._rotate_api_key()

                delay = self._retry_delay_seconds(attempt)
                print(
                    f"  [judge] Model {model_id} hit a retryable error: {exc}. "
                    f"Retrying in {delay:.1f}s (attempt {attempt})..."
                )
                time.sleep(delay)


def _make_task_key(row: Dict[str, Any]) -> str:
    query_id = str(row.get("Query ID", "")).strip()
    book_id = str(row.get("Book ID", "")).strip()
    if query_id and book_id:
        return f"{query_id}__{book_id}"

    query = str(row.get("Query", "")).strip()
    if query and book_id:
        return f"{query}__{book_id}"

    return ""


def _is_row_scored(row: Dict[str, Any]) -> bool:
    score = str(row.get("Score (0-3)", "")).strip()
    comment = str(row.get("Comment", "")).strip()
    return bool(score or comment)


def load_existing_annotations(csv_path: Path) -> Dict[str, Dict[str, str]]:
    annotations: Dict[str, Dict[str, str]] = {}
    if not csv_path.exists():
        return annotations

    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = _make_task_key(row)
            if not key:
                continue

            score = str(row.get("Score (0-3)", "")).strip()
            comment = str(row.get("Comment", "")).strip()
            if score or comment:
                annotations[key] = {"score": score, "comment": comment}

    return annotations


def save_annotations(tasks: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    row_index: Dict[str, int] = {}
    fieldnames: List[str] = list(ANNOTATION_COLUMNS)

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*fieldnames, *reader.fieldnames]))
            for row in reader:
                merged_rows.append(dict(row))
                key = _make_task_key(row)
                if key:
                    row_index[key] = len(merged_rows) - 1

    for task in tasks:
        if not _is_row_scored(task):
            continue

        key = _make_task_key(task)
        cleaned_task = {
            column: task.get(column, "")
            for column in dict.fromkeys([*fieldnames, *task.keys()])
        }

        if key and key in row_index:
            merged_rows[row_index[key]] = cleaned_task
        else:
            merged_rows.append(cleaned_task)
            if key:
                row_index[key] = len(merged_rows) - 1

        for column in cleaned_task:
            if column not in fieldnames:
                fieldnames.append(column)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)


def run_judge(
    experiment_dir: str,
    experiment_name: Optional[str] = None,
    model_id: Optional[str] = None,
    batch_size: int = 10,
    annotations_dir: str = "data/experiments/annotations",
) -> None:
    pooled_queries = load_experiment_pool(experiment_dir)
    tasks = build_annotation_rows(pooled_queries)
    if not tasks:
        print("No pooled candidates found.")
        return

    annotation_path = resolve_annotation_path(annotations_dir)
    existing = load_existing_annotations(annotation_path)

    already_done = 0
    for task in tasks:
        key = _make_task_key(task)
        if key in existing:
            task["Score (0-3)"] = existing[key]["score"]
            task["Comment"] = existing[key]["comment"]
            already_done += 1

    label = experiment_name or Path(experiment_dir).name
    print(f"Experiment: {label}")
    print(f"Run directory: {Path(experiment_dir)}")
    print(f"Pooled candidates: {len(tasks)}")
    print(f"Existing annotations reused: {already_done}")
    print(f"Annotation file: {annotation_path}")

    judge = LLMJudge(model_id=model_id)
    print(f"Judge model: {judge.model_id}")
    print("=" * 60)

    scored_count = already_done
    total = len(tasks)

    # Gather tasks that need scoring
    unscored_tasks = [ (i, t) for i, t in enumerate(tasks, start=1) if not _is_row_scored(t) ]

    if not unscored_tasks:
        print("\n" + "=" * 60)
        print("Judging complete")
        print(f"Total pooled candidates: {total}")
        print(f"Annotated candidates: {scored_count}")
        print(f"Saved to: {annotation_path}")
        return

    import concurrent.futures
    import threading

    lock = threading.Lock()

    def _process_task(item):
        index, task = item
        print(f"\n[{index}/{total}] Query ID: {task['Query ID']} | Book: {task['Title'][:40]}")
        result = judge.judge_single(
            query=task["Query"],
            title=task["Title"],
            tags=task.get("Tags", ""),
            intro=task.get("Intro", ""),
        )
        task["Score (0-3)"] = str(result["score"])
        task["Comment"] = result["reasoning"]
        print(f"  [{index}/{total}] Score: {task['Score (0-3)']}/3 | {task['Comment'][:80]}")

        nonlocal scored_count
        with lock:
            scored_count += 1
            if scored_count % batch_size == 0:
                save_annotations(tasks, annotation_path)
                print(f"  Saved progress ({scored_count}/{total})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(_process_task, unscored_tasks))

    save_annotations(tasks, annotation_path)

    print("\n" + "=" * 60)
    print("Judging complete")
    print(f"Total pooled candidates: {total}")
    print(f"Annotated candidates: {scored_count}")
    print(f"Saved to: {annotation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge pooled run candidates with an LLM")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="data/experiments/runs",
        help="Directory containing run JSON files",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional label shown in output",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Judge model (default: {JUDGE_MODELS[0]})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Save progress every N newly scored candidates",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/experiments/annotations",
        help="Directory containing the shared annotation CSV",
    )
    args = parser.parse_args()

    run_judge(
        experiment_dir=args.experiment_dir,
        experiment_name=args.experiment,
        model_id=args.model,
        batch_size=args.batch_size,
        annotations_dir=args.annotations_dir,
    )
