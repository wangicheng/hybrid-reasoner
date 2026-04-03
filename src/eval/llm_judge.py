"""
LLM-as-a-Judge: 自動化實驗結果評分

使用 LLM 自動對檢索結果進行 0-3 分的關聯度評分。
參考論文: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023)

用法:
    python -m src.eval.llm_judge
    python -m src.eval.llm_judge --experiment pilot_test --model gemini-2.5-flash-lite
"""

import os
import re
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.core.api_utils import retry_on_rate_limit, _is_retryable, get_current_api_key

# Load environment variables from .env file
load_dotenv()

# 可用模型清單 (依優先順序排列)
JUDGE_MODELS = ["gemma-3-27b-it", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]

# ============================================================
# Prompt 設計
# ============================================================

SYSTEM_PROMPT = """\
You are an expert relevance assessor for a web novel recommendation system.
Your task is to judge how well a recommended book matches a user's search query.

### Scoring Rubric (0-3 scale):
- **Score 0 – Irrelevant**: The book is completely unrelated to the user's core intent.
- **Score 1 – Marginally Relevant**: The book shares only a superficial or tangential connection to the query. It might match one minor keyword but fails to satisfy the core intent.
- **Score 2 – Partially Relevant**: The book satisfies some of the user's key requirements but misses important aspects. It could be a reasonable recommendation but is not ideal.
- **Score 3 – Highly Relevant**: The book is an excellent match for the query. It satisfies the user's core intent and most (if not all) stated preferences. This is the kind of book the user is clearly looking for.

### Important Guidelines:
1. Focus on the USER'S INTENT, not just keyword overlap.
2. Ignore hard constraints like status (ongoing/completed) or word count for your scoring. Only focus on semantic and genre relevance (tags, themes, plot).
3. Books with empty or missing information (Title "Unknown", no intro) should be scored 0.
4. Provide a brief reasoning in Traditional Chinese (繁體中文) explaining your score.

### Output Format:
Return a JSON object with exactly two fields:
{
  "reasoning": "<brief explanation in Traditional Chinese>",
  "score": <integer 0-3>
}
"""


def build_user_prompt(query: str, title: str, tags: str, intro: str) -> str:
    """組裝 LLM 評分用的 User Prompt"""
    return f"""\
### 使用者查詢 (User Query):
{query}

### 推薦書籍資訊 (Recommended Book):
- 書名: {title}
- 標籤: {tags}
- 簡介: {intro}

Please evaluate the relevance of this book to the user's query and return your judgment as JSON.
"""


# ============================================================
# 核心評分邏輯
# ============================================================

class LLMJudge:
    """使用 LLM 進行自動化關聯度評分"""

    def __init__(self, model_id: Optional[str] = None):
        try:
            api_key = get_current_api_key()
        except ValueError as e:
            raise ValueError(f"無法取得 API Key: {e}")

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id or JUDGE_MODELS[0]
        self.models_to_try = [self.model_id] + [m for m in JUDGE_MODELS if m != self.model_id]

        # JSON Schema for structured output
        self.response_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the score in Traditional Chinese"
                },
                "score": {
                    "type": "integer",
                    "description": "Relevance score from 0 to 3"
                }
            },
            "required": ["reasoning", "score"]
        }

    def judge_single(self, query: str, title: str, tags: str, intro: str) -> Dict[str, Any]:
        """
        對單一 Query-Book pair 進行評分。
        回傳 {"score": int, "reasoning": str}
        """
        # 防呆：書籍資訊不完整直接給 0 分
        if not title or title == "Unknown" or title.strip() == "":
            return {"score": 0, "reasoning": "書籍資訊不完整（Unknown），無法評分。"}

        if not intro or intro.strip() == "" or intro.strip() == "[標籤: ]\n":
            return {"score": 0, "reasoning": "書籍缺少簡介資訊，無法評估相關性。"}

        user_prompt = build_user_prompt(query, title, tags, intro)
        last_exception = None

        for model_id in self.models_to_try:
            try:
                result = self._call_llm(model_id, user_prompt)
                return result
            except Exception as e:
                last_exception = e
                print(f"  [judge] 模型 {model_id} 失敗: {e}")
                continue

        # 所有模型都失敗
        print(f"  [judge] 所有模型皆失敗，最後錯誤: {last_exception}")
        return {"score": 0, "reasoning": f"LLM 評分失敗: {last_exception}"}

    def _call_llm(self, model_id: str, user_prompt: str) -> Dict[str, Any]:
        """呼叫 LLM 進行評分 (含 retry 機制)"""

        @retry_on_rate_limit(max_retries=3, base_delay=5.0)
        def _do_generate():
            is_gemma = "gemma" in model_id.lower()

            if is_gemma:
                # Gemma 不支援 structured output
                config_args = {}
                final_contents = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"{user_prompt}\n\n"
                    "IMPORTANT: Output ONLY valid JSON with keys 'reasoning' (string) and 'score' (integer 0-3) in that order. No markdown."
                )
            else:
                config_args = {
                    "response_mime_type": "application/json",
                    "response_schema": self.response_schema,
                    "system_instruction": SYSTEM_PROMPT,
                }
                final_contents = user_prompt

            response = self.client.models.generate_content(
                model=model_id,
                contents=final_contents,
                config=types.GenerateContentConfig(**config_args)
            )

            if not response.text:
                raise ValueError("Empty response from LLM")

            raw_text = response.text.strip()
            # Strip markdown code fence if present
            raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```\s*$", "", raw_text)
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            # Validate score range
            score = int(parsed.get("score", 0))
            score = max(0, min(3, score))  # Clamp to 0-3

            return {
                "score": score,
                "reasoning": parsed.get("reasoning", "")
            }

        return _do_generate()


# ============================================================
# CSV 讀寫與主流程
# ============================================================

def load_blind_tasks(csv_path: str) -> List[Dict[str, Any]]:
    """讀取盲測 CSV"""
    tasks = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)
    return tasks


def load_existing_annotations(csv_path: str) -> Dict[str, Dict[str, str]]:
    """讀取已標註的結果，回傳 {query_id_book_id: {"score": score, "comment": comment}}"""
    annotated = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_id = row.get("Query ID", "").strip()
                b_id = row.get("Book ID", "").strip()
                if not q_id or not b_id:
                    continue
                key = f"{q_id}_{b_id}"
                score = row.get("Score (0-3)", "")
                comment = row.get("Comment", "")
                if score != "":
                    # 儲存分數與評論，以便續傳時恢復
                    annotated[key] = {
                        "score": score,
                        "comment": comment
                    }
    return annotated


def _make_task_key(row: Dict[str, Any]) -> str:
    q_id = str(row.get("Query ID", "")).strip()
    b_id = str(row.get("Book ID", "")).strip()
    return f"{q_id}_{b_id}" if q_id and b_id else ""


def _is_row_scored(row: Dict[str, Any]) -> bool:
    score = str(row.get("Score (0-3)", "")).strip()
    comment = str(row.get("Comment", "")).strip()
    return bool(score or comment)


def save_annotated_csv(tasks: List[Dict[str, Any]], csv_path: str):
    """將標註結果合併存入 CSV，避免洗掉既有的非 blind 列。"""
    if not tasks and not os.path.exists(csv_path):
        return

    merged_rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    row_index: Dict[str, int] = {}

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                merged_rows.append(dict(row))
                key = _make_task_key(row)
                if key:
                    row_index[key] = len(merged_rows) - 1

    for task in tasks:
        if not _is_row_scored(task):
            continue

        key = _make_task_key(task)
        cleaned_task = {k: v for k, v in task.items() if v != ""}

        if key and key in row_index:
            target_row = merged_rows[row_index[key]]
            for key_name, value in cleaned_task.items():
                if value != "":
                    target_row[key_name] = value
        else:
            merged_rows.append(cleaned_task)
            if key:
                row_index[key] = len(merged_rows) - 1

        for column_name in cleaned_task.keys():
            if column_name not in fieldnames:
                fieldnames.append(column_name)

    if not merged_rows:
        return

    if not fieldnames:
        fieldnames = list(merged_rows[0].keys())
    else:
        for row in merged_rows:
            for column_name in row.keys():
                if column_name not in fieldnames:
                    fieldnames.append(column_name)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)


def _resolve_pools_dir(base_dir: Path) -> Path:
    return base_dir / "pools"


def run_judge(
    experiment_name: str = "pilot_test",
    model_id: Optional[str] = None,
    batch_size: int = 10,
    experiment_dir: str = "data/experiments/pools",
):
    """
    主流程：讀取盲測資料，使用 LLM 進行評分，儲存結果。
    支援中斷續傳：已評過的題目會自動跳過。
    """
    base_dir = _resolve_pools_dir(Path(experiment_dir))
    blind_csv = base_dir / f"{experiment_name}_blind.csv"
    annotated_csv = base_dir / f"{experiment_name}_annotated.csv"

    if not blind_csv.exists():
        print(f"❌ 找不到盲測檔案: {blind_csv}")
        return

    # 1. 讀取資料
    print(f"📂 讀取盲測資料: {blind_csv}")
    tasks = load_blind_tasks(str(blind_csv))
    print(f"   共 {len(tasks)} 筆待評資料")

    # 2. 讀取已標註的結果 (支援續傳)
    existing = load_existing_annotations(str(annotated_csv))
    already_done_total = len(existing)
    
    # 找出當前 tasks 中有多少是已經評分過的
    tasks_already_done = 0
    for task in tasks:
        key = f"{task.get('Query ID', '').strip()}_{task.get('Book ID', '').strip()}"
        if key in existing:
            tasks_already_done += 1
            
    if already_done_total > 0:
        print(f"   ⏩ 資料庫中已有 {already_done_total} 筆紀錄")
        print(f"   ⏩ 目前待評清單中有 {tasks_already_done} 筆已評分，將自動跳過")

    # 3. 初始化 LLM Judge
    judge = LLMJudge(model_id=model_id)
    print(f"🤖 使用模型: {judge.model_id}")
    print(f"{'='*60}")

    # 4. 逐筆評分
    total = len(tasks)
    scored_count = tasks_already_done
    skipped_unknown = 0

    for i, task in enumerate(tasks):
        q_id = task.get("Query ID", "").strip()
        b_id = task.get("Book ID", "").strip()
        key = f"{q_id}_{b_id}"

        # 跳過已評分的
        if key in existing:
            task["Score (0-3)"] = existing[key]["score"]
            if "Comment" in task or "comment" in task: # 避免 key 大小寫問題
                task["Comment"] = existing[key]["comment"]
            else:
                task["Comment"] = existing[key]["comment"]
            continue

        query = task["Query"]
        title = task.get("Title", "")
        intro_raw = task.get("Intro", "")

        tags_str = ""
        clean_intro = intro_raw
        if "[標籤:" in intro_raw:
            start_idx = intro_raw.find("[標籤:") + 4
            end_idx = intro_raw.find("]", start_idx)
            if end_idx != -1:
                tags_str = intro_raw[start_idx:end_idx].strip()
            clean_intro = re.sub(r"\[標籤:.*?\]\s*\n*", "", intro_raw, count=1).strip()

        # 進度顯示
        print(f"\n[{i+1}/{total}] Query ID: {task['Query ID']} | Book: {title[:30]}...")

        result = judge.judge_single(query, title, tags_str, clean_intro)
        score = result["score"]
        reasoning = result["reasoning"]

        task["Score (0-3)"] = str(score)
        task["Comment"] = reasoning
        existing[key] = {"score": str(score), "comment": reasoning}
        scored_count += 1

        # 標記跳過的 Unknown
        if title == "Unknown" or title.strip() == "":
            skipped_unknown += 1

        print(f"   ✅ Score: {score}/3 | {reasoning[:60]}...")

        # 定期儲存 (每 batch_size 筆存一次)
        if scored_count % batch_size == 0:
            save_annotated_csv(tasks, str(annotated_csv))
            print(f"\n   💾 已儲存進度 ({scored_count}/{total})")

    # 5. 最終儲存
    save_annotated_csv(tasks, str(annotated_csv))

    # 6. 報告
    print(f"\n{'='*60}")
    print(f"🎉 評分完成！")
    print(f"   📊 總計: {total} 筆")
    print(f"   ✅ 已評分: {scored_count} 筆")
    print(f"   ⏭️  Unknown 自動 0 分: {skipped_unknown} 筆")
    print(f"   💾 結果已儲存至: {annotated_csv}")
    print(f"\n   下一步: 執行 python -m src.eval.metrics --experiment {experiment_name} 計算 NDCG")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge 自動評分")
    parser.add_argument("--experiment", type=str, default="pilot_test",
                        help="實驗名稱 (預設: pilot_test)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"指定 LLM 模型 (預設: {JUDGE_MODELS[0]})")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每幾筆儲存一次進度 (預設: 10)")
    parser.add_argument("--experiment-dir", type=str, default="data/experiments/runs/batch_YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    run_judge(
        experiment_name=args.experiment,
        model_id=args.model,
        batch_size=args.batch_size,
        experiment_dir=args.experiment_dir,
    )
