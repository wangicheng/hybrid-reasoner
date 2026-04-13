from typing import Any, Dict, List

from google import genai

from src.core.api_utils import (
    _is_retryable,
    get_api_key_rotator,
    get_current_api_key,
    get_rate_limiter,
    is_rate_limit_error,
)
from src.core.model_catalog import normalize_model_id


def generate_explanation(
    query: str,
    book_item: Dict[str, Any],
    context_chunks: List[str] = None,
    score_breakdown: List[Dict] = None,
    runtime_state: Dict[str, Any] = None,
    model_id: str = None,
) -> str:
    """
    使用單一指定模型產生推薦解釋。
    若模型呼叫失敗，直接拋出例外，不回傳預設說明文字。
    """
    selected_model = normalize_model_id(model_id)

    state = runtime_state if runtime_state is not None else {}
    gemini_fail_count = int(state.get("gemini_fail_count", 0))
    gemini_fail_threshold = int(state.get("gemini_fail_threshold", 3))

    def is_gemini_model(model_name: str) -> bool:
        return "gemini" in (model_name or "").lower()

    context_text = "\n---\n".join(context_chunks) if context_chunks else str(book_item.get("intro", ""))

    evidence_text = "無可用評分細節。"
    if score_breakdown:
        lines: List[str] = []
        for item in score_breakdown:
            criteria_name = item.get("criteria")
            params = item.get("params", {})
            raw_score = item.get("raw_score", 0)
            weighted_score = item.get("weighted_score", 0)

            if criteria_name == "numeric_range":
                field_name = params.get("field", "欄位")
                min_val = params.get("min_val")
                max_val = params.get("max_val")
                if field_name == "words_total":
                    field_name = "字數"
                if min_val and max_val:
                    lines.append(f"- 數值條件: {field_name} {min_val:,.0f} ~ {max_val:,.0f} ({'符合' if raw_score > 0 else '不符合'})")
                elif min_val:
                    lines.append(f"- 數值條件: {field_name} > {min_val:,.0f} ({'符合' if raw_score > 0 else '不符合'})")
                elif max_val:
                    lines.append(f"- 數值條件: {field_name} < {max_val:,.0f} ({'符合' if raw_score > 0 else '不符合'})")
            elif criteria_name == "keyword_match":
                keyword = params.get("keyword", "")
                field_name = params.get("field", "")
                if field_name == "classification":
                    field_label = "分類"
                elif field_name == "tags":
                    field_label = "標籤"
                else:
                    field_label = field_name or "欄位"
                lines.append(f"- 關鍵字比對: {field_label} 包含「{keyword}」({ '符合' if raw_score > 0 else '不符合' })")
            elif criteria_name == "semantic_similarity":
                lines.append(f"- 語意相似度: {raw_score:.2f}")
            elif criteria_name == "status_check":
                target = params.get("target_status", "")
                lines.append(f"- 狀態條件: 目標為「{target}」({ '符合' if raw_score > 0 else '不符合' })")
            elif criteria_name == "author_match":
                author = params.get("author_name", "")
                lines.append(f"- 作者條件: 指定作者「{author}」({ '符合' if raw_score > 0 else '不符合' })")
            elif weighted_score > 0:
                lines.append(f"- {criteria_name}: 加權分數 {weighted_score:.3f}")

        if lines:
            evidence_text = "\n".join(lines)

    prompt = f"""
你是網路小說推薦系統的解說助手。請根據使用者需求、作品資訊與評分依據，寫出一段精簡、可信、具體的推薦說明。

[使用者需求]
{query}

[評分依據]
{evidence_text}

[作品資訊]
作品名: {book_item.get('name')}
作者: {book_item.get('author')}
標籤: {book_item.get('tags')}

[作品內容]
{context_text}

[寫作要求]
1. 先點出這本作品和需求最相關的 2 到 3 個原因。
2. 若有不完全符合之處，可以簡短補充，但不要過度否定。
3. 僅根據提供的資訊說明，不要捏造劇情。
4. 控制在 80 到 140 字左右。
""".strip()

    last_exception = None
    is_gemma = "gemma" in selected_model.lower()
    max_attempts = 3 if is_gemma else 1
    attempt = 0
    api_key_attempts = 0
    max_api_key_attempts = len(get_api_key_rotator().api_keys)

    print(f"[explainer] trying explanation model: {selected_model}")

    while attempt < max_attempts:
        try:
            api_key = get_current_api_key()
            client = genai.Client(api_key=api_key)
            get_rate_limiter().wait(api_key)
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
            )
            result = (response.text or "").strip()
            if not result:
                raise ValueError("Explanation model returned empty text")
            print(f"[explainer] explanation succeeded: {selected_model}")
            return result
        except Exception as exc:
            last_exception = exc

            if is_gemini_model(selected_model):
                gemini_fail_count += 1
                state["gemini_fail_count"] = gemini_fail_count
                if gemini_fail_count >= gemini_fail_threshold:
                    state["gemini_disabled"] = True

            if is_rate_limit_error(exc) and api_key_attempts < max_api_key_attempts - 1:
                get_api_key_rotator().on_rate_limit_error()
                api_key_attempts += 1
                print("[explainer] current API key hit a rate limit, rotating key...")
                continue

            if is_gemma and _is_retryable(exc) and attempt < max_attempts - 1:
                print(f"[explainer] retryable error on {selected_model} (attempt {attempt + 1}/{max_attempts}): {exc}")
                import time

                time.sleep(2)
                attempt += 1
                continue

            print(f"[explainer] explanation failed on {selected_model}: {exc}")
            break

    if last_exception:
        raise RuntimeError("generate_explanation failed") from last_exception
    raise RuntimeError("generate_explanation failed without a concrete exception")
