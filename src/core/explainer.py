import os
from typing import Dict, Any, List
from google import genai
from src.core.api_utils import retry_on_rate_limit, _is_retryable

# 可用模型清單 (與 llm.py 保持一致)
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemma-3-27b-it"]

def generate_explanation(query: str, book_item: Dict[str, Any], context_chunks: List[str] = None) -> str:
    """
    使用 Google GenAI SDK 生成推薦解釋。
    支援多模型 fallback：當主要模型遇到配額限制時，自動嘗試下一個模型。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    selected_model = os.getenv("LLM_MODEL_ID", "").strip()

    # 建立模型嘗試順序
    if selected_model and selected_model in FALLBACK_MODELS:
        models_to_try = [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
    elif selected_model:
        models_to_try = [selected_model] + FALLBACK_MODELS
    else:
        models_to_try = FALLBACK_MODELS
    
    client = genai.Client(api_key=api_key)

    context_text = ""
    if context_chunks:
        context_text = "\n---\n".join(context_chunks)
    else:
        context_text = book_item.get("intro", "無內容")

    prompt = f"""
    任務：你是一位專業的小說閱讀顧問。請根據以下提供的[書籍內容片段]，向使用者解釋為什麼這本書符合他的[查詢需求]。

    [使用者查詢]
    "{query}"

    [書籍基本資訊]
    書名: {book_item.get('name')}
    作者: {book_item.get('author')}
    標籤: {book_item.get('tags')}

    [書籍內容片段與評論資料]
    {context_text}

    撰寫要求：
    1. 引用內容片段中的情節或描述。
    2. 明確指出如何滿足查詢需求。
    3. 口語化，150字以內。
    """

    # --- 模型 Fallback 迴圈 ---
    last_exception = None
    for model_id in models_to_try:
        print(f"[explainer] 嘗試使用模型: {model_id}")
        
        @retry_on_rate_limit(max_retries=2, base_delay=5.0)
        def _call_api():
            response = client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            if response.text:
                return response.text.strip()
            else:
                return "無法生成解釋 (可能觸發安全過濾或無回應)。"

        try:
            result = _call_api()
            print(f"[explainer] 成功使用模型: {model_id}")
            return result
        except Exception as e:
            last_exception = e
            if _is_retryable(e):
                print(f"[explainer] 模型 {model_id} 配額受限，嘗試下一個模型...")
                continue
            else:
                print(f"[explainer] 模型 {model_id} 發生錯誤: {e}，嘗試下一個模型...")
                continue

    # 所有模型都失敗
    print(f"[explainer] 所有模型皆失敗，最後錯誤: {last_exception}")
    return "Gemini 正在閱讀大量資料，暫時無法生成解釋。"

