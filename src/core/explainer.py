import os
import time
import google.generativeai as genai
from typing import Dict, Any, List, Optional

# 定義備用模型清單 (優先順序: 最快/最便宜 -> 強大/貴)
# 注意: 請確保你的 API Key 有權限存取這些模型
FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",    # 如果有的話
    "gemini-1.5-pro",
    "gemini-1.0-pro"
]

def generate_explanation(query: str, book_item: Dict[str, Any], context_chunks: List[str] = None) -> str:
    """
    使用 Google Generative AI SDK 生成推薦解釋。
    具備自動切換模型 (Fallback Mechanism) 功能。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    
    # 準備 Prompt (只做一次)
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

    # 取得優先使用的模型 ID (從環境變數)，如果沒有就用列表第一個
    primary_model = os.getenv("LLM_MODEL_ID", FALLBACK_MODELS[0])
    
    # 建立嘗試順序：優先模型 -> 其他備用模型
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    
    last_error = None

    for model_name in models_to_try:
        try:
            # print(f"🤖 Trying model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            last_error = e
            # print(f"⚠️ Model {model_name} failed: {error_msg}. Switching to next...")
            
            # 如果是配額問題 (429)，稍微睡一下再試下一個，避免瞬間打爆所有限制
            if "429" in error_msg or "Quota" in error_msg:
                time.sleep(1)
            continue

    # 如果所有模型都失敗了，回傳備援簡介
    print(f"❌ All models failed. Last error: {last_error}")
    return _get_fallback_msg(book_item)

def _get_fallback_msg(book_item: Dict[str, Any]) -> str:
    """產生不依賴 AI 的靜態推薦語"""
    msg = f"我們認為《{book_item.get('name')}》這本書相當符合您的條件。"
    if book_item.get("intro"):
        intro = book_item.get("intro")
        # 去除換行，取前 60 字
        clean_intro = intro.replace('\n', '')[:60]
        msg += f" 簡介提到：{clean_intro}..."
    return msg