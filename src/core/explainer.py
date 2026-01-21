import os
from typing import Dict, Any, List
from google import genai

def generate_explanation(query: str, book_item: Dict[str, Any], context_chunks: List[str] = None) -> str:
    """
    使用 Google GenAI SDK 生成推薦解釋。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    model_id = os.getenv("LLM_MODEL_ID", "gemini-2.0-flash")
    
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

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        if response.text:
            return response.text.strip()
        else:
            return "無法生成解釋 (可能觸發安全過濾或無回應)。"
    except Exception as e:
        print(f"Error generating explanation with Google GenAI: {e}")
        return "Gemini 正在閱讀大量資料，暫時無法生成解釋。"
