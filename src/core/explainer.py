import os
from typing import Dict, Any, List
from google import genai
from src.core.api_utils import retry_on_rate_limit, _is_retryable

# 可用模型清單 (與 llm.py 保持一致)
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemma-3-27b-it"]

def generate_explanation(
    query: str, 
    book_item: Dict[str, Any], 
    context_chunks: List[str] = None,
    score_breakdown: List[Dict] = None  # 新增：評分細節
) -> str:
    """
    使用 Google GenAI SDK 生成推薦解釋。
    支援多模型 fallback：當主要模型遇到配額限制時，自動嘗試下一個模型。
    現在包含評分證據 (score_breakdown) 以產出基於證據的解釋。
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

    # --- 將結構化的評分數據轉為自然語言描述 (Evidence Formatting) ---
    evidence_text = "無評分資料"
    if score_breakdown:
        lines = []
        for b in score_breakdown:
            c_name = b.get('criteria')
            params = b.get('params', {})
            raw_score = b.get('raw_score', 0)
            weighted_score = b.get('weighted_score', 0)

            if c_name == 'numeric_range':
                f_name = params.get('field', '欄位')
                min_val = params.get('min_val')
                max_val = params.get('max_val')
                range_desc = f_name
                if f_name == 'words_total':
                    range_desc = '總字數'
                if min_val and max_val:
                    lines.append(f"- 數值條件: {range_desc} 範圍 {min_val:,.0f} ~ {max_val:,.0f} ({'符合' if raw_score > 0 else '不符合'})")
                elif min_val:
                    lines.append(f"- 數值條件: {range_desc} > {min_val:,.0f} ({'符合' if raw_score > 0 else '不符合'})")
                elif max_val:
                    lines.append(f"- 數值條件: {range_desc} < {max_val:,.0f} ({'符合' if raw_score > 0 else '不符合'})")

            elif c_name == 'keyword_match':
                kw = params.get('keyword', '')
                field = params.get('field', '')
                field_label = '分類' if field == 'classification' else ('標籤' if field == 'tags' else field)
                lines.append(f"- 關鍵字匹配: {field_label}「{kw}」({'匹配成功' if raw_score > 0 else '未匹配'})")

            elif c_name == 'semantic_similarity':
                lines.append(f"- 內容語意相似度: {raw_score:.2f} 分")

            elif c_name == 'status_check':
                target = params.get('target_status', '')
                lines.append(f"- 連載狀態: 要求「{target}」({'符合' if raw_score > 0 else '不符合'})")

            elif c_name == 'author_match':
                author = params.get('author_name', '')
                lines.append(f"- 作者匹配:「{author}」({'符合' if raw_score > 0 else '不符合'})")

            else:
                if weighted_score > 0:
                    lines.append(f"- {c_name}: 得分 {weighted_score:.3f}")

        if lines:
            evidence_text = "\n".join(lines)

    prompt = f"""
    任務：你是一位專業的小說推薦顧問。請根據[系統評分證據]與[書籍內容]，解釋這本書為何適合使用者的查詢。

    [使用者查詢]
    "{query}"

    [系統評分證據] (這是系統演算法選中此書的硬性理由)
    {evidence_text}

    [書籍基本資訊]
    書名: {book_item.get('name')}
    作者: {book_item.get('author')}
    標籤: {book_item.get('tags')}

    [書籍內容片段與評論資料]
    {context_text}

    [撰寫要求]
    1. **必須提及評分證據**：例如「這本書不僅符合您對『奇幻』類型的要求...」或「字數達 XX 萬字，滿足閱讀份量...」。
    2. **結合內容**：引用簡介中的情節或描述來佐證推薦。
    3. **語氣親切**：像書店店員一樣推薦，約 100-150 字。
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

