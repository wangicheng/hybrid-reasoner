import json
import os
import re
from typing import Any, Dict, List

from google import genai
from src.core.api_utils import retry_on_rate_limit


FALLBACK_MODELS = ["gemma-3-27b-it", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]


class LLMReranker:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _build_candidates(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        selected = candidates[: max(1, top_k)]
        packed = []
        for item in selected:
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            name = str(item.get("name") or item.get("title") or "")
            intro = str(item.get("intro") or item.get("summary") or item.get("description") or "")
            intro = intro.replace("\n", " ").strip()[:220]
            packed.append({
                "id": item_id,
                "name": name,
                "intro": intro,
            })
        return packed

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        packed = self._build_candidates(candidates, top_k)
        if not packed:
            return []

        selected_model = os.getenv("LLM_MODEL_ID", "").strip()
        if selected_model and selected_model in FALLBACK_MODELS:
            models_to_try = [selected_model] + [m for m in FALLBACK_MODELS if m != selected_model]
        elif selected_model:
            models_to_try = [selected_model] + FALLBACK_MODELS
        else:
            models_to_try = FALLBACK_MODELS

        prompt = (
            "你是一個小說搜尋排序器。\n"
            "請根據使用者查詢，對候選小說按相關性排序，並回傳 JSON。\n"
            "你的查詢重點是使用者對於情節以及故事相關的內容，必須根據使用者喜好進行評分。\n"
            "對於硬性的資訊（如作者、出版日或是小說長度等）不需要關注。\n"
            "給出0.00~1.00的分數。\n"
            "回傳格式必須是："
            "{\"ranking\":[{\"id\":\"...\",\"score\":0.0-1.0}],\"note\":\"optional\"}\n"
            "只輸出 JSON，不要 markdown。\n\n"
            f"使用者查詢：{query}\n\n"
            f"候選清單：{json.dumps(packed, ensure_ascii=False)}"
        )

        for model_id in models_to_try:
            try:
                @retry_on_rate_limit(max_retries=2, base_delay=5.0)
                def _call():
                    return self.client.models.generate_content(model=model_id, contents=prompt)

                response = _call()
                raw_text = (response.text or "").strip()
                if not raw_text:
                    continue

                raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```\s*$", "", raw_text).strip()
                parsed = json.loads(raw_text)

                ranking = parsed.get("ranking", []) if isinstance(parsed, dict) else []
                normalized = []
                for rank_idx, entry in enumerate(ranking):
                    if not isinstance(entry, dict):
                        continue
                    candidate_id = str(entry.get("id", "")).strip()
                    if not candidate_id:
                        continue
                    try:
                        score = float(entry.get("score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    score = max(0.0, min(1.0, score))
                    normalized.append({
                        "id": candidate_id,
                        "llm_rerank_score": score,
                        "llm_rerank_rank": rank_idx + 1,
                    })

                if normalized:
                    return normalized
            except Exception:
                continue

        return []
