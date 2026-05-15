"""
DAT Router — Dynamic Alpha Tuning for the 3+1 Layer Scoring Pipeline.

This module provides the ``DATRouter`` class, which uses a lightweight LLM
call to dynamically balance the weight between semantic (Plot) and lexical
(BM25) scores in Layer 1 of the scoring pipeline.

The router works by:
1. Taking the Top-1 result from both Vector search and BM25 search (Scout Phase)
2. Asking an LLM to rate how well each result satisfies the user query (0-5 scale)
3. Computing α = e_plot / (e_plot + e_bm25), clamped to [0.1, 0.9]

α is then used in Layer 1:  Base = α × Norm_S_plot + (1-α) × Norm_S_BM25.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings


# ─────────────────────────────────────────────────────────────────────
# DAT LLM Prompt & Schema
# ─────────────────────────────────────────────────────────────────────

_DAT_SYSTEM_PROMPT = """\
你是一個專業的小說搜尋引擎相關性評估專家。
你的任務是評估三篇候選書籍簡介（文件 A、文件 B 與 文件 C）對使用者搜尋意圖的「有效性」。

讀者的搜尋意圖通常分為兩種：
1. 精確搜尋：找特定的書名、主角名或專有名詞。
2. 語意搜尋：找特定的劇情套路、風格、流派或世界觀設定（例如：賽博龐克、後宮、系統、高智商）。

請仔細閱讀使用者的搜尋詞與三本書的簡介，並嚴格按照以下 0~5 分的標準進行獨立評分：

【0~5 分評估標準】
- 5分 (Direct hit/完美命中)：書名、核心設定或主線劇情完全符合使用者的搜尋意圖。讀者看到這本書會覺得「這就是我要找的」。
- 4分 (Highly relevant/高度相關)：雖然不完全是精確命中，但包含了高度相關的核心元素或套路，極大機率能滿足讀者的閱讀期望。
- 3分 (Good wrong result/部分契合)：書籍的整體氛圍或次要元素與查詢沾邊。算是一個可以接受的替代推薦，但並非完美解答。
- 2分 (Tangential/糟糕的擦邊)：查詢的元素僅作為極為次要的背景提及。整體劇情走向與讀者的期望有明顯落差，讀者大概率會失望。
- 1分 (Misleading/誤導性陷阱)：字面命中但語意完全錯誤（例如搜「系統流」卻給出「太陽系百科」）。
- 0分 (Completely off-track/毫無關聯)：書籍內容與使用者的搜尋意圖完全無關。

【輸出限制 - 絕對嚴格】
你只能輸出一個合法的 JSON 物件，絕對不能包含任何 Markdown 標記 (如 ```json)、思考過程或其他文字。
JSON 格式必須完全一致如下：
{"plot_score": <文件A的0-5評分>, "bm25_score": <文件B的0-5評分>, "tag_score": <文件C的0-5評分>}
"""

_DAT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "plot_score": {
            "type": "INTEGER",
            "description": "Relevance score for the semantic (plot) candidate, 0-5.",
        },
        "bm25_score": {
            "type": "INTEGER",
            "description": "Relevance score for the keyword (BM25) candidate, 0-5.",
        },
        "tag_score": {
            "type": "INTEGER",
            "description": "Relevance score for the tag vector candidate, 0-5.",
        },
    },
    "required": ["plot_score", "bm25_score", "tag_score"],
}


# ─────────────────────────────────────────────────────────────────────
# Query Cache (in-memory with TTL)
# ─────────────────────────────────────────────────────────────────────

_alpha_cache: Dict[str, Tuple[float, float, float]] = {}   # {query_hash: (alpha, beta, timestamp)}
CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _cache_key(query: str) -> str:
    """Generate a stable cache key from the query string."""
    return hashlib.md5(query.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# DATRouter
# ─────────────────────────────────────────────────────────────────────

class DATRouter:
    """Dynamic Alpha Tuning router.

    Determines the optimal α for balancing Plot (semantic vector) and BM25
    (lexical keyword) scores in Layer 1 of the 3+1 scoring pipeline.

    Parameters
    ----------
    default_alpha : float
        Fallback α value when LLM is unavailable or an early-exit fires.
    timeout_ms : int
        Maximum milliseconds to wait for the LLM response.
    short_query_threshold : int
        Queries with character count <= this value trigger the short-query
        early exit.
    short_query_alpha : float
        Fixed α for short queries (typically biased toward BM25).
    model_id : str
        LLM model identifier.  Empty string = use project default.
    """

    def __init__(
        self,
        default_alpha: float = 0.5,
        timeout_ms: int = 2000,
        short_query_threshold: int = 2,
        short_query_alpha: float = 0.3,
        model_id: str = "",
    ):
        self.default_alpha = default_alpha
        self.timeout_sec = timeout_ms / 1000.0
        self.short_query_threshold = short_query_threshold
        self.short_query_alpha = short_query_alpha
        self.model_id = model_id or None

    # ── Public API ──────────────────────────────────────────────────

    def get_dynamic_alpha_and_beta(
        self,
        query: str,
        plot_top1: Dict[str, Any],
        bm25_top1: Dict[str, Any],
        tag_top1: Dict[str, Any],
        model_id: Optional[str] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Compute the dynamic α and β for a given query (Tri-Track DAT).

        Returns
        -------
        alpha : float
            The computed α value in [0.1, 0.9].
        beta : float
            The computed β value.
        info : dict
            Metadata dict with keys: source, plot_score, bm25_score, tag_score,
            latency_ms, plot_top1_id, bm25_top1_id, tag_top1_id.
        """
        info: Dict[str, Any] = {
            "source": "fallback",
            "plot_score": None,
            "bm25_score": None,
            "tag_score": None,
            "latency_ms": 0.0,
            "plot_top1_id": None,
            "bm25_top1_id": None,
            "tag_top1_id": None,
        }

        # ── Early Exit 1: Short Query ──
        max_beta = settings.DAT_MAX_BETA
        fallback_beta = settings.TAG_BONUS_BETA

        if len(query.strip()) <= self.short_query_threshold:
            alpha = self.short_query_alpha
            info["source"] = "early_exit_short"
            print(f"[DAT:EarlyExit] Short query (len={len(query.strip())}), α={alpha:.3f}")
            return alpha, fallback_beta, info

        # ── Early Exit 2: Cache Hit ──
        cache_key = _cache_key(query.strip())
        cached = _alpha_cache.get(cache_key)
        if cached is not None:
            cached_alpha, cached_beta, cached_ts = cached
            if (time.time() - cached_ts) < CACHE_TTL_SECONDS:
                info["source"] = "cache"
                print(f"[DAT:Cache] Hit for query hash={cache_key[:8]}, α={cached_alpha:.3f}, β={cached_beta:.3f}")
                return cached_alpha, cached_beta, info

        # ── Extract IDs ──
        plot_id = self._extract_id(plot_top1)
        bm25_id = self._extract_id(bm25_top1)
        tag_id = self._extract_id(tag_top1)
        info["plot_top1_id"] = plot_id
        info["bm25_top1_id"] = bm25_id
        info["tag_top1_id"] = tag_id

        # ── Early Exit 3: Same Book ──
        if plot_id and bm25_id and tag_id and plot_id == bm25_id == tag_id:
            alpha = self.default_alpha
            beta = fallback_beta
            info["source"] = "early_exit_same_book"
            print(f"[DAT:EarlyExit] Same book (id={plot_id}), α={alpha:.3f}, β={beta:.3f}")
            return alpha, beta, info

        # ── Extract summaries for LLM ──
        plot_title = self._extract_title(plot_top1)
        plot_summary = self._extract_summary(plot_top1)
        bm25_title = self._extract_title(bm25_top1)
        bm25_summary = self._extract_summary(bm25_top1)
        tag_title = self._extract_title(tag_top1)
        tag_summary = self._extract_summary(tag_top1)

        plot_score_raw = plot_top1.get("score", 0.0)
        bm25_score_raw = bm25_top1.get("score", 0.0)
        tag_score_raw = tag_top1.get("score", 0.0)
        print(
            f"[DAT:Scout] Plot: 《{plot_title}》({plot_score_raw:.4f}), "
            f"BM25: 《{bm25_title}》({bm25_score_raw:.2f}), "
            f"Tag: 《{tag_title}》({tag_score_raw:.4f})"
        )

        # ── LLM Call ──
        started = time.perf_counter()
        try:
            e_plot, e_bm25, e_tag = self._llm_score(
                query, plot_title, plot_summary, bm25_title, bm25_summary, tag_title, tag_summary,
                model_id=model_id,
                plot_top1_raw=plot_top1, bm25_top1_raw=bm25_top1, tag_top1_raw=tag_top1,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            info["latency_ms"] = latency_ms
            info["plot_score"] = e_plot
            info["bm25_score"] = e_bm25
            info["tag_score"] = e_tag

            # ── Compute α and β ──
            if e_plot == 0 and e_bm25 == 0:
                alpha = self.default_alpha
                info["source"] = "early_exit_zero"
                print(f"[DAT:LLM] Both scores=0, fallback α={alpha:.3f} ({latency_ms:.0f}ms)")
            else:
                raw_alpha = e_plot / (e_plot + e_bm25)
                alpha = max(0.1, min(0.9, raw_alpha))
                info["source"] = "llm"
            
            beta = max_beta * (e_tag / 5.0)
            
            if info["source"] == "llm":
                print(
                    f"[DAT:LLM] plot={e_plot}, bm25={e_bm25}, tag={e_tag} → "
                    f"α={alpha:.3f}, β={beta:.3f} ({latency_ms:.0f}ms)"
                )

        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            info["latency_ms"] = latency_ms
            alpha = self.default_alpha
            beta = fallback_beta
            info["source"] = "fallback"
            print(f"[DAT:Error] LLM failed ({latency_ms:.0f}ms): {exc}, fallback α={alpha:.3f}, β={beta:.3f}")

        # ── Cache Result ──
        _alpha_cache[cache_key] = (alpha, beta, time.time())

        return alpha, beta, info

    # ── Private Helpers ─────────────────────────────────────────────

    def _llm_score(
        self,
        query: str,
        plot_title: str,
        plot_summary: str,
        bm25_title: str,
        bm25_summary: str,
        tag_title: str,
        tag_summary: str,
        model_id: Optional[str] = None,
        plot_top1_raw: Optional[Dict[str, Any]] = None,
        bm25_top1_raw: Optional[Dict[str, Any]] = None,
        tag_top1_raw: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int, int]:
        """Call the LLM to score three candidates. Returns (plot_score, bm25_score, tag_score)."""
        from src.core.llm import _generate_json_from_contents

        plot_tags = self._extract_tags(plot_top1_raw) if hasattr(self, '_extract_tags') else ""
        bm25_tags = self._extract_tags(bm25_top1_raw) if hasattr(self, '_extract_tags') else ""
        tag_tags = self._extract_tags(tag_top1_raw) if hasattr(self, '_extract_tags') else ""
        
        plot_info = plot_summary[:500] if plot_summary else "(無簡介)"
        if plot_tags:
            plot_info = f"{plot_info}\n標籤：{plot_tags}"
            
        bm25_info = bm25_summary[:500] if bm25_summary else "(無簡介)"
        if bm25_tags:
            bm25_info = f"{bm25_info}\n標籤：{bm25_tags}"

        tag_info = tag_summary[:500] if tag_summary else "(無簡介)"
        if tag_tags:
            tag_info = f"{tag_info}\n標籤：{tag_tags}"

        user_message = (
            f"使用者搜尋詞：\"{query}\"\n\n"
            f"【文件 A】(來源：語意檢索)\n"
            f"書名：《{plot_title}》\n"
            f"簡介/標籤：{plot_info}\n\n"
            f"【文件 B】(來源：關鍵字檢索)\n"
            f"書名：《{bm25_title}》\n"
            f"簡介/標籤：{bm25_info}\n\n"
            f"【文件 C】(來源：標籤向量檢索)\n"
            f"書名：《{tag_title}》\n"
            f"簡介/標籤：{tag_info}"
        )

        effective_model = model_id or self.model_id
        parsed, _metadata = _generate_json_from_contents(
            contents=user_message,
            task_label="dat_tri_track",
            system_instruction=_DAT_SYSTEM_PROMPT,
            response_schema=_DAT_RESPONSE_SCHEMA,
            model_id=effective_model,
            sampling_temperature=0.1,
            enforce_rate_limit=True,
        )

        e_plot = int(parsed.get("plot_score", 0))
        e_bm25 = int(parsed.get("bm25_score", 0))
        e_tag = int(parsed.get("tag_score", 0))
        
        # Clamp to valid range
        e_plot = max(0, min(5, e_plot))
        e_bm25 = max(0, min(5, e_bm25))
        e_tag = max(0, min(5, e_tag))

        return e_plot, e_bm25, e_tag

    @staticmethod
    def _extract_id(result: Dict[str, Any]) -> Optional[str]:
        """Extract book ID from a search result dict."""
        # Vector search returns payload.id, BM25 returns item.id
        payload = result.get("payload") or {}
        item = result.get("item") or {}
        bid = payload.get("id") or item.get("id") or result.get("id")
        return str(bid).strip() if bid else None

    @staticmethod
    def _extract_title(result: Dict[str, Any]) -> str:
        """Extract book title from a search result dict."""
        payload = result.get("payload") or {}
        item = result.get("item") or {}
        return str(payload.get("name") or item.get("name") or "(unknown)").strip()

    @staticmethod
    def _extract_summary(result: Dict[str, Any]) -> str:
        """Extract book intro/synopsis from a search result dict."""
        payload = result.get("payload") or {}
        item = result.get("item") or {}
        return str(payload.get("intro") or item.get("intro") or "").strip()

    @staticmethod
    def _extract_tags(result: Dict[str, Any]) -> str:
        """Extract tags from a search result dict, formatted as comma-separated string."""
        payload = result.get("payload") or {}
        item = result.get("item") or {}
        tags = payload.get("tags") or item.get("tags") or []
        if isinstance(tags, str):
            try:
                import json as _json
                tags = _json.loads(tags)
            except Exception:
                return tags.strip()
        if isinstance(tags, list):
            return "、".join(str(t).strip() for t in tags[:10] if str(t).strip())
        return ""
