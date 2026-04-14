"""
階段一：檢索前的意圖解析與查詢重構 (Pre-Retrieval Query Preprocessor)

核心功能：
1. LLM 結構化意圖拆解 — 將自然語言拆分為正向搜尋詞 / 硬排除 / 軟排除
2. 查詢淨化 — 只將正向詞送入 BM25，避免否定詞汙染計分
3. 布林邏輯轉換 — 將排除約束轉化為 AND NOT 布林表達式
4. 負權重機制 — 對軟排除命中的文件進行 BM25 扣分

Example:
    query = "推薦魔法學校小說，不要悲劇，盡量不要後宮"
    ↓ LLM 拆解
    positive_terms = ["魔法學校", "奇幻", "校園"]
    hard_exclusions = [NegationConstraint(term="悲劇", strength="hard")]
    soft_exclusions = [NegationConstraint(term="後宮", strength="soft", weight=-0.5)]
    ↓ 查詢淨化
    sanitized_bm25_query = "魔法學校 奇幻 校園"
    boolean_query = "(魔法學校 奇幻 校園) AND NOT (悲劇)"
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.schemas import NegationConstraint, QueryIntent, QueryParseResult


# ────────────────────────────────────────────
# 否定詞偵測 patterns（用於 fallback / 驗證）
# ────────────────────────────────────────────
# 否定關鍵字清單（用於構建正則邊界）
_NEG_KEYWORDS = "|".join([
    "不要", "不想要", "不能有", "排除", "禁止", 
    "沒有", "別有", "不含", "除了", "盡量不要", 
    "最好不要", "少一點", "希望不要", "儘量避免", 
    "盡量避免", "不太想"
])

_HARD_NEGATION_PATTERNS = [
    r"不要((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的])+)",
    r"不想要((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的])+)",
    r"不能有((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的])+)",
    r"排除((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的])+)",
    r"禁止((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、意圖])+)",
    r"沒有((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、])+?)的",
    r"別有((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的])+)",
    r"不含((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的])+)",
    r"除了((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、])+?)之外",
]

_SOFT_NEGATION_PATTERNS = [
    r"盡量不要((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
    r"最好不要((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
    r"少一點((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
    r"希望不要((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
    r"儘量避免((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
    r"盡量避免((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
    r"不太想((?:(?!" + _NEG_KEYWORDS + r")[^\s，,。！!；;、的小說])+)",
]

# 否定修飾詞（用於清洗 search_terms）
_NEGATION_PREFIXES = frozenset({
    "不要", "不想要", "不能有", "排除", "禁止",
    "別有", "不含", "盡量不要", "最好不要",
    "少一點", "希望不要", "儘量避免", "盡量避免",
    "不太想", "沒有",
})


def _regex_fallback_extract(query: str) -> QueryIntent:
    """
    當 LLM 未能回傳 query_intent 時的正則 fallback。
    用 regex 從原始查詢中提取否定約束。
    """
    hard_exclusions: List[NegationConstraint] = []
    soft_exclusions: List[NegationConstraint] = []
    negated_spans: List[Tuple[int, int]] = []

    # 先抓軟排除（更長的 pattern 要先匹配，避免被硬排除吃掉）
    for pattern in _SOFT_NEGATION_PATTERNS:
        for match in re.finditer(pattern, query):
            term = match.group(1).strip()
            if term and len(term) >= 1:
                soft_exclusions.append(NegationConstraint(
                    term=term,
                    strength="soft",
                    weight=-0.5,
                    reason=f"regex soft match: {match.group(0).strip()}"
                ))
                negated_spans.append((match.start(), match.end()))

    # 再抓硬排除
    for pattern in _HARD_NEGATION_PATTERNS:
        for match in re.finditer(pattern, query):
            # 跳過已被軟排除匹配的 span
            if any(match.start() >= s and match.start() < e for s, e in negated_spans):
                continue
            term = match.group(1).strip()
            if term and len(term) >= 1:
                hard_exclusions.append(NegationConstraint(
                    term=term,
                    strength="hard",
                    weight=-1.0,
                    reason=f"regex hard match: {match.group(0).strip()}"
                ))
                negated_spans.append((match.start(), match.end()))

    # 提取正向詞：去掉所有否定 span 後的剩餘文本
    positive_parts = []
    last_end = 0
    for start, end in sorted(negated_spans):
        if last_end < start:
            positive_parts.append(query[last_end:start])
        last_end = end
    if last_end < len(query):
        positive_parts.append(query[last_end:])

    positive_text = " ".join(positive_parts).strip()
    # 進一步清理常見非搜尋字詞與標點
    positive_text = re.sub(r"[，,。！!？?；;、]+", " ", positive_text)
    positive_text = re.sub(r"\s+", " ", positive_text).strip()

    raw_terms = [t.strip() for t in positive_text.split() if t.strip()]
    positive_terms = []
    for t in raw_terms:
        # 移除領頭的助詞（例如從 "不要後宮的異世界" 剩下的 "的異世界" 中移除 "的"）
        t = re.sub(r"^(的|之類的|之外|之外的|及|與|和)", "", t)
        if t:
            positive_terms.append(t)

    return QueryIntent(
        positive_terms=positive_terms,
        hard_exclusions=hard_exclusions,
        soft_exclusions=soft_exclusions,
        sanitized_bm25_query=" ".join(positive_terms),
        boolean_query=_build_boolean_query(positive_terms, hard_exclusions),
    )


def _build_boolean_query(
    positive_terms: List[str],
    hard_exclusions: List[NegationConstraint],
) -> str:
    """構建布林邏輯查詢表達式。"""
    if not positive_terms:
        return ""

    positive_part = "(" + " ".join(positive_terms) + ")"

    if hard_exclusions:
        # 每個 hard exclusion 都用 AND NOT 連接
        exclusion_terms = [exc.term for exc in hard_exclusions]
        exclusion_part = " AND NOT ".join(f'("{t}")' for t in exclusion_terms)
        return f'{positive_part} AND NOT ({exclusion_part})'

    return positive_part


def sanitize_search_terms(search_terms: str) -> str:
    """
    清洗 search_terms：移除否定修飾詞及其後面的被否定詞彙。
    確保送入 BM25 的字串不含否定詞。
    """
    if not search_terms:
        return ""

    cleaned = search_terms
    # 按照順序移除匹配項，先處理長的 pattern
    patterns = sorted(_SOFT_NEGATION_PATTERNS + _HARD_NEGATION_PATTERNS, key=len, reverse=True)
    
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    # 移除殘留的否定前綴
    for prefix in _NEGATION_PREFIXES:
        if prefix in cleaned:
            cleaned = cleaned.replace(prefix, " ")

    # 移除常見的連結助詞結尾（如：不要後宮"的"、排除BL"之類的"）
    # 但只在它們處於詞末或空格前時移除
    cleaned = re.sub(r"的(\s|$)", " ", cleaned)
    cleaned = re.sub(r"之類的(\s|$)", " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_query_intent(
    parse_result: QueryParseResult,
    user_query: str,
) -> QueryIntent:
    """
    從 LLM 解析結果建立完整的 QueryIntent。

    優先使用 LLM 回傳的 query_intent（若存在），
    否則從 criteria 中的 is_negative 欄位 + regex fallback 重建。
    """
    # 情況 1：LLM 已回傳結構化 query_intent
    if parse_result.query_intent and parse_result.query_intent.positive_terms:
        intent = parse_result.query_intent
        # 確保 sanitized_bm25_query 和 boolean_query 已設定
        if not intent.sanitized_bm25_query:
            intent.sanitized_bm25_query = " ".join(intent.positive_terms)
        if not intent.boolean_query:
            intent.boolean_query = _build_boolean_query(
                intent.positive_terms, intent.hard_exclusions
            )
        print(f"[QueryPreprocessor] 使用 LLM 結構化意圖")
        _log_intent(intent)
        return intent

    # 情況 2：從 criteria 的 is_negative 欄位 + regex 重建
    print(f"[QueryPreprocessor] LLM 未回傳 query_intent，進入 fallback 重建")

    # 2a. 從 criteria 提取 negative semantic terms
    hard_exclusions: List[NegationConstraint] = []
    for criteria in parse_result.criteria:
        if criteria.name == "semantic_similarity" and criteria.is_negative:
            query_text = (criteria.parameters.query_text or "").strip()
            if query_text:
                hard_exclusions.append(NegationConstraint(
                    term=query_text,
                    strength="hard",
                    weight=-1.0,
                    reason=f"from criteria is_negative: {criteria.description or query_text}"
                ))

    # 2b. 用 regex 從原始查詢補充
    regex_intent = _regex_fallback_extract(user_query)

    # 合併（去重）
    existing_terms = {exc.term for exc in hard_exclusions}
    for exc in regex_intent.hard_exclusions:
        if exc.term not in existing_terms:
            hard_exclusions.append(exc)
            existing_terms.add(exc.term)

    soft_exclusions = regex_intent.soft_exclusions

    # 2c. 建立正向詞列表
    # 從 search_terms 清洗否定詞
    sanitized = sanitize_search_terms(parse_result.search_terms or user_query)
    positive_terms = [t for t in sanitized.split() if t.strip()]

    # 加入 generated_keywords 作為正向補充
    existing_positive = set(positive_terms)
    excluded_terms = {exc.term for exc in hard_exclusions + soft_exclusions}
    for kw in parse_result.generated_keywords:
        if kw not in existing_positive and kw not in excluded_terms:
            positive_terms.append(kw)
            existing_positive.add(kw)

    intent = QueryIntent(
        positive_terms=positive_terms,
        hard_exclusions=hard_exclusions,
        soft_exclusions=soft_exclusions,
        sanitized_bm25_query=" ".join(positive_terms),
        boolean_query=_build_boolean_query(positive_terms, hard_exclusions),
    )

    _log_intent(intent)
    return intent


def _log_intent(intent: QueryIntent) -> None:
    """印出結構化意圖的 debug 資訊。"""
    print(f"[QueryPreprocessor] -- 結構化意圖解析結果 --")
    print(f"[QueryPreprocessor]   正向搜尋詞: {intent.positive_terms}")
    if intent.hard_exclusions:
        for exc in intent.hard_exclusions:
            print(f"[QueryPreprocessor]   [Hard Exclusion]: '{exc.term}' (AND NOT)")
    if intent.soft_exclusions:
        for exc in intent.soft_exclusions:
            print(f"[QueryPreprocessor]   [Soft Exclusion]: '{exc.term}' (weight={exc.weight})")
    print(f"[QueryPreprocessor]   淨化 BM25 查詢: \"{intent.sanitized_bm25_query}\"")
    print(f"[QueryPreprocessor]   布林表達式: {intent.boolean_query}")


def apply_negative_boost(
    scored_items: List[Dict[str, Any]],
    soft_exclusions: List[NegationConstraint],
    normalize_tags_fn,
) -> List[Dict[str, Any]]:
    """
    負權重機制 (Negative Boosting)：
    對命中軟排除詞的文件進行 BM25 扣分。

    Args:
        scored_items: 已計分的候選列表
        soft_exclusions: 軟排除約束
        normalize_tags_fn: tag 正規化函數

    Returns:
        經過負權重調整的候選列表（修改 score 和 breakdown）
    """
    if not soft_exclusions:
        return scored_items

    for result in scored_items:
        item = result["item"]
        book_tags = set(normalize_tags_fn(item.get("tags", [])))
        book_intro = (item.get("intro") or "").lower()
        book_name = (item.get("name") or "").lower()

        total_penalty = 0.0
        penalty_details = []

        for exc in soft_exclusions:
            term_lower = exc.term.lower()

            # 檢查 tag 命中
            tag_hit = any(
                term_lower in tag.lower() or tag.lower() in term_lower
                for tag in book_tags
            )

            # 檢查簡介命中
            intro_hit = term_lower in book_intro

            # 檢查書名命中
            name_hit = term_lower in book_name

            if tag_hit or intro_hit or name_hit:
                hit_sources = []
                if tag_hit:
                    hit_sources.append("tag")
                if intro_hit:
                    hit_sources.append("intro")
                if name_hit:
                    hit_sources.append("name")

                penalty = exc.weight  # 負值
                total_penalty += penalty
                penalty_details.append(
                    f"{exc.term}({','.join(hit_sources)})→{penalty:.2f}"
                )

        if total_penalty < 0:
            result["score"] += total_penalty  # 扣分
            result["breakdown"].append({
                "criteria": "negative_boost",
                "label": "負權重降權",
                "reason": f"軟排除命中: {'; '.join(penalty_details)}",
                "raw_score": total_penalty,
                "weighted_score": total_penalty,
                "is_filter": False,
            })

    return scored_items


def apply_hard_exclusions(
    scored_items: List[Dict[str, Any]],
    hard_exclusions: List[NegationConstraint],
    normalize_tags_fn,
) -> List[Dict[str, Any]]:
    """
    硬排除 (AND NOT)：命中任一硬排除詞的文件直接過濾掉。

    Args:
        scored_items: 候選列表
        hard_exclusions: 硬排除約束
        normalize_tags_fn: tag 正規化函數

    Returns:
        過濾後的候選列表
    """
    if not hard_exclusions:
        return scored_items

    filtered = []
    removed_count = 0

    for result in scored_items:
        item = result["item"]
        book_tags = set(normalize_tags_fn(item.get("tags", [])))
        excluded = False

        for exc in hard_exclusions:
            if any(
                exc.term in tag or tag in exc.term
                for tag in book_tags
            ):
                excluded = True
                break

        if excluded:
            removed_count += 1
        else:
            filtered.append(result)

    if removed_count > 0:
        print(
            f"[QueryPreprocessor] 硬排除過濾: {len(scored_items)} → {len(filtered)} "
            f"(移除 {removed_count})"
        )

    return filtered
