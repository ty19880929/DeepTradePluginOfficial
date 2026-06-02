"""Prompt templates for the 7 LLM sections (design §5.4).

Two pieces per section:

- A static **system prompt** declaring the section's focus + the shared
  :data:`HARD_DISCIPLINE` block (anti-hallucination + JSON-only output).
- A **user prompt builder** that serializes the relevant PR-3 metric
  dataclasses into a compact JSON block the LLM reads. Earlier-section
  context (``theme_tags`` + ``market_tone`` + optionally the previous
  section's narrative tail) is injected so the 7 sections speak with a
  consistent voice (design §5.4.4).

The builders return raw strings; :mod:`market_review.pipeline` glues
``system`` + ``user`` together with the framework's
:meth:`LLMClient.complete_json`, which adds the schema validator on top.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from .schemas import OverviewSection, SectionName

# ---------------------------------------------------------------------------
# Hard discipline — every system prompt appends this verbatim (design §5.4.2)
# ---------------------------------------------------------------------------


HARD_DISCIPLINE = """
【硬性纪律】
1. 严禁使用外部搜索、新闻网站、公告网站、实时行情、社交媒体、机构观点
   或任何未在 user prompt 中提供的数据。
2. 严禁编造数据；不在输入中的字段一律不引用。
3. evidence 必须以 {field, value, unit, interpretation} 四元组形式给出，
   field 必须是本次 user prompt 中出现过的字段名（不能凭空起名）。
   evidence **只能**作为 ``findings[*].evidence`` 数组项出现；严禁在 section
   顶层另起一个 ``evidence`` 字段（schema 会以 extra_forbidden 报错）。
4. 仅输出 JSON，不要 Markdown 代码块包裹，不要解释性前后缀。
5. 单 section 的 narrativeMd 总长度不超过 1200 中文字，分 3~6 段；
   每段第一句必须是结论性句子，后续句给数据支撑。
6. value 必须是标量（字符串/数字/null），严禁数组或对象。
"""

_NARRATIVE_TARGET_RANGE = "200~1200 字 / 3~6 段"


# ---------------------------------------------------------------------------
# System prompts — one per section
# ---------------------------------------------------------------------------


_OVERVIEW_SYSTEM = """
你是 A 股盘后复盘分析师。你的任务是给出"今天/本周市场整体处于什么状态"的
高层总结：用 ``market_tone`` 给市场打一个语义标签（强势普涨 / 结构性上涨 /
震荡分化 / 结构性下跌 / 弱势普跌 / 止跌反弹 / 高位见顶 / 未知 之一），用
``headline_metrics`` 给 3~6 个最具代表性的统计量（每个含 label/value/unit），
``theme_tags`` 给 1~4 个本日 / 本周的核心主题词。这些 theme_tags 将传递给
后续 6 个 section 作为统一论调。narrativeMd 用 {target} 阐述结论。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_SECTORS_SYSTEM = """
你是板块轮动分析师。基于 user prompt 提供的当日 / 区间板块涨幅 + 板块持续性
+ 板块强度矩阵，判断主线归属 / 主线轮动 / 退潮信号：

- ``today_top`` / ``range_top`` 给出今日 Top 10 与区间累计 Top 10。每一项必须
  是完整对象 ``{name, pctChg, netInflowYi?, limitUpCount?, leaderTsCode?,
  leaderName?, persistenceDays?}``，**严禁只填 ts_code 字符串**。``name`` /
  ``pctChg`` 为必填，其余字段在 user prompt 缺数据时省略即可。
- ``classification.new_mainline`` 列「今日 Top 10 中区间前半段未进 Top 10」
  的板块（新主线候选），每一项**与 today_top 同形状的完整对象**，不是 ts_code。
- ``classification.relay`` 列「今日和前半段都进 Top 10」的板块（接力），形状
  同上。
- ``classification.fading`` 列「前半段进 Top 10，今日跌出」的板块（退潮），形状
  同上。
- ``rotation_commentary`` 用 200~600 字概括轮动逻辑。

请保持 system / user 中已注入的 theme_tags + market_tone 的论调一致。
""".strip()


_SENTIMENT_SYSTEM = """
你是市场情绪分析师。``series`` 给出窗口每个交易日的 0-100 情绪温度计；
``avg_score`` 是均值；``strongest_day`` / ``weakest_day`` 是极值日。请：

1. ``money_effect`` 选 strong / neutral / weak —— 反映赚钱效应强度。
2. ``losing_effect`` 选 light / moderate / heavy —— 反映亏钱效应。
3. narrativeMd 用 {target} 描述情绪温度曲线、强弱日的特征、是否出现明显的
   情绪反转或承接行为。

保持 theme_tags + market_tone 论调一致。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_CAPITAL_SYSTEM = """
你是资金面分析师。基于六类资金（北向 / 北向 Top10 / 大盘主力 vs 散户 /
行业 / 概念 / 个股 Top&Bottom / 龙虎榜）数据，给出：

- ``north_summary`` 直接列每日北向净流入序列。
- ``industry_top`` / ``concept_top`` 各 ≤ 10 条 (name / netInflowYi / pctChg)。
- ``stock_top`` / ``stock_bottom`` 各 ≤ 10 个 universe 内个股。
- ``lhb_highlights`` 用 ≤ 5 条 Finding 提炼游资席位 / 重点个股的亮点 / 异常。
- narrativeMd 用 {target} 串联资金口径之间的逻辑（如：北向走 ↔ 行业流向匹配；
  主力净额 ↔ 散户净额方向相反 ↔ 趋势可信度）。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_LEADERS_SYSTEM = """
你是龙头识别分析师。基于已经计算好的 4 维交叉打分 (ladder / return /
capital / theme，每维 0-25)，筛选 ``primary`` (Top K=5) 与 ``secondary``
(Top K=10) 龙头候选。每位候选必须给出 ``rationale``（不超过 240 字）解释
为什么入榜，引用 score_breakdown 字段作证。

``sector_map`` 给「板块名 → 入榜 ts_code 列表」，便于交叉确认板块归属。
narrativeMd 用 {target} 解释整体龙头格局（梯队是否完整 / 是否有断板 / 是否
存在被低估的二线接力候选）。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_STYLE_SYSTEM = """
你是市场风格分析师。基于大盘 / 小盘 / 成长 / 价值各风格指数的窗口序列：

- ``dominant_style`` 选 large_cap / small_cap / growth / value / balanced /
  rotating 之一。
- ``flip_signal`` 标识窗口前后半段是否发生大小盘相对强度反转。
- narrativeMd 用 {target} 描述风格切换的节奏、可能的驱动（北向偏好 / 行业资金
  / 题材热度 / 估值差），引用 evidence 字段。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_RISK_OUTLOOK_SYSTEM = """
你是风险与展望分析师。先解读 8 个内部风险信号（``signals``），给出：

- ``overall_risk`` 选 low / moderate / elevated / high。
- 每条 ``signal`` 的 detail 用一句话翻译指标含义。

然后给出 1~4 条 ``hypotheses``，每条含：
- ``title`` 简明判断；
- ``rationale`` 不超过 240 字论据；
- ``watch_points`` 列 1~5 个观察点；
- ``fail_triggers`` 列 1~5 个可证伪条件（"如果出现 X 就放弃该假设"）。

narrativeMd 用 {target} 总结风险 + 下个交易日 / 下周关注点。绝对禁止
"无法预测"等推诿语；用可观测的具体条件代替。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


SECTION_SYSTEM_PROMPTS: dict[SectionName, str] = {
    "overview": _OVERVIEW_SYSTEM,
    "sectors": _SECTORS_SYSTEM,
    "sentiment": _SENTIMENT_SYSTEM,
    "capital": _CAPITAL_SYSTEM,
    "leaders": _LEADERS_SYSTEM,
    "style": _STYLE_SYSTEM,
    "risk_outlook": _RISK_OUTLOOK_SYSTEM,
}


def system_prompt(section: SectionName) -> str:
    """Return ``SECTION_SYSTEM_PROMPTS[section]`` + the shared discipline block."""
    if section not in SECTION_SYSTEM_PROMPTS:
        raise KeyError(f"unknown section: {section!r}")
    return SECTION_SYSTEM_PROMPTS[section] + "\n\n" + HARD_DISCIPLINE.strip()


# ---------------------------------------------------------------------------
# Prev-context plumbing (theme_tags + market_tone from §1 → §2..§7)
# ---------------------------------------------------------------------------


def build_prev_context(overview: OverviewSection | None) -> dict[str, Any]:
    """Extract the lightweight context handed from §1 to §2..§7.

    Returns ``{}`` if overview failed (theme_tags / market_tone default to
    safe placeholders in the user prompt, see :func:`_inject_prev_context`).
    """
    if overview is None:
        return {}
    return {
        "marketTone": overview.market_tone,
        "themeTags": list(overview.theme_tags),
    }


def _inject_prev_context(payload: dict[str, Any], prev: dict[str, Any]) -> dict[str, Any]:
    """Merge prev_context under a fixed key so the LLM sees it consistently."""
    if prev:
        payload["prevContext"] = prev
    return payload


# ---------------------------------------------------------------------------
# User prompt builders — one per section
# ---------------------------------------------------------------------------


def build_overview_user_prompt(
    *,
    window: Any,
    breadth: Any,
    sentiment: Any,
    capital: Any,
    sectors: Any,
    leaders: Any,
    style: Any,
    risk: Any,
) -> str:
    payload = {
        "window": _serialize(window),
        "breadthSummary": _summarize_breadth(breadth),
        "sentimentSummary": _summarize_sentiment(sentiment),
        "capitalSummary": _summarize_capital(capital),
        "sectorsSummary": _summarize_sectors(sectors),
        "leadersSummary": _summarize_leaders(leaders),
        "styleSummary": _summarize_style(style),
        "riskSummary": _summarize_risk(risk),
    }
    return _dump(payload)


def build_sectors_user_prompt(
    *, window: Any, sectors: Any, prev_context: dict[str, Any]
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "sectors": _serialize(sectors),
    }, prev_context)
    return _dump(payload)


def build_sentiment_user_prompt(
    *, window: Any, sentiment: Any, prev_context: dict[str, Any]
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "sentiment": _serialize(sentiment),
    }, prev_context)
    return _dump(payload)


def build_capital_user_prompt(
    *, window: Any, capital: Any, prev_context: dict[str, Any]
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "capital": _serialize(capital),
    }, prev_context)
    return _dump(payload)


def build_leaders_user_prompt(
    *, window: Any, leaders: Any, sectors: Any, prev_context: dict[str, Any]
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "leaders": _serialize(leaders),
        "sectorsContext": _summarize_sectors(sectors),  # tighter context here
    }, prev_context)
    return _dump(payload)


def build_style_user_prompt(
    *, window: Any, style: Any, prev_context: dict[str, Any]
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "style": _serialize(style),
    }, prev_context)
    return _dump(payload)


def build_risk_outlook_user_prompt(
    *,
    window: Any,
    risk: Any,
    breadth: Any,
    capital: Any,
    prev_context: dict[str, Any],
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "risk": _serialize(risk),
        "breadthSummary": _summarize_breadth(breadth),
        "capitalSummary": _summarize_capital(capital),
    }, prev_context)
    return _dump(payload)


# ---------------------------------------------------------------------------
# Tight summarizers — used by overview (full context) + risk_outlook (cross-cut)
# ---------------------------------------------------------------------------


def _summarize_breadth(breadth: Any) -> dict[str, Any]:
    if breadth is None:
        return {}
    series = getattr(breadth, "series", [])
    return {
        "nDays": len(series),
        "medianUpCount": getattr(breadth, "median_up_count", 0),
        "strongestDay": getattr(breadth, "sentiment_extreme_day", (None, None))[0],
        "weakestDay": getattr(breadth, "sentiment_extreme_day", (None, None))[1],
        "anchor": _serialize(series[-1]) if series else None,
    }


def _summarize_sentiment(sentiment: Any) -> dict[str, Any]:
    if sentiment is None:
        return {}
    return {
        "avgScore": getattr(sentiment, "avg_score", 0.0),
        "strongestDay": getattr(sentiment, "strongest_day", None),
        "weakestDay": getattr(sentiment, "weakest_day", None),
        "anchor": _serialize(sentiment.series[-1]) if getattr(sentiment, "series", None) else None,
    }


def _summarize_capital(capital: Any) -> dict[str, Any]:
    if capital is None:
        return {}
    return {
        "northTotalYi": getattr(capital, "north_total_yi", 0.0),
        "industryTopAnchor": [
            _serialize(r) for r in getattr(capital, "industry_top", [])[:5]
        ],
        "industryBottomAnchor": [
            _serialize(r) for r in getattr(capital, "industry_bottom", [])[:3]
        ],
        "stockTopAnchor": [
            _serialize(r) for r in getattr(capital, "stock_top_inflow", [])[:5]
        ],
    }


def _summarize_sectors(sectors: Any) -> dict[str, Any]:
    if sectors is None:
        return {}
    return {
        "todayTop": [_serialize(e) for e in getattr(sectors, "today_top", [])[:10]],
        "rangeTop": [_serialize(e) for e in getattr(sectors, "range_top", [])[:10]],
        "newMainline": [_serialize(e) for e in getattr(sectors, "new_mainline", [])],
        "relay": [_serialize(e) for e in getattr(sectors, "relay", [])],
        "fading": [_serialize(e) for e in getattr(sectors, "fading", [])],
    }


def _summarize_leaders(leaders: Any) -> dict[str, Any]:
    if leaders is None:
        return {}
    return {
        "primary": [_serialize(c) for c in getattr(leaders, "primary", [])],
        "secondaryCount": len(getattr(leaders, "secondary", [])),
        "minScore": getattr(leaders, "min_score", 0.0),
    }


def _summarize_style(style: Any) -> dict[str, Any]:
    if style is None:
        return {}
    return {
        "dominantStyle": getattr(style, "dominant_style", None),
        "flipSignal": getattr(style, "flip_signal", False),
        "rangeSummary": getattr(style, "range_summary", {}),
    }


def _summarize_risk(risk: Any) -> dict[str, Any]:
    if risk is None:
        return {}
    signals = getattr(risk, "signals", [])
    triggered = [s for s in signals if getattr(s, "triggered", False)]
    return {
        "totalSignals": len(signals),
        "triggeredCount": len(triggered),
        "triggeredNames": [getattr(s, "name", "") for s in triggered],
    }


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Convert a dataclass / frozenset / tuple into JSON-friendly types."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, frozenset):
        return sorted(_serialize(v) for v in obj)
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # Fall back to repr for unknown types (e.g., datetime / numpy)
    return str(obj)


def _dump(payload: dict[str, Any]) -> str:
    """Compact JSON dump, deterministic key order, UTF-8 safe."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
