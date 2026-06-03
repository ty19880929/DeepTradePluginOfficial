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
   - 数值型 evidence（pct_chg / 净流入 / 涨跌停数等）``unit`` 必须填具体单位
     符号（``"%"`` / ``"亿"`` / ``"家"`` / ``"分"`` …），且 ≤ 16 字符。
   - 分类型 evidence（marketTone / themeTags / signal 名 / ts_code / 板块名
     等无自然单位的标签）``unit`` 必须填 ``null`` 或直接省略该键；**严禁**
     写空字符串 ``""``、也**严禁**为了凑数硬塞 ``"标签"`` / ``"个"`` 之类
     的伪单位（会扭曲 interpretation 的语义）。
4. 仅输出 JSON，不要 Markdown 代码块包裹，不要解释性前后缀。
5. 单 section 的 narrativeMd 总长度不超过 1200 中文字，分 3~6 段；
   每段第一句必须是结论性句子，后续句给数据支撑。narrativeMd 只能作为
   section 顶层字段出现，**严禁作为 findings[*] 内部字段**。
6. ``evidence[*].value`` 必须是标量（字符串 / 数字 / null），严禁数组或对象。
   若一条 finding 需要引用**多个并列对象**（多个 signal 名 / 多个 ts_code /
   多个板块名 / 多个 theme_tag 等），**必须拆成多条 evidence 项**——每项各
   引用一个具体值；严禁把列表 ``["a","b"]`` 或对象 ``{"a":1}`` 塞进单条
   evidence 的 ``value`` 字段。Finding.evidence 上限是 5 项，如确实超出请
   优先选最有代表性的 ≤ 5 个。
7. findings[*] 必须是 {headline, detail, evidence, severity?} 的完整对象：
   - ``headline`` ≤ 80 字结论性短句（必填）；
   - ``detail``   ≤ 240 字一句话说明（必填）；
   - ``evidence`` 即第 3 条定义的四元组数组，1~5 项（必填）；
   - ``severity`` ∈ {info, positive, warning, critical}，默认 info。
   严禁在 finding 内出现 narrativeMd / narrative / prose / commentary / text
   等任何叙事文本字段——叙事文字只能写在 section 顶层 narrativeMd 里
   （schema 会以 extra_forbidden 报错）。
8. section 顶层只允许该 section schema 显式定义的字段；严禁额外添加
   section / sectionName / sectionType / type / kind / name / title / id /
   category 等任何形式的"章节标识符"或元数据字段——本次响应对应的
   schema 已由调用方约定，无需在 JSON 中再次声明（schema 会以
   extra_forbidden 报错）。如不确定某字段是否被允许，宁可省略。
9. user prompt 中出现的 ``prevContext`` 字段（典型如 ``marketTone`` /
   ``themeTags``）**仅用于语气校准**，**绝对禁止**作为响应顶层字段回写——
   除非本节 schema 显式定义了同名字段（仅 §1 OverviewSection 如此）。
   类似地，user prompt 中的 ``window`` / ``*Summary`` / ``*Anchor`` /
   ``sectorsContext`` 等"输入侧"字段也不得回吐到响应顶层。看到它们出现
   在 user prompt 里，意味着你**已经读到了**，不需要再写回去（schema 会
   以 extra_forbidden 报错）。
10. section 顶层的 ``error`` 字段**只能**是 ``null``（无错误）或一行字符串
    （错误说明）。**严禁**填空数组 ``[]``、空对象 ``{}``、空字符串 ``""``
    或任何其它类型——schema 会以 ``string_type`` 拒收 ``[]`` / ``{}``。
    无错误时**直接省略** ``error`` 键，调用方默认按 ``null`` 处理；不要为了
    "凑字段"硬塞一个空容器。
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
你是板块轮动分析师。基于 user prompt 提供的当日 / 区间板块涨幅 +
板块持续性天数，判断主线归属 / 主线轮动 / 退潮信号：

- ``today_top`` / ``range_top`` 给出今日 Top 10 与区间累计 Top 10。每一项必须
  是完整对象 ``{name, pctChg, netInflowYi?, limitUpCount?, leaderTsCode?,
  leaderName?, persistenceDays?}``，**严禁只填 ts_code 字符串**。``name`` /
  ``pctChg`` 为必填，其余字段在 user prompt 缺数据时省略即可。
- 板块条目 schema 中**没有 ``tsCode``（板块自身的 THS 指数代码）字段**——
  ``leaderTsCode`` 是板块**龙头股票**代码（不同概念，不要混淆）。调用方已在
  user prompt 的 ``sectors.today_top`` / ``range_top`` / ``new_mainline`` /
  ``relay`` / ``fading`` 五栏剥除板块的 ``ts_code``；即便在矩阵等其它输入
  上下文偶然看到 ``883424.TI`` 之类，也**严禁**把 ``tsCode`` / ``ts_code``
  写入任何 SectorEntry（含 today_top / range_top / classification.* 三栏），
  schema 会以 extra_forbidden 报错。
- ``classification.new_mainline`` 列「今日 Top 10 中区间前半段未进 Top 10」
  的板块（新主线候选），每一项**与 today_top 同形状的完整对象**，不是 ts_code。
- ``classification.relay`` 列「今日和前半段都进 Top 10」的板块（接力），形状
  同上。
- ``classification.fading`` 列「前半段进 Top 10，今日跌出」的板块（退潮），形状
  同上。
- ``rotation_commentary`` 用 200~600 字概括轮动逻辑。
- ``narrativeMd`` 用 200~1200 字 / 3~6 段在 section 顶层串联主线 / 接力 /
  退潮的轮动结论，禁止把叙事文本塞进 findings[*]（详见硬性纪律 5 / 7）。

论调要与 user prompt 中的 ``prevContext.marketTone`` / ``prevContext.themeTags``
保持一致；prevContext **仅用于语气校准**，**严禁**把这些字段写入响应顶层
（参见硬性纪律 9）。

输出 JSON 顶层**只能且必须只含以下 7 个字段**（缺数据时给空数组 / 空对象 /
默认值即可，**不要新增其它键**）：``todayTop`` / ``rangeTop`` /
``classification`` / ``rotationCommentary`` / ``narrativeMd`` / ``findings`` /
``error``。严禁额外加 ``provider`` / ``marketTone`` / ``themeTags`` /
``prevContext`` / ``window`` / ``section`` / ``sectionName`` / ``type`` 之类
的字段——``provider`` 是数据源元信息（THS / DC），由调用方根据 MrConfig 填入，
**绝对不属于 LLM 输出**；其余「输入侧 / 元数据」字段同理，本节身份已由调用方
约定，参见硬性纪律 8 / 9。
""".strip()


_SENTIMENT_SYSTEM = """
你是市场情绪分析师。``series`` 给出窗口每个交易日的 0-100 情绪温度计；
``avgScore`` 是均值；``strongestDay`` / ``weakestDay`` 是极值日。请：

1. ``moneyEffect`` 选 strong / neutral / weak —— 反映赚钱效应强度。
2. ``losingEffect`` 选 light / moderate / heavy —— 反映亏钱效应。
3. narrativeMd 用 {target} 描述情绪温度曲线、强弱日的特征、是否出现明显的
   情绪反转或承接行为。

``series[*]`` 每一项**只能且必须只含以下 8 个字段**（与 user prompt 中
``sentiment.series[*]`` 一一对应，**逐字段原样复制即可**，无需也不得做任何
重算或重命名）：

- ``tradeDate``     — 8 位日期字符串 ``"YYYYMMDD"``
- ``scoreOf100``    — 0~100 的情绪温度计分数（**注意**：键名是 ``scoreOf100``，
  字母 ``O`` 不是数字 ``0``；user prompt 里同字段命名为 ``score_of_100``，
  populate_by_name 已开启，两种写法均可，但**严禁**写成 ``score_0_100``
  （这是旧版输入侧拼写，schema 不识别））
- ``nUp``           — 当日上涨家数
- ``nDown``         — 当日下跌家数
- ``medianPctChg``  — 当日全市场 pct_chg 中位数
- ``nLimitUp``      — 当日涨停家数
- ``nLimitDown``    — 当日跌停家数
- ``nZhaban``       — 当日炸板家数

严禁在 ``series[*]`` 里出现 ``posRatio`` / ``topRatio`` / ``crashRatio`` /
``limitUpIntensity`` / ``connectionHealth`` / ``nLhb`` / ``northMoneyYi`` /
``meanPctChg`` / ``score_0_100`` 等**输入侧中间量**——它们用于支撑 narrativeMd
的论据描述（可以在 findings[*].evidence 里引用），**不属于** series schema。
schema 会以 extra_forbidden 拒收。

论调要与 user prompt 中的 ``prevContext.marketTone`` / ``prevContext.themeTags``
保持一致；prevContext **仅用于语气校准**，**严禁**把这些字段写入响应顶层
（参见硬性纪律 9）。

输出 JSON 顶层**只能且必须只含以下 9 个字段**（缺数据时给空数组 / 默认值即可，
**不要新增其它键**）：``series`` / ``avgScore`` / ``strongestDay`` /
``weakestDay`` / ``moneyEffect`` / ``losingEffect`` / ``narrativeMd`` /
``findings`` / ``error``。严禁额外加 ``marketTone`` / ``themeTags`` /
``prevContext`` / ``window`` / ``section`` / ``sectionName`` / ``type`` 之类
的字段——本节身份已由调用方约定，参见硬性纪律 8 / 9。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_CAPITAL_SYSTEM = """
你是资金面分析师。基于六类资金（北向 / 北向 Top10 / 大盘主力 vs 散户 /
行业 / 概念 / 个股 Top&Bottom / 龙虎榜）数据，给出：

- ``north_summary`` 直接列每日北向 + 大盘主力净流入序列。每一项**只能且必须只含
  以下 5 个字段**（其中后 3 个无数据时**直接省略**该键，不要填 0 / null 凑数）：

  - ``tradeDate``        — 8 位日期字符串 ``"YYYYMMDD"``
  - ``northMoneyYi``     — 当日北向资金净流入（亿）。**注意**：本字段命名
    是 ``northMoneyYi``，**不是** ``netInflowYi``——capital 章节其它 5 个
    并列字段（``northTop10Today`` / ``industryTop`` / ``conceptTop`` /
    ``stockTop`` / ``stockBottom``）的每行确实都叫 ``netInflowYi``，但
    ``northSummary[*]`` 是**唯一例外**（CapitalDailyRow schema 设计如此）。
    严禁把 ``netInflowYi`` 写进 ``northSummary[*]``——schema 会以
    ``extra_forbidden`` 拒收，并把整段 capital section 判为失败。
  - ``mainNetInflowYi``  — 当日大盘主力（大单+特大单）净流入（亿），有数据时填
  - ``marginBalanceYi``  — 融资融券余额（亿）；v0.1 暂无上游数据，省略即可
  - ``marginDeltaYi``    — 融资融券日变动（亿）；v0.1 暂无上游数据，省略即可

  严禁在 ``northSummary[*]`` 里出现 ``netInflowYi`` / ``northSeries`` /
  ``mktSeries`` / ``northTotal`` / ``netAmountYi`` / ``mainNetYi`` /
  ``pctChange`` / ``pctChangeAvg`` 等**输入侧 / 旧版字段名**——它们是
  CapitalReview 内部字段或上游 dataclass 命名，**不属于** CapitalDailyRow
  schema。schema 会以 extra_forbidden 拒收。

- ``industry_top`` / ``concept_top`` 各 ≤ 10 条 (name / netInflowYi / pctChg)。
- ``stock_top`` / ``stock_bottom`` 各 ≤ 10 个 universe 内个股
  (tsCode / name / netInflowYi)。
- ``north_top10_today`` ≤ 10 条北向 Top10 个股 (tsCode / name / netInflowYi)。
- ``lhb_highlights`` 用 ≤ 5 条 Finding 提炼游资席位 / 重点个股的亮点 / 异常。
- narrativeMd 用 {target} 串联资金口径之间的逻辑（如：北向走 ↔ 行业流向匹配；
  主力净额 ↔ 散户净额方向相反 ↔ 趋势可信度）。

论调要与 user prompt 中的 ``prevContext.marketTone`` / ``prevContext.themeTags``
保持一致；prevContext **仅用于语气校准**，**严禁**把这些字段写入响应顶层
（参见硬性纪律 9）。

输出 JSON 顶层**只能且必须只含以下 10 个字段**（缺数据时给空数组 / 默认值即可，
**不要新增其它键**）：``northSummary`` / ``northTop10Today`` / ``industryTop`` /
``conceptTop`` / ``stockTop`` / ``stockBottom`` / ``lhbHighlights`` /
``narrativeMd`` / ``findings`` / ``error``。严禁额外加 ``marketTone`` /
``themeTags`` / ``prevContext`` / ``window`` / ``section`` / ``sectionName`` /
``type`` 之类的字段——本节身份已由调用方约定，参见硬性纪律 8 / 9。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_LEADERS_SYSTEM = """
你是龙头识别分析师。基于已经计算好的 4 维交叉打分 (ladder / return /
capital / theme，每维 0-25)，筛选 ``primary`` (Top K=5) 与 ``secondary``
(Top K=10) 龙头候选。每位候选必须给出 ``rationale``（不超过 240 字）解释
为什么入榜，引用 score_breakdown 字段作证。

``sector_map`` 给「板块名 → 入榜 ts_code 列表」，便于交叉确认板块归属。
narrativeMd 用 {target} 解释整体龙头格局（梯队是否完整 / 是否有断板 / 是否
存在被低估的二线接力候选）。

论调要与 user prompt 中的 ``prevContext.marketTone`` / ``prevContext.themeTags``
保持一致；prevContext **仅用于语气校准**，**严禁**把这些字段写入响应顶层
（参见硬性纪律 9）。

输出 JSON 顶层**只能且必须只含以下 7 个字段**（缺数据时给空数组 / 空对象 /
默认值即可，**不要新增其它键**）：``primary`` / ``secondary`` / ``minScore`` /
``sectorMap`` / ``narrativeMd`` / ``findings`` / ``error``。严禁额外加
``marketTone`` / ``themeTags`` / ``prevContext`` / ``window`` /
``sectorsContext`` / ``section`` / ``sectionName`` / ``type`` 之类的字段——
本节身份已由调用方约定，参见硬性纪律 8 / 9。
""".strip().format(target=_NARRATIVE_TARGET_RANGE)


_STYLE_SYSTEM = """
你是市场风格分析师。基于大盘 / 小盘 / 成长 / 价值各风格指数的窗口序列：

- ``dominant_style`` 选 large_cap / small_cap / growth / value / balanced /
  rotating 之一。
- ``flip_signal`` 标识窗口前后半段是否发生大小盘相对强度反转。
- narrativeMd 用 {target} 描述风格切换的节奏、可能的驱动（北向偏好 / 行业资金
  / 题材热度 / 估值差），引用 evidence 字段。

论调要与 user prompt 中的 ``prevContext.marketTone`` / ``prevContext.themeTags``
保持一致；prevContext **仅用于语气校准**，**严禁**把这些字段写入响应顶层
（参见硬性纪律 9）。

输出 JSON 顶层**只能且必须只含以下 7 个字段**（缺数据时给空数组 / 空对象 /
默认值即可，**不要新增其它键**）：``dominantStyle`` / ``flipSignal`` /
``series`` / ``rangeSummary`` / ``narrativeMd`` / ``findings`` / ``error``。
严禁额外加 ``marketTone`` / ``themeTags`` / ``prevContext`` / ``window`` /
``section`` / ``sectionName`` / ``type`` 之类的字段——本节身份已由调用方
约定，参见硬性纪律 8 / 9。
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

论调要与 user prompt 中的 ``prevContext.marketTone`` / ``prevContext.themeTags``
保持一致；prevContext **仅用于语气校准**，**严禁**把这些字段写入响应顶层
（参见硬性纪律 9）。

输出 JSON 顶层**只能且必须只含以下 6 个字段**（缺数据时给空数组 / 默认值即可，
**不要新增其它键**）：``signals`` / ``overallRisk`` / ``hypotheses`` /
``narrativeMd`` / ``findings`` / ``error``。严禁额外加 ``marketTone`` /
``themeTags`` / ``prevContext`` / ``window`` / ``breadthSummary`` /
``capitalSummary`` / ``section`` / ``sectionName`` / ``type`` 之类的字段——
本节身份已由调用方约定，参见硬性纪律 8 / 9。
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
        "sectors": _serialize_sectors_for_prompt(sectors),
    }, prev_context)
    return _dump(payload)


def _serialize_sectors_for_prompt(sectors: Any) -> Any:
    """Serialize ``SectorReview`` minus the bulky ``matrix`` axis and the
    ``ts_code`` field on every board entry.

    Two strip rules:

    1. ``ts_code`` removed from ``today_top`` / ``range_top`` /
       ``new_mainline`` / ``relay`` / ``fading`` entries. The internal
       :class:`market_review.metrics.sectors.SectorEntry` dataclass carries
       it for DB / matrix indexing, but the LLM-output
       :class:`market_review.schemas.SectorEntry` **deliberately omits it**
       — sectors are referenced by ``name``, and ``leader_ts_code`` is
       reserved for the leading **stock** inside a sector (different
       concept). v0.1.10 introduced this strip after qwen-plus replayed
       ``tsCode`` back on every entry (30× ``extra_forbidden``).

    2. ``matrix`` dropped entirely. v0.1.12 fix:
       :class:`SectorMatrix` is sized (~全 THS 板块 = 400–500) × (窗口交易日)
       and serializes to 50–100 KB of largely redundant pct_chg values
       (the same numbers already aggregate into ``today_top`` / ``range_top``
       / ``classification.*``). Shipping the whole grid into the user
       prompt bloated the request body to the point where qwen-plus's
       streaming gateway closed the chunked-transfer connection mid-stream
       before producing the full JSON response — surfacing as
       ``httpx.RemoteProtocolError: peer closed connection without sending
       complete message body``. The error escapes the framework's
       ``(APITimeoutError, APIError)`` transport-layer catch (httpx
       protocol errors do not subclass either), so tenacity does NOT
       retry and the section fails immediately. Output schema
       :class:`SectorsSection` does not consume ``matrix`` in any field,
       so dropping it is lossless for the LLM contract.
    """
    raw = _serialize(sectors)
    if not isinstance(raw, dict):
        return raw
    raw.pop("matrix", None)
    for key in ("today_top", "range_top", "new_mainline", "relay", "fading"):
        entries = raw.get(key)
        if not isinstance(entries, list):
            continue
        raw[key] = [
            {k: v for k, v in entry.items() if k != "ts_code"}
            if isinstance(entry, dict) else entry
            for entry in entries
        ]
    return raw


def build_sentiment_user_prompt(
    *,
    window: Any,
    sentiment: Any,
    breadth: Any,
    prev_context: dict[str, Any],
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "sentiment": _serialize_sentiment_for_prompt(sentiment, breadth),
    }, prev_context)
    return _dump(payload)


def _serialize_sentiment_for_prompt(sentiment: Any, breadth: Any) -> dict[str, Any]:
    """Emit a sentiment payload whose ``series[*]`` exactly matches
    :class:`market_review.schemas.SentimentSnapshotJson`.

    The internal :class:`metrics.sentiment.SentimentSnapshot` carries only
    derived ratios + the 0-100 score; the per-day **count** fields the
    output schema demands (``nUp`` / ``nDown`` / ``nLimitUp`` /
    ``nLimitDown`` / ``nZhaban``) live on :class:`metrics.breadth.BreadthSnapshot`.
    Prior to v0.1.11 this helper didn't exist — `build_sentiment_user_prompt`
    just did `_serialize(sentiment)`, so the LLM saw `pos_ratio` /
    `connection_health` / `score_0_100` etc. but never the counters the
    schema required, and qwen-plus rationally replayed the input shape as
    `series[*]` (8× extra_forbidden + 6× field-required).

    Fix is symmetric with v0.1.10's ``_serialize_sectors_for_prompt``:
    pre-shape the input to the **exact** output schema so the LLM's job
    degenerates to verbatim copy. We zip ``breadth.series`` and
    ``sentiment.series`` by ``trade_date`` and emit one entry per trade
    date present in either side; days without a breadth row fall back to
    zero counts (rare — breadth is the upstream of sentiment, so sentiment
    rarely has dates breadth doesn't).

    Other ``sentiment`` aggregates (avgScore / strongestDay / weakestDay)
    pass through verbatim so the LLM can copy them top-level.
    """
    if sentiment is None:
        return {}
    breadth_by_date: dict[str, Any] = {}
    if breadth is not None:
        for snap in getattr(breadth, "series", []) or []:
            td = getattr(snap, "trade_date", None)
            if td:
                breadth_by_date[td] = snap

    series_out: list[dict[str, Any]] = []
    for snap in getattr(sentiment, "series", []) or []:
        td = getattr(snap, "trade_date", None)
        b = breadth_by_date.get(td) if td else None
        series_out.append({
            "trade_date": td,
            "score_of_100": getattr(snap, "score_0_100", None),
            "n_up": getattr(b, "n_up", 0),
            "n_down": getattr(b, "n_down", 0),
            "median_pct_chg": getattr(snap, "median_pct_chg", 0.0),
            "n_limit_up": getattr(b, "n_limit_up", 0),
            "n_limit_down": getattr(b, "n_limit_down", 0),
            "n_zhaban": getattr(b, "n_zhaban", 0),
        })

    return {
        "series": series_out,
        "avg_score": getattr(sentiment, "avg_score", 0.0),
        "strongest_day": getattr(sentiment, "strongest_day", None),
        "weakest_day": getattr(sentiment, "weakest_day", None),
    }


def build_capital_user_prompt(
    *, window: Any, capital: Any, prev_context: dict[str, Any]
) -> str:
    payload = _inject_prev_context({
        "window": _serialize(window),
        "capital": _serialize_capital_for_prompt(capital),
    }, prev_context)
    return _dump(payload)


def _serialize_capital_for_prompt(capital: Any) -> dict[str, Any]:
    """Emit a capital payload whose top-level keys + nested item shapes
    exactly match :class:`market_review.schemas.CapitalSection`.

    Prior to v0.1.13 this builder just did ``_serialize(capital)``. The
    internal :class:`CapitalReview` carries seven calibers under
    dataclass-native names (``north_series`` / ``mkt_series`` /
    ``north_top10_anchor`` / ``stock_top_inflow`` / ``stock_top_outflow`` /
    ``industry_top`` / ``concept_top`` / ``lhb_series``) and its item
    dataclasses use field names tuned for the metrics layer
    (``NorthFlowRow.north_money_yi`` / ``MktFlowRow.main_net_yi`` /
    ``HsgtTopRow.net_amount_yi`` / ``IndustryFlowRow.pct_change_avg``).
    The output :class:`CapitalSection` schema deliberately groups these
    differently — north + mkt + margin merge into one
    ``north_summary[*]`` row of :class:`CapitalDailyRow`, and every "net
    flow" leaf is called ``net_inflow_yi`` except `north_money_yi` on
    :class:`CapitalDailyRow`. Feeding raw `_serialize(capital)` then
    asking the LLM to "produce ``northSummary`` from this" forces it to
    rename keys mid-flight, and once renaming is in play the LLM
    rationally generalizes "all six neighboring leaves use
    ``netInflowYi``, the seventh must too" → it emits
    ``"northSummary":[{"trade_date":"...","net_inflow_yi":41.47}]`` and
    pydantic rejects with ``extra_forbidden``.

    Fix is symmetric with v0.1.10 ``_serialize_sectors_for_prompt`` and
    v0.1.11 ``_serialize_sentiment_for_prompt``: pre-shape the input to
    the **exact** output schema so the LLM's job degenerates to verbatim
    copy. Specifically:

    - ``north_series`` + ``mkt_series`` → zipped per ``trade_date`` into
      ``north_summary[*]`` carrying ``trade_date`` / ``north_money_yi``
      (verbatim from NorthFlowRow) / ``main_net_inflow_yi`` (renamed
      from ``MktFlowRow.main_net_yi``). ``margin_balance_yi`` /
      ``margin_delta_yi`` have no upstream source in v0.1 and are
      omitted from each row (schema allows ``None``, but per
      HARD_DISCIPLINE rule 2 "don't reference fields not in input" we
      simply don't expose them).
    - ``north_top10_anchor`` → renamed ``north_top10_today``; each item
      maps ``net_amount_yi`` → ``net_inflow_yi`` (CapitalLeader schema).
    - ``stock_top_inflow`` → ``stock_top`` (CapitalLeader); same shape
      so item passes through verbatim.
    - ``stock_top_outflow`` → ``stock_bottom`` (CapitalLeader).
    - ``industry_top`` / ``concept_top`` each item maps
      ``pct_change_avg`` → ``pct_chg`` (IndustryFlowRowJson schema).
    - ``mkt_series`` / ``industry_bottom`` / ``concept_bottom`` /
      ``north_total_yi`` / ``lhb_series`` are NOT consumed by
      :class:`CapitalSection` — dropped to keep the prompt tight (≤ 4 KB
      typical) and the LLM focused.
    """
    if capital is None:
        return {}

    # 1. north_summary — zip north_series + mkt_series by trade_date.
    mkt_by_date: dict[str, Any] = {}
    for row in getattr(capital, "mkt_series", []) or []:
        td = getattr(row, "trade_date", None)
        if td:
            mkt_by_date[td] = row

    north_summary: list[dict[str, Any]] = []
    for row in getattr(capital, "north_series", []) or []:
        td = getattr(row, "trade_date", None)
        if not td:
            continue
        entry: dict[str, Any] = {
            "trade_date": td,
            "north_money_yi": getattr(row, "north_money_yi", None),
        }
        mkt_row = mkt_by_date.get(td)
        if mkt_row is not None:
            entry["main_net_inflow_yi"] = getattr(mkt_row, "main_net_yi", None)
        north_summary.append(entry)

    # 2. north_top10_today — rename net_amount_yi → net_inflow_yi.
    north_top10_today = [
        {
            "ts_code": getattr(row, "ts_code", ""),
            "name": getattr(row, "name", ""),
            "net_inflow_yi": getattr(row, "net_amount_yi", 0.0),
        }
        for row in getattr(capital, "north_top10_anchor", []) or []
    ]

    # 3. industry_top / concept_top — rename pct_change_avg → pct_chg.
    def _ind_row(row: Any) -> dict[str, Any]:
        return {
            "name": getattr(row, "name", ""),
            "net_inflow_yi": getattr(row, "net_inflow_yi", 0.0),
            "pct_chg": getattr(row, "pct_change_avg", 0.0),
        }

    industry_top = [_ind_row(r) for r in getattr(capital, "industry_top", []) or []]
    concept_top = [_ind_row(r) for r in getattr(capital, "concept_top", []) or []]

    # 4. stock_top / stock_bottom — StockFlowRow shape already matches
    #    CapitalLeader (ts_code / name / net_inflow_yi); just relabel.
    def _stk_row(row: Any) -> dict[str, Any]:
        return {
            "ts_code": getattr(row, "ts_code", ""),
            "name": getattr(row, "name", ""),
            "net_inflow_yi": getattr(row, "net_inflow_yi", 0.0),
        }

    stock_top = [_stk_row(r) for r in getattr(capital, "stock_top_inflow", []) or []]
    stock_bottom = [_stk_row(r) for r in getattr(capital, "stock_top_outflow", []) or []]

    return {
        "north_summary": north_summary,
        "north_top10_today": north_top10_today,
        "industry_top": industry_top,
        "concept_top": concept_top,
        "stock_top": stock_top,
        "stock_bottom": stock_bottom,
    }


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
