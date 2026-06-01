"""Pydantic schemas for market-review LLM section outputs (design §5.4 + §15).

Seven section models — one per LLM call in the §5.4.1 pipeline:

- :class:`OverviewSection`    — §1 大盘整体描述 (+ theme_tags / market_tone
  consumed by the other 6 sections)
- :class:`SectorsSection`     — §2 板块轮动 / 主线 / 接力 / 退潮
- :class:`SentimentSection`   — §3 情绪与赚钱效应 (LLM-side; numeric series
  from PR-3 metrics is passed in via prompt JSON)
- :class:`CapitalSection`     — §4 北向 / 行业 / 概念 / 龙虎榜 解读
- :class:`LeadersSection`     — §5 龙头梳理 (primary + secondary + rationale)
- :class:`StyleSection`       — §6 风格切换
- :class:`RiskOutlookSection` — §7 风险 + 次日 / 下周展望

Every section model inherits from :class:`SectionBase` for the common
``narrative_md`` + ``findings`` + ``error`` shape, and uses pydantic v2
``ConfigDict(extra="forbid")`` so any unexpected field (an LLM hallucination
or upstream schema drift) surfaces immediately as a :class:`ValidationError`
rather than being silently absorbed.

**Field naming convention** — internal Python attributes use snake_case;
the LLM-facing JSON wire format uses camelCase via ``Field(alias=...)``.
``populate_by_name=True`` on every model allows constructors / fixtures to
keep using snake_case while ``model_dump(by_alias=True)`` produces the
camelCase JSON the design §15 contract requires.

PR-5 will pull these into the :class:`ReviewReportSchema` root + add the
``MetricsBlock`` for structured numerics. Until then these are the
canonical persistence shape: ``mr_stage_results.response_json`` rows hold
one of these models per ``(run_id, section)``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

SectionName = Literal[
    "overview",
    "sectors",
    "sentiment",
    "capital",
    "leaders",
    "style",
    "risk_outlook",
]

WindowMode = Literal["day", "range"]

MarketTone = Literal[
    "强势普涨",
    "结构性上涨",
    "震荡分化",
    "结构性下跌",
    "弱势普跌",
    "止跌反弹",
    "高位见顶",
    "未知",
]

Severity = Literal["info", "positive", "warning", "critical"]


# ---------------------------------------------------------------------------
# Common building blocks (design §15.4)
# ---------------------------------------------------------------------------


def _camel(field_name: str) -> str:
    """snake_case → camelCase alias (no leading underscore handling needed)."""
    head, *tail = field_name.split("_")
    return head + "".join(part.capitalize() for part in tail)


class _StrictModel(BaseModel):
    """Shared base — strict extras + alias-by-name + camelCase wire format."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=_camel,
    )


class EvidenceItem(_StrictModel):
    """4-tuple proof a narrative claim leans on (design §15.4).

    ``field`` MUST name a key actually present in this section's user prompt
    (the LLM is forbidden from inventing field names — design §5.4.2).
    ``value`` is restricted to scalars (str / int / float / None); arrays /
    objects are explicitly disallowed via :class:`ConfigDict.extra="forbid"`
    on this model and the type annotation here.
    """

    field: str = Field(min_length=1, max_length=64)
    value: str | int | float | None = None
    unit: str = Field(min_length=1, max_length=16)
    interpretation: str = Field(min_length=1, max_length=120)


class Finding(_StrictModel):
    """Structured observation alongside the prose narrative (design §15.4)."""

    headline: str = Field(min_length=1, max_length=80)
    detail: str = Field(min_length=1, max_length=240)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=5)
    severity: Severity = "info"


class HeadlineMetric(_StrictModel):
    """Top-of-page core indicator (design §15.3).

    ``label`` is the human-readable name; ``unit`` is a short symbol like
    ``"亿"`` / ``"%"`` / ``"家"`` / ``"分"``. ``"none"`` allowed for
    categorical labels.
    """

    label: str = Field(min_length=1, max_length=24)
    value: str | int | float | None = None
    unit: str = Field(min_length=1, max_length=16)
    delta: float | None = None
    delta_unit: str | None = Field(default=None, max_length=16)
    interpretation: str | None = Field(default=None, max_length=80)


class SectionBase(_StrictModel):
    """Shared response envelope every section returns.

    ``narrative_md`` is allowed to be empty when the LLM call failed —
    the corresponding ``error`` field then carries the failure reason and
    PR-5 :class:`ReportMeta.failedSections` accumulates the section name.
    """

    narrative_md: str = Field(default="", max_length=2400)
    findings: list[Finding] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# §1 — Overview
# ---------------------------------------------------------------------------


class OverviewSection(SectionBase):
    """Establishes ``market_tone`` + ``theme_tags`` for later sections to
    align tone on (design §5.4.4)."""

    market_tone: MarketTone
    headline_metrics: list[HeadlineMetric] = Field(min_length=3, max_length=6)
    theme_tags: list[str] = Field(min_length=0, max_length=4)


# ---------------------------------------------------------------------------
# §2 — Sectors
# ---------------------------------------------------------------------------


class SectorEntry(_StrictModel):
    name: str = Field(min_length=1, max_length=40)
    pct_chg: float
    net_inflow_yi: float | None = None
    limit_up_count: int | None = None
    leader_ts_code: str | None = None
    leader_name: str | None = None
    persistence_days: int | None = None


class SectorClassification(_StrictModel):
    new_mainline: list[SectorEntry] = Field(default_factory=list)
    relay: list[SectorEntry] = Field(default_factory=list)
    fading: list[SectorEntry] = Field(default_factory=list)


class SectorsSection(SectionBase):
    provider: Literal["ths", "dc"] = "ths"
    today_top: list[SectorEntry] = Field(default_factory=list)
    range_top: list[SectorEntry] = Field(default_factory=list)
    classification: SectorClassification = Field(default_factory=SectorClassification)
    rotation_commentary: str = Field(default="", max_length=600)


# ---------------------------------------------------------------------------
# §3 — Sentiment
# ---------------------------------------------------------------------------


class SentimentSnapshotJson(_StrictModel):
    """LLM-side per-day snapshot. Same shape as PR-3
    :class:`market_review.metrics.sentiment.SentimentSnapshot` but expressed
    as a strict pydantic model so the LLM response stays self-consistent."""

    trade_date: str = Field(min_length=8, max_length=8)
    score_of_100: float = Field(ge=0, le=100)
    n_up: int = Field(ge=0)
    n_down: int = Field(ge=0)
    median_pct_chg: float
    n_limit_up: int = Field(ge=0)
    n_limit_down: int = Field(ge=0)
    n_zhaban: int = Field(ge=0)


class SentimentSection(SectionBase):
    series: list[SentimentSnapshotJson] = Field(default_factory=list)
    avg_score: float = Field(default=0.0, ge=0, le=100)
    strongest_day: str | None = None
    weakest_day: str | None = None
    money_effect: Literal["strong", "neutral", "weak"] = "neutral"
    losing_effect: Literal["light", "moderate", "heavy"] = "moderate"


# ---------------------------------------------------------------------------
# §4 — Capital
# ---------------------------------------------------------------------------


class CapitalLeader(_StrictModel):
    ts_code: str = Field(min_length=1, max_length=16)
    name: str = Field(min_length=1, max_length=40)
    net_inflow_yi: float


class IndustryFlowRowJson(_StrictModel):
    name: str = Field(min_length=1, max_length=40)
    net_inflow_yi: float
    pct_chg: float


class CapitalDailyRow(_StrictModel):
    trade_date: str = Field(min_length=8, max_length=8)
    north_money_yi: float | None = None
    main_net_inflow_yi: float | None = None
    margin_balance_yi: float | None = None
    margin_delta_yi: float | None = None


class CapitalSection(SectionBase):
    north_summary: list[CapitalDailyRow] = Field(default_factory=list)
    north_top10_today: list[CapitalLeader] = Field(default_factory=list)
    industry_top: list[IndustryFlowRowJson] = Field(default_factory=list)
    concept_top: list[IndustryFlowRowJson] = Field(default_factory=list)
    stock_top: list[CapitalLeader] = Field(default_factory=list)
    stock_bottom: list[CapitalLeader] = Field(default_factory=list)
    lhb_highlights: list[Finding] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §5 — Leaders
# ---------------------------------------------------------------------------


class LeaderCandidateJson(_StrictModel):
    ts_code: str = Field(min_length=1, max_length=16)
    name: str = Field(min_length=1, max_length=40)
    score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    industries: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    sector_top_hit: list[str] = Field(default_factory=list)
    ladder_height: int | None = None
    range_pct_chg: float | None = None
    cum_main_inflow_yi: float | None = None
    rationale: str = Field(default="", max_length=240)


class LeadersSection(SectionBase):
    primary: list[LeaderCandidateJson] = Field(default_factory=list)
    secondary: list[LeaderCandidateJson] = Field(default_factory=list)
    min_score: float = Field(default=50.0, ge=0, le=100)
    sector_map: dict[str, list[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# §6 — Style
# ---------------------------------------------------------------------------


DominantStyle = Literal[
    "large_cap", "small_cap", "growth", "value", "balanced", "rotating",
]


class StyleSeriesPointJson(_StrictModel):
    trade_date: str = Field(min_length=8, max_length=8)
    large_cap_ret: float
    small_cap_ret: float
    value_ret: float | None = None
    growth_ret: float | None = None
    big_to_small_ratio: float


class StyleSection(SectionBase):
    dominant_style: DominantStyle = "balanced"
    flip_signal: bool = False
    series: list[StyleSeriesPointJson] = Field(default_factory=list)
    range_summary: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# §7 — Risk + Outlook
# ---------------------------------------------------------------------------


RiskLevel = Literal["low", "moderate", "elevated", "high"]


class RiskSignalJson(_StrictModel):
    name: str = Field(min_length=1, max_length=40)
    triggered: bool
    severity: Literal["info", "warning", "critical"] = "info"
    detail: str = Field(min_length=1, max_length=240)
    affected_samples: list[str] = Field(default_factory=list, max_length=5)


class OutlookHypothesis(_StrictModel):
    """LLM-issued 可证伪 hypothesis for the next session / week (design §15.5)."""

    title: str = Field(min_length=1, max_length=60)
    rationale: str = Field(min_length=1, max_length=240)
    watch_points: list[str] = Field(min_length=1, max_length=5)
    fail_triggers: list[str] = Field(min_length=1, max_length=5)


class RiskOutlookSection(SectionBase):
    signals: list[RiskSignalJson] = Field(default_factory=list)
    overall_risk: RiskLevel = "moderate"
    hypotheses: list[OutlookHypothesis] = Field(min_length=1, max_length=4)


# ---------------------------------------------------------------------------
# Section name → schema map (pipeline.py + render.py consume this)
# ---------------------------------------------------------------------------


SECTION_SCHEMAS: dict[SectionName, type[SectionBase]] = {
    "overview": OverviewSection,
    "sectors": SectorsSection,
    "sentiment": SentimentSection,
    "capital": CapitalSection,
    "leaders": LeadersSection,
    "style": StyleSection,
    "risk_outlook": RiskOutlookSection,
}


# Ordered list — runner / pipeline iterate by index so ``overview`` (the
# theme-anchor section) is always first.
SECTION_ORDER: tuple[SectionName, ...] = (
    "overview", "sectors", "sentiment", "capital",
    "leaders", "style", "risk_outlook",
)
