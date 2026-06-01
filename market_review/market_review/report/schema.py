"""ReviewReportSchema — the website contract (design §15).

The website front-end consumes the ``summary.json`` this schema serializes;
any schema drift here is a breaking change unless coordinated with the
front-end via ``meta.schemaVersion``.

Design principles (§15.1) carried into code:

1. **Strict schemas** — every model has ``extra="forbid"``; unknown JSON
   fields raise :class:`ValidationError` rather than being silently
   absorbed.
2. **Single ``_extras`` escape hatch** — only the **root** model declares
   an explicit ``extras: dict`` slot (alias ``_extras``) for forward-compat.
   Adding a new field today requires either bumping ``schemaVersion`` and
   editing this module, or stuffing it through ``_extras``.
3. **camelCase wire format** — every multi-token field has a camelCase
   alias generated via :func:`_camel`. ``model_dump(by_alias=True)`` is
   how the canonical JSON is produced.
4. **Failed-section continuity** — a section that failed mid-pipeline
   still appears as its own subtree, with ``narrativeMd=""`` and the
   ``error`` field populated. ``meta.failedSections`` indexes which.
5. **Failure is uploaded too** — ``status`` admits the
   ``success / partial_failed / failed / cancelled`` quadrant; even
   ``failed`` runs serialize a valid :class:`ReviewReportSchema` so the
   website can show the failure card.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import (
    SCHEMA_VERSION,
    CapitalDailyRow,
    CapitalSection,
    LeadersSection,
    OverviewSection,
    RiskOutlookSection,
    SectionName,
    SectorsSection,
    SentimentSection,
    StyleSection,
    StyleSeriesPointJson,
    _camel,
)


__all__ = [
    "ReportStatus",
    "WindowMeta",
    "ReportMeta",
    "HeadlineMetric",  # re-export for convenience
    "ReportHeadline",
    "BreadthSnapshotJson",
    "IndexReturnJson",
    "SectorMatrixJson",
    "MetricsLeaderRow",
    "MetricsRiskSignal",
    "StyleSeriesJson",
    "MetricsBlock",
    "ReviewReportSchema",
    "SCHEMA_VERSION",
]


# ---------------------------------------------------------------------------
# Report-level enums + meta
# ---------------------------------------------------------------------------


ReportStatus = Literal["success", "partial_failed", "failed", "cancelled"]
MarketToneStr = Literal[
    "强势普涨", "结构性上涨", "震荡分化",
    "结构性下跌", "弱势普跌", "止跌反弹", "高位见顶", "未知",
]


class _ReportStrict(BaseModel):
    """Shared model_config — strict + alias-by-name + camelCase wire."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=_camel,
    )


class WindowMeta(_ReportStrict):
    """Time-window metadata for the run (design §15.3)."""

    mode: Literal["day", "range"]
    start: str = Field(min_length=8, max_length=8)
    end: str = Field(min_length=8, max_length=8)
    anchor: str = Field(min_length=8, max_length=8)
    n_days: int = Field(ge=1)
    trade_dates: list[str] = Field(default_factory=list)


class ReportMeta(_ReportStrict):
    """Top-of-report bookkeeping the website front-end keys most cards on."""

    title: str = Field(min_length=1, max_length=120)
    run_id: str = Field(min_length=1)
    window: WindowMeta
    status: ReportStatus
    failed_sections: list[SectionName] = Field(default_factory=list)
    sections_enabled: list[SectionName] = Field(default_factory=list)
    llm_provider: str = Field(default="", description="实际使用的 provider 名；空字符串 = 未知")
    schema_version: str = Field(default=SCHEMA_VERSION)
    plugin_version: str = Field(default="0.1.0")
    input_fingerprint: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="64-char sha256(规范化指标 JSON + window + plugin 版本) — 设计 §15.3",
    )
    generated_at: str = Field(..., description="ISO 8601, 含时区")
    error: str | None = Field(default=None, description="status != success 时填充")


# ---------------------------------------------------------------------------
# Headline card (design §15.3)
# ---------------------------------------------------------------------------


class HeadlineMetric(_ReportStrict):
    """One badge on the首屏 banner (设计 §15.3)."""

    label: str = Field(min_length=1, max_length=24)
    value: str | int | float | None = None
    unit: str = Field(min_length=1, max_length=16)
    delta: float | None = None
    delta_unit: str | None = Field(default=None, max_length=16)
    interpretation: str | None = Field(default=None, max_length=80)


class ReportHeadline(_ReportStrict):
    """首屏 — one-liner + market tone + core metric tiles + theme tags."""

    one_liner: str = Field(min_length=1, max_length=120)
    market_tone: MarketToneStr
    core_metrics: list[HeadlineMetric] = Field(min_length=0, max_length=8)
    theme_tags: list[str] = Field(default_factory=list, max_length=4)


# ---------------------------------------------------------------------------
# MetricsBlock sub-models (design §15.6 — structured numerics for charts)
# ---------------------------------------------------------------------------


class BreadthSnapshotJson(_ReportStrict):
    """Per-day breadth row — front-end charts read this directly."""

    trade_date: str
    n_total: int = Field(ge=0)
    n_up: int = Field(ge=0)
    n_down: int = Field(ge=0)
    n_flat: int = Field(ge=0)
    n_up5pct: int = Field(ge=0)
    n_down5pct: int = Field(ge=0)
    n_limit_up: int = Field(ge=0)
    n_limit_down: int = Field(ge=0)
    n_zhaban: int = Field(ge=0)
    # Ladder as string-keyed dict so JSON keys are valid (int keys can't
    # round-trip through json.dumps without coercion).
    up_ladder: dict[str, int] = Field(default_factory=dict)
    total_amount_yi: float
    index_returns: dict[str, float] = Field(default_factory=dict)


class IndexReturnJson(_ReportStrict):
    """Per-index series for the chart on the indices panel (§15.6).

    ``close_series`` / ``amount_series_yi`` are window-length arrays. v0.1
    PR-5 builder leaves them empty when the source data isn't passed in
    (the builder is pure — no DB reads). PR-6 runner will populate them
    from ``mr_index_daily`` reads if the design demands chart data.
    """

    ts_code: str
    name: str
    pct_chg_window: float = Field(default=0.0, description="区间累计涨幅 %")
    close_series: list[float] = Field(default_factory=list)
    amount_series_yi: list[float] = Field(default_factory=list)


class SectorMatrixJson(_ReportStrict):
    """Sector × date strength matrix — mirrors PR-3
    :class:`market_review.metrics.sectors.SectorMatrix`."""

    sectors: list[str] = Field(default_factory=list)
    sector_names: list[str] = Field(default_factory=list)
    trade_dates: list[str] = Field(default_factory=list)
    pct_chg: list[list[float]] = Field(default_factory=list)
    cum_pct_chg: list[float] = Field(default_factory=list)
    persistence_days: list[int] = Field(default_factory=list)


class MetricsLeaderRow(_ReportStrict):
    """Leader candidate row inside MetricsBlock — no ``rationale`` (design
    §15.6 keeps the metrics block free of LLM prose).

    Identical field shape to :class:`market_review.schemas.LeaderCandidateJson`
    minus ``rationale``; declared separately so adding LLM-derived fields to
    the section model doesn't accidentally bleed into the chart payload.
    """

    ts_code: str
    name: str
    score: float = Field(ge=0, le=100)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    industries: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    sector_top_hit: list[str] = Field(default_factory=list)
    ladder_height: int | None = None
    range_pct_chg: float | None = None
    cum_main_inflow_yi: float | None = None


class StyleSeriesJson(_ReportStrict):
    """Window-level风格 series + summary (design §15.6)."""

    series: list[StyleSeriesPointJson] = Field(default_factory=list)
    avg_big_to_small: float = 0.0
    avg_value_to_growth: float | None = None
    half_period_flip: bool = False


class MetricsRiskSignal(_ReportStrict):
    """Risk signal in :class:`MetricsBlock` — no LLM ``detail`` (design §15.6).

    Distinct from :class:`market_review.schemas.RiskSignalJson` which carries
    the LLM-translated ``detail`` string. The metrics-block version is the
    pure numeric distillation: a ``sample_count`` integer + capped
    ``samples_top_k`` ts_code list, suitable for chart filters and badges.
    """

    name: str = Field(min_length=1, max_length=40)
    triggered: bool
    severity: Literal["info", "warning", "critical"] = "info"
    sample_count: int = Field(ge=0)
    samples_top_k: list[str] = Field(default_factory=list, max_length=10)


class MetricsBlock(_ReportStrict):
    """All charts the front-end renders from PR-3 numerics — no LLM
    intermediation (design §15.6).
    """

    breadth_series: list[BreadthSnapshotJson] = Field(default_factory=list)
    index_returns: dict[str, IndexReturnJson] = Field(default_factory=dict)
    sector_matrix: SectorMatrixJson = Field(default_factory=SectorMatrixJson)
    capital_daily: list[CapitalDailyRow] = Field(default_factory=list)
    leader_table: list[MetricsLeaderRow] = Field(default_factory=list)
    style_series: StyleSeriesJson = Field(default_factory=StyleSeriesJson)
    risk_signals: list[MetricsRiskSignal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root model (design §15.7)
# ---------------------------------------------------------------------------


class ReviewReportSchema(BaseModel):
    """Root — ``await fetch(url).json()`` consumes this directly.

    ``extras`` (aliased to ``_extras`` for the JSON wire format) is the
    single forward-compat hatch declared at the root per design §15.1.3.
    Today's builder always emits ``{}``; PR-7+ can stuff new fields there
    without bumping ``schemaVersion`` for opaque additions.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        alias_generator=_camel,
    )

    meta: ReportMeta
    headline: ReportHeadline
    overview: OverviewSection
    sectors: SectorsSection
    sentiment: SentimentSection
    capital: CapitalSection
    leaders: LeadersSection
    style: StyleSection
    risk_outlook: RiskOutlookSection
    metrics: MetricsBlock
    extras: dict[str, Any] = Field(default_factory=dict, alias="_extras")
