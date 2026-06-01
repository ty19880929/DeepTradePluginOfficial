"""build_review_report — pure assembly (design §15.8).

The runner (PR-6) computes status / metrics / sections; this function just
turns them into a :class:`ReviewReportSchema`. No IO, no DB, no LLM —
deterministic in / out, fully testable.

PR-3 dataclasses → MetricsBlock pydantic conversion is concentrated here.
Field shape preservation lives in tests (``tests/test_report_builder.py``).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from ..schemas import (
    CapitalDailyRow,
    CapitalSection,
    HeadlineMetric as SchemaHeadlineMetric,
    LeadersSection,
    OverviewSection,
    RiskOutlookSection,
    SectionBase,
    SectionName,
    SectorsSection,
    SentimentSection,
    StyleSection,
)
from .schema import (
    BreadthSnapshotJson,
    HeadlineMetric,
    IndexReturnJson,
    MetricsBlock,
    MetricsLeaderRow,
    MetricsRiskSignal,
    ReportHeadline,
    ReportMeta,
    ReportStatus,
    ReviewReportSchema,
    SectorMatrixJson,
    StyleSeriesJson,
    WindowMeta,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.breadth import BreadthReview
    from ..metrics.capital import CapitalReview
    from ..metrics.leaders import LeaderReview
    from ..metrics.risk import RiskReview
    from ..metrics.sectors import SectorReview
    from ..metrics.sentiment import SentimentReview
    from ..metrics.style import StyleReview
    from ..windows import Window


# Names for the 8 wide-base indices PR-2 syncs into mr_index_daily.
INDEX_NAMES: dict[str, str] = {
    "000001.SH": "上证综指",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "899050.BJ": "北证50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
}

CN_TZ = timezone(timedelta(hours=8))


def build_review_report(
    *,
    status: ReportStatus,
    window: Window,
    breadth: BreadthReview,
    sentiment: SentimentReview,
    capital: CapitalReview,
    sectors: SectorReview,
    leaders: LeaderReview,
    style: StyleReview,
    risk: RiskReview,
    sections: dict[SectionName, SectionBase],
    failed_sections: list[SectionName],
    run_id: str,
    llm_provider: str,
    plugin_version: str,
    input_fingerprint: str,
    generated_at: datetime | None = None,
    error: str | None = None,
) -> ReviewReportSchema:
    """Assemble :class:`ReviewReportSchema` from runner state + metric reviews.

    Field-by-field this is a series of small adapter functions; they're
    named ``_to_<target>`` and live just below this entry.

    ``sections`` must contain every section name in ``SECTION_ORDER``;
    failed sections are still present (carrying error + placeholder
    payload) so the root model passes its ``extra="forbid"`` validation.
    ``failed_sections`` is the explicit roster for the UI banner; the
    builder doesn't infer it from ``sections[*].error`` so the runner can
    differentiate "section failed" from "section disabled by user".
    """
    overview = _typed_section(sections, "overview", OverviewSection)
    sectors_section = _typed_section(sections, "sectors", SectorsSection)
    sentiment_section = _typed_section(sections, "sentiment", SentimentSection)
    capital_section = _typed_section(sections, "capital", CapitalSection)
    leaders_section = _typed_section(sections, "leaders", LeadersSection)
    style_section = _typed_section(sections, "style", StyleSection)
    risk_outlook = _typed_section(sections, "risk_outlook", RiskOutlookSection)

    meta = _build_meta(
        window=window,
        status=status,
        failed_sections=failed_sections,
        run_id=run_id,
        llm_provider=llm_provider,
        plugin_version=plugin_version,
        input_fingerprint=input_fingerprint,
        generated_at=generated_at,
        error=error,
    )
    headline = _build_headline(overview, window=window)
    metrics_block = _build_metrics_block(
        breadth=breadth, sectors=sectors, capital=capital,
        leaders=leaders, style=style, risk=risk,
    )

    return ReviewReportSchema(
        meta=meta,
        headline=headline,
        overview=overview,
        sectors=sectors_section,
        sentiment=sentiment_section,
        capital=capital_section,
        leaders=leaders_section,
        style=style_section,
        risk_outlook=risk_outlook,
        metrics=metrics_block,
        extras={},
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def _build_meta(
    *,
    window: Window,
    status: ReportStatus,
    failed_sections: list[SectionName],
    run_id: str,
    llm_provider: str,
    plugin_version: str,
    input_fingerprint: str,
    generated_at: datetime | None,
    error: str | None,
) -> ReportMeta:
    title = _build_title(window)
    win_meta = WindowMeta(
        mode=window.mode,
        start=window.start,
        end=window.end,
        anchor=window.anchor,
        n_days=window.n_days,
        trade_dates=list(window.trade_dates),
    )
    ts = (generated_at or datetime.now(CN_TZ)).astimezone(CN_TZ)
    return ReportMeta(
        title=title,
        run_id=run_id,
        window=win_meta,
        status=status,
        failed_sections=list(failed_sections),
        sections_enabled=[
            "overview", "sectors", "sentiment", "capital",
            "leaders", "style", "risk_outlook",
        ],
        llm_provider=llm_provider or "",
        plugin_version=plugin_version,
        input_fingerprint=input_fingerprint,
        generated_at=ts.isoformat(),
        error=error,
    )


def _build_title(window: Window) -> str:
    if window.mode == "day":
        return f"市场复盘 — {window.anchor}"
    return f"市场复盘 — {window.start} → {window.end}（{window.n_days} 个交易日）"


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------


def _build_headline(overview: OverviewSection, *, window: Window) -> ReportHeadline:
    """First-screen card; degrades gracefully when overview failed.

    The one-liner is the first paragraph of the overview narrative trimmed
    to 120 chars; for failed runs we fall back to a generic placeholder so
    the front-end never has to special-case "empty headline".
    """
    if overview.narrative_md:
        # First non-empty line only — front-end shows this as the hero text.
        nm = overview.narrative_md.strip()
        first_line = nm.split("\n", 1)[0].strip()
        one_liner = first_line[:120] or f"{window.anchor} 市场复盘"
    else:
        one_liner = f"{window.anchor} 市场复盘"

    # Re-pack OverviewSection.headline_metrics (snake_case Python attrs) into
    # the report-schema HeadlineMetric (same fields, sibling pydantic model).
    core_metrics = [
        HeadlineMetric(
            label=m.label, value=m.value, unit=m.unit,
            delta=m.delta, delta_unit=m.delta_unit, interpretation=m.interpretation,
        )
        for m in overview.headline_metrics
    ]
    # Trim to schema cap (max_length=8) defensively.
    core_metrics = core_metrics[:8]

    return ReportHeadline(
        one_liner=one_liner,
        market_tone=overview.market_tone,
        core_metrics=core_metrics,
        theme_tags=list(overview.theme_tags)[:4],
    )


# ---------------------------------------------------------------------------
# MetricsBlock
# ---------------------------------------------------------------------------


def _build_metrics_block(
    *,
    breadth: BreadthReview,
    sectors: SectorReview,
    capital: CapitalReview,
    leaders: LeaderReview,
    style: StyleReview,
    risk: RiskReview,
) -> MetricsBlock:
    return MetricsBlock(
        breadth_series=[_breadth_to_json(s) for s in breadth.series],
        index_returns=_index_returns_block(breadth),
        sector_matrix=_sector_matrix_to_json(sectors),
        capital_daily=_capital_daily(capital),
        leader_table=_leader_table(leaders),
        style_series=_style_series(style),
        risk_signals=[_risk_signal_to_metric(s) for s in risk.signals],
    )


def _breadth_to_json(snap) -> BreadthSnapshotJson:
    return BreadthSnapshotJson(
        trade_date=snap.trade_date,
        n_total=snap.n_total,
        n_up=snap.n_up,
        n_down=snap.n_down,
        n_flat=snap.n_flat,
        n_up5pct=snap.n_up5pct,
        n_down5pct=snap.n_down5pct,
        n_limit_up=snap.n_limit_up,
        n_limit_down=snap.n_limit_down,
        n_zhaban=snap.n_zhaban,
        # PR-3 ladder keys are int; JSON requires str.
        up_ladder={str(k): int(v) for k, v in snap.up_ladder.items()},
        total_amount_yi=snap.total_amount_yi,
        index_returns=dict(snap.index_returns),
    )


def _index_returns_block(breadth: BreadthReview) -> dict[str, IndexReturnJson]:
    """Aggregate per-index window-cumulative returns + leave series empty.

    PR-3 breadth.series carries daily pct_chg per index but no close /
    amount series; PR-5 builder is pure (no DB), so close_series /
    amount_series_yi stay empty. PR-6 runner can populate them later by
    querying mr_index_daily — design §15.6 charts will render empty arrays
    as "data unavailable" rather than crashing.
    """
    if not breadth.series:
        return {}
    # Per ts_code: collect daily pct_chg in window order, then geometric chain.
    per_index: dict[str, list[float]] = {}
    for snap in breadth.series:
        for ts_code, pct in snap.index_returns.items():
            per_index.setdefault(ts_code, []).append(float(pct))
    out: dict[str, IndexReturnJson] = {}
    for ts_code, dailies in per_index.items():
        factor = 1.0
        for p in dailies:
            factor *= 1.0 + p / 100.0
        cum_pct = (factor - 1.0) * 100.0
        out[ts_code] = IndexReturnJson(
            ts_code=ts_code,
            name=INDEX_NAMES.get(ts_code, ts_code),
            pct_chg_window=cum_pct,
            close_series=[],
            amount_series_yi=[],
        )
    return out


def _sector_matrix_to_json(sectors: SectorReview) -> SectorMatrixJson:
    matrix = sectors.matrix
    if matrix is None:
        return SectorMatrixJson()
    return SectorMatrixJson(
        sectors=list(matrix.sectors),
        sector_names=list(matrix.sector_names),
        trade_dates=list(matrix.trade_dates),
        pct_chg=[list(row) for row in matrix.values],
        cum_pct_chg=list(matrix.cum_pct_chg),
        persistence_days=list(matrix.persistence_days),
    )


def _capital_daily(capital: CapitalReview) -> list[CapitalDailyRow]:
    """Merge north + mkt series by trade_date into a single per-day row.

    Margin balance / delta fields are left None — PR-3 capital module
    doesn't carry the margin series (it's the risk module that reads
    margin for the融资骤变 signal). PR-6 runner can layer in a margin
    series if the front-end demands it.
    """
    north_by_date = {r.trade_date: r.north_money_yi for r in capital.north_series}
    mkt_by_date = {r.trade_date: r.main_net_yi for r in capital.mkt_series}
    all_dates = sorted(set(north_by_date) | set(mkt_by_date))
    return [
        CapitalDailyRow(
            trade_date=td,
            north_money_yi=north_by_date.get(td),
            main_net_inflow_yi=mkt_by_date.get(td),
            margin_balance_yi=None,
            margin_delta_yi=None,
        )
        for td in all_dates
    ]


def _leader_table(leaders: LeaderReview) -> list[MetricsLeaderRow]:
    rows: list[MetricsLeaderRow] = []
    for cand in list(leaders.primary) + list(leaders.secondary):
        rows.append(MetricsLeaderRow(
            ts_code=cand.ts_code,
            name=cand.name,
            score=cand.score,
            score_breakdown=dict(cand.score_breakdown),
            industries=list(cand.industries),
            concepts=list(cand.concepts),
            sector_top_hit=list(cand.sector_top_hit),
            ladder_height=cand.ladder_height,
            range_pct_chg=cand.range_pct_chg,
            cum_main_inflow_yi=cand.cum_main_inflow_yi,
        ))
    return rows


def _style_series(style: StyleReview) -> StyleSeriesJson:
    series_json: list = []
    # PR-3 StyleSeriesPoint → PR-4 StyleSeriesPointJson — fields line up.
    for pt in style.series:
        series_json.append({
            "trade_date": pt.trade_date,
            "large_cap_ret": pt.large_cap_ret,
            "small_cap_ret": pt.small_cap_ret,
            "value_ret": pt.value_ret,
            "growth_ret": pt.growth_ret,
            "big_to_small_ratio": pt.big_to_small_ratio,
        })
    avg_big_to_small = float(style.range_summary.get("avg_big_to_small_ratio", 0.0))
    return StyleSeriesJson.model_validate({
        "series": series_json,
        "avg_big_to_small": avg_big_to_small,
        "avg_value_to_growth": None,
        "half_period_flip": style.flip_signal,
    })


def _risk_signal_to_metric(s) -> MetricsRiskSignal:
    samples = list(s.affected_samples)
    return MetricsRiskSignal(
        name=s.name,
        triggered=s.triggered,
        # PR-3 risk has 'positive' too which doesn't fit metric-block's
        # smaller enum; map any unknown to "info" defensively.
        severity=s.severity if s.severity in ("info", "warning", "critical") else "info",
        sample_count=len(samples),
        samples_top_k=samples[:10],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _typed_section(
    sections: dict[SectionName, SectionBase],
    name: SectionName,
    cls: type,
):
    """Pull ``name`` out of ``sections`` and assert it's the expected type.

    The pipeline (PR-4 :func:`run_sections`) always returns one entry per
    section; this defensive cast keeps the builder honest about the shape
    so a mis-typed mapping fails immediately rather than via a confusing
    pydantic ValidationError on the root.
    """
    if name not in sections:
        raise KeyError(f"sections missing required key: {name!r}")
    schema = sections[name]
    if not isinstance(schema, cls):
        raise TypeError(
            f"sections[{name!r}] expected {cls.__name__}, got {type(schema).__name__}"
        )
    return schema
