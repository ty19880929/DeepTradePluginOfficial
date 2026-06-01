"""build_review_report — pure assembly verification."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from market_review.metrics.breadth import BreadthReview, BreadthSnapshot
from market_review.metrics.capital import (
    CapitalReview, NorthFlowRow, MktFlowRow,
)
from market_review.metrics.leaders import LeaderCandidate, LeaderReview
from market_review.metrics.risk import RiskReview, RiskSignal
from market_review.metrics.sectors import (
    SectorEntry, SectorMatrix, SectorReview,
)
from market_review.metrics.sentiment import SentimentReview
from market_review.metrics.style import StyleReview
from market_review.report.builder import INDEX_NAMES, build_review_report
from market_review.report.schema import ReviewReportSchema
from market_review.schemas import (
    CapitalSection,
    HeadlineMetric,
    LeadersSection,
    OutlookHypothesis,
    OverviewSection,
    RiskOutlookSection,
    SectorsSection,
    SentimentSection,
    StyleSection,
)
from market_review.windows import Window


def _window() -> Window:
    return Window(
        mode="range", start="20260528", end="20260530",
        trade_dates=("20260528", "20260529", "20260530"),
        anchor="20260530",
    )


def _make_overview(*, market_tone="震荡分化", error=None) -> OverviewSection:
    return OverviewSection(
        market_tone=market_tone,
        headline_metrics=[
            HeadlineMetric(label="北向", value=42.0, unit="亿", interpretation="正向"),
            HeadlineMetric(label="情绪", value=68.0, unit="分"),
            HeadlineMetric(label="涨家", value=2300, unit="家"),
        ],
        theme_tags=["主线A", "主线B"],
        narrative_md="今日 A 股呈现结构性上涨。\n\n北向资金延续净买。",
        error=error,
    )


def _make_risk_outlook() -> RiskOutlookSection:
    return RiskOutlookSection(
        hypotheses=[OutlookHypothesis(
            title="震荡延续",
            rationale="量能温和",
            watch_points=["5日"],
            fail_triggers=["跌破60日"],
        )],
    )


def _make_sections(*, overview=None, sectors=None) -> dict:
    return {
        "overview": overview or _make_overview(),
        "sectors": sectors or SectorsSection(provider="ths"),
        "sentiment": SentimentSection(),
        "capital": CapitalSection(),
        "leaders": LeadersSection(),
        "style": StyleSection(),
        "risk_outlook": _make_risk_outlook(),
    }


def _minimal_call(**overrides):
    defaults = dict(
        status="success",
        window=_window(),
        breadth=BreadthReview(),
        sentiment=SentimentReview(),
        capital=CapitalReview(),
        sectors=SectorReview(),
        leaders=LeaderReview(),
        style=StyleReview(),
        risk=RiskReview(),
        sections=_make_sections(),
        failed_sections=[],
        run_id="run-1",
        llm_provider="deepseek",
        plugin_version="0.1.0",
        input_fingerprint="a" * 64,
    )
    defaults.update(overrides)
    return build_review_report(**defaults)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_builder_returns_review_report_schema() -> None:
    rpt = _minimal_call()
    assert isinstance(rpt, ReviewReportSchema)


def test_builder_assembles_meta_title_for_range() -> None:
    rpt = _minimal_call()
    assert rpt.meta.title == "市场复盘 — 20260528 → 20260530（3 个交易日）"


def test_builder_assembles_meta_title_for_single_day() -> None:
    win = Window(mode="day", start="20260530", end="20260530",
                 trade_dates=("20260530",), anchor="20260530")
    rpt = _minimal_call(window=win)
    assert rpt.meta.title == "市场复盘 — 20260530"


def test_builder_meta_carries_input_fingerprint_and_run_id() -> None:
    rpt = _minimal_call(run_id="r-XYZ", input_fingerprint="b" * 64)
    assert rpt.meta.run_id == "r-XYZ"
    assert rpt.meta.input_fingerprint == "b" * 64


def test_builder_generated_at_uses_cn_timezone() -> None:
    ts = datetime(2026, 5, 30, 18, 0, 0, tzinfo=timezone(timedelta(hours=8)))
    rpt = _minimal_call(generated_at=ts)
    assert rpt.meta.generated_at.endswith("+08:00")
    assert "2026-05-30T18:00" in rpt.meta.generated_at


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------


def test_headline_one_liner_from_overview_first_line() -> None:
    rpt = _minimal_call()
    assert rpt.headline.one_liner == "今日 A 股呈现结构性上涨。"


def test_headline_one_liner_falls_back_to_window_anchor_on_empty_narrative() -> None:
    overview = _make_overview()
    overview = OverviewSection.model_validate({
        **overview.model_dump(by_alias=True), "narrativeMd": "",
    })
    rpt = _minimal_call(sections=_make_sections(overview=overview))
    assert rpt.headline.one_liner == "20260530 市场复盘"


def test_headline_market_tone_and_theme_tags_from_overview() -> None:
    rpt = _minimal_call(sections=_make_sections(overview=_make_overview(market_tone="高位见顶")))
    assert rpt.headline.market_tone == "高位见顶"
    assert rpt.headline.theme_tags == ["主线A", "主线B"]


def test_headline_core_metrics_copied_from_overview() -> None:
    rpt = _minimal_call()
    assert len(rpt.headline.core_metrics) == 3
    assert rpt.headline.core_metrics[0].label == "北向"


# ---------------------------------------------------------------------------
# MetricsBlock conversions
# ---------------------------------------------------------------------------


def test_breadth_series_ladder_keys_stringified() -> None:
    breadth = BreadthReview(series=[BreadthSnapshot(
        trade_date="20260530",
        n_total=5000, n_up=2500, n_down=2300, n_flat=200,
        n_up5pct=80, n_down5pct=20,
        n_limit_up=50, n_limit_down=3, n_zhaban=10,
        up_ladder={2: 12, 3: 5, 4: 1},
        n_lhb=20,
        total_amount_yi=8500.0,
        index_returns={"000001.SH": 0.5},
    )])
    rpt = _minimal_call(breadth=breadth)
    snap = rpt.metrics.breadth_series[0]
    assert snap.up_ladder == {"2": 12, "3": 5, "4": 1}


def test_index_returns_geometric_chain_and_name_lookup() -> None:
    breadth = BreadthReview(series=[
        BreadthSnapshot(
            trade_date="20260529",
            n_total=0, n_up=0, n_down=0, n_flat=0,
            n_up5pct=0, n_down5pct=0,
            n_limit_up=0, n_limit_down=0, n_zhaban=0,
            up_ladder={}, n_lhb=0, total_amount_yi=0.0,
            index_returns={"000300.SH": 2.0},
        ),
        BreadthSnapshot(
            trade_date="20260530",
            n_total=0, n_up=0, n_down=0, n_flat=0,
            n_up5pct=0, n_down5pct=0,
            n_limit_up=0, n_limit_down=0, n_zhaban=0,
            up_ladder={}, n_lhb=0, total_amount_yi=0.0,
            index_returns={"000300.SH": 3.0},
        ),
    ])
    rpt = _minimal_call(breadth=breadth)
    hs300 = rpt.metrics.index_returns["000300.SH"]
    # 1.02 * 1.03 = 1.0506 → 5.06%
    assert hs300.pct_chg_window == pytest.approx(5.06, abs=0.01)
    assert hs300.name == INDEX_NAMES["000300.SH"]
    # Pure builder: close + amount series stay empty for v0.1.
    assert hs300.close_series == []


def test_sector_matrix_transferred_when_present() -> None:
    sectors = SectorReview(
        matrix=SectorMatrix(
            sectors=["S1", "S2"], sector_names=["光模块", "机器人"],
            trade_dates=["20260529", "20260530"],
            values=[[2.0, 3.0], [-1.0, -2.0]],
            cum_pct_chg=[5.06, -2.98],
            persistence_days=[2, 1],
        ),
    )
    rpt = _minimal_call(sectors=sectors)
    assert rpt.metrics.sector_matrix.sectors == ["S1", "S2"]
    assert rpt.metrics.sector_matrix.cum_pct_chg == [5.06, -2.98]


def test_sector_matrix_empty_when_no_matrix() -> None:
    rpt = _minimal_call(sectors=SectorReview())
    assert rpt.metrics.sector_matrix.sectors == []
    assert rpt.metrics.sector_matrix.cum_pct_chg == []


def test_capital_daily_merges_north_and_mkt_series() -> None:
    capital = CapitalReview(
        north_series=[
            NorthFlowRow(trade_date="20260529", north_money_yi=10.0),
            NorthFlowRow(trade_date="20260530", north_money_yi=-5.0),
        ],
        mkt_series=[
            MktFlowRow(trade_date="20260529", main_net_yi=20.0, retail_net_yi=5.0),
            MktFlowRow(trade_date="20260530", main_net_yi=-15.0, retail_net_yi=8.0),
        ],
    )
    rpt = _minimal_call(capital=capital)
    daily = rpt.metrics.capital_daily
    assert len(daily) == 2
    assert daily[0].trade_date == "20260529"
    assert daily[0].north_money_yi == 10.0
    assert daily[0].main_net_inflow_yi == 20.0
    assert daily[0].margin_balance_yi is None  # PR-3 doesn't carry it


def test_leader_table_combines_primary_and_secondary() -> None:
    leaders = LeaderReview(
        primary=[LeaderCandidate(
            ts_code="A.SH", name="A名", score=80.0,
            score_breakdown={"ladder": 20, "return": 22, "capital": 20, "theme": 18},
            industries=["光模块"], concepts=[], sector_top_hit=["光模块"],
            ladder_height=4, range_pct_chg=25.0, cum_main_inflow_yi=15.0,
        )],
        secondary=[LeaderCandidate(
            ts_code="B.SH", name="B名", score=60.0,
            score_breakdown={"ladder": 15, "return": 15, "capital": 15, "theme": 15},
            industries=["机器人"], concepts=[], sector_top_hit=[],
            ladder_height=2, range_pct_chg=10.0, cum_main_inflow_yi=5.0,
        )],
    )
    rpt = _minimal_call(leaders=leaders)
    rows = rpt.metrics.leader_table
    assert [r.ts_code for r in rows] == ["A.SH", "B.SH"]
    # MetricsLeaderRow has NO rationale; pydantic strips any LLM addition.
    payload = rows[0].model_dump(by_alias=True)
    assert "rationale" not in payload


def test_risk_signals_drop_detail_and_keep_sample_count() -> None:
    risk = RiskReview(signals=[RiskSignal(
        name="north_capital_outflow",
        triggered=True,
        severity="warning",
        detail="anchor 日北向 -15.0 亿；窗内 -40.0 亿",
        affected_samples=["A.SH", "B.SH", "C.SH"],
    )])
    rpt = _minimal_call(risk=risk)
    sigs = rpt.metrics.risk_signals
    assert len(sigs) == 1
    s = sigs[0]
    assert s.sample_count == 3
    assert s.samples_top_k == ["A.SH", "B.SH", "C.SH"]
    # detail (LLM prose) is dropped on the metrics side.
    payload = s.model_dump(by_alias=True)
    assert "detail" not in payload


def test_risk_signal_severity_positive_collapses_to_info() -> None:
    """PR-3 RiskSignal allows 'positive' but MetricsRiskSignal does not."""
    risk = RiskReview(signals=[RiskSignal(
        name="x", triggered=False, severity="positive",
        detail="test", affected_samples=[],
    )])
    rpt = _minimal_call(risk=risk)
    assert rpt.metrics.risk_signals[0].severity == "info"


def test_style_series_collapses_flip_signal_and_avg() -> None:
    style = StyleReview(
        dominant_style="large_cap", flip_signal=True,
        range_summary={"avg_big_to_small_ratio": 2.5},
    )
    rpt = _minimal_call(style=style)
    assert rpt.metrics.style_series.half_period_flip is True
    assert rpt.metrics.style_series.avg_big_to_small == 2.5


# ---------------------------------------------------------------------------
# Failed section path
# ---------------------------------------------------------------------------


def test_failed_section_kept_in_root_with_error() -> None:
    sections = _make_sections(
        sectors=SectorsSection(error="LLMValidationError: boom"),
    )
    rpt = _minimal_call(
        sections=sections, failed_sections=["sectors"], status="partial_failed",
    )
    assert rpt.sectors.error == "LLMValidationError: boom"
    assert rpt.meta.failed_sections == ["sectors"]
    assert rpt.meta.status == "partial_failed"


def test_missing_section_in_sections_dict_raises() -> None:
    sections = _make_sections()
    del sections["leaders"]
    with pytest.raises(KeyError):
        _minimal_call(sections=sections)


def test_wrong_type_in_sections_dict_raises_typeerror() -> None:
    sections = _make_sections()
    sections["leaders"] = SectorsSection(provider="ths")  # wrong type
    with pytest.raises(TypeError):
        _minimal_call(sections=sections)
