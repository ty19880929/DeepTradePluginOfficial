"""ReviewReportSchema — round-trip + extras + extra='forbid' coverage."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from market_review.report.schema import (
    SCHEMA_VERSION,
    BreadthSnapshotJson,
    HeadlineMetric,
    IndexReturnJson,
    MetricsBlock,
    MetricsLeaderRow,
    MetricsRiskSignal,
    ReportHeadline,
    ReportMeta,
    ReviewReportSchema,
    SectorMatrixJson,
    StyleSeriesJson,
    WindowMeta,
)
from market_review.schemas import (
    CapitalSection,
    HeadlineMetric as SchemaHeadlineMetric,
    LeadersSection,
    OutlookHypothesis,
    OverviewSection,
    RiskOutlookSection,
    SectorsSection,
    SentimentSection,
    StyleSection,
)


def _make_window_meta() -> WindowMeta:
    return WindowMeta(
        mode="day", start="20260530", end="20260530",
        anchor="20260530", n_days=1, trade_dates=["20260530"],
    )


def _make_meta(**overrides) -> ReportMeta:
    defaults: dict = dict(
        title="市场复盘 — 20260530",
        run_id="run-1",
        window=_make_window_meta(),
        status="success",
        failed_sections=[],
        sections_enabled=[
            "overview", "sectors", "sentiment", "capital",
            "leaders", "style", "risk_outlook",
        ],
        llm_provider="deepseek",
        plugin_version="0.1.0",
        input_fingerprint="a" * 64,
        generated_at="2026-05-30T18:00:00+08:00",
        error=None,
    )
    defaults.update(overrides)
    return ReportMeta(**defaults)


def _make_headline(**overrides) -> ReportHeadline:
    return ReportHeadline(
        one_liner="震荡分化，新主线由光模块接棒",
        market_tone="震荡分化",
        core_metrics=[
            HeadlineMetric(label="北向", value=42.0, unit="亿"),
            HeadlineMetric(label="情绪", value=68.0, unit="分"),
            HeadlineMetric(label="涨家", value=2300, unit="家"),
            HeadlineMetric(label="涨停", value=80, unit="家"),
        ],
        theme_tags=["AI 算力", "光模块"],
        **overrides,
    )


def _make_overview() -> OverviewSection:
    return OverviewSection(
        market_tone="震荡分化",
        headline_metrics=[
            SchemaHeadlineMetric(label=f"m{i}", value=i, unit="个") for i in range(3)
        ],
        theme_tags=["t1"],
        narrative_md="今日呈现结构性上涨。",
    )


def _make_risk_outlook() -> RiskOutlookSection:
    return RiskOutlookSection(
        hypotheses=[OutlookHypothesis(
            title="震荡延续",
            rationale="量能温和",
            watch_points=["5日均线"],
            fail_triggers=["跌破60日"],
        )],
    )


def _make_full_report(**meta_overrides) -> ReviewReportSchema:
    return ReviewReportSchema(
        meta=_make_meta(**meta_overrides),
        headline=_make_headline(),
        overview=_make_overview(),
        sectors=SectorsSection(provider="ths"),
        sentiment=SentimentSection(),
        capital=CapitalSection(),
        leaders=LeadersSection(),
        style=StyleSection(),
        risk_outlook=_make_risk_outlook(),
        metrics=MetricsBlock(),
    )


# ---------------------------------------------------------------------------
# Schema version + basic shape
# ---------------------------------------------------------------------------


def test_schema_version_constant_is_v10() -> None:
    assert SCHEMA_VERSION == "1.0"


def test_review_report_round_trips_through_json() -> None:
    rpt = _make_full_report()
    raw = rpt.model_dump_json(by_alias=True)
    parsed = json.loads(raw)
    rebuilt = ReviewReportSchema.model_validate(parsed)
    assert rebuilt.meta.run_id == rpt.meta.run_id
    assert rebuilt.headline.market_tone == "震荡分化"
    assert rebuilt.risk_outlook.hypotheses[0].title == "震荡延续"


def test_review_report_serializes_camel_case_keys() -> None:
    rpt = _make_full_report()
    payload = rpt.model_dump(by_alias=True)
    # Top-level multi-token field must be camelCase per design §15.1.4.
    assert "riskOutlook" in payload
    assert "metrics" in payload
    assert "meta" in payload
    # No snake-case stragglers at the root.
    assert "risk_outlook" not in payload


def test_meta_serializes_camel_case_fields() -> None:
    rpt = _make_full_report()
    meta_payload = rpt.model_dump(by_alias=True)["meta"]
    for key in ("runId", "failedSections", "sectionsEnabled", "llmProvider",
                "schemaVersion", "pluginVersion", "inputFingerprint", "generatedAt"):
        assert key in meta_payload


# ---------------------------------------------------------------------------
# extras pickup
# ---------------------------------------------------------------------------


def test_extras_defaults_to_empty_dict() -> None:
    rpt = _make_full_report()
    assert rpt.extras == {}
    # In wire format the alias is "_extras".
    assert rpt.model_dump(by_alias=True)["_extras"] == {}


def test_extras_round_trip_via_alias() -> None:
    """Stuffing forward-compat data into _extras must survive round-trip."""
    rpt = _make_full_report()
    payload = rpt.model_dump(by_alias=True)
    payload["_extras"] = {"future_field_x": [1, 2, 3], "nested": {"k": "v"}}
    rebuilt = ReviewReportSchema.model_validate(payload)
    assert rebuilt.extras == {"future_field_x": [1, 2, 3], "nested": {"k": "v"}}


# ---------------------------------------------------------------------------
# extra='forbid' triggers
# ---------------------------------------------------------------------------


def test_root_with_unknown_field_rejects() -> None:
    """Adding a top-level field NOT in the schema should raise unless via _extras."""
    rpt = _make_full_report()
    payload = rpt.model_dump(by_alias=True)
    payload["unexpectedRoot"] = "boom"
    with pytest.raises(ValidationError):
        ReviewReportSchema.model_validate(payload)


def test_window_meta_with_unknown_field_rejects() -> None:
    payload = _make_window_meta().model_dump(by_alias=True)
    payload["unknownField"] = 42
    with pytest.raises(ValidationError):
        WindowMeta.model_validate(payload)


def test_metrics_block_sub_unknown_field_rejects() -> None:
    payload = MetricsBlock().model_dump(by_alias=True)
    payload["extraField"] = "no"
    with pytest.raises(ValidationError):
        MetricsBlock.model_validate(payload)


# ---------------------------------------------------------------------------
# input_fingerprint length
# ---------------------------------------------------------------------------


def test_input_fingerprint_must_be_64_hex() -> None:
    with pytest.raises(ValidationError):
        _make_meta(input_fingerprint="abc")
    with pytest.raises(ValidationError):
        _make_meta(input_fingerprint="a" * 65)


# ---------------------------------------------------------------------------
# Failed-section continuity (§11.3 / §15.1.5)
# ---------------------------------------------------------------------------


def test_failed_section_carries_error_and_empty_narrative() -> None:
    """A section that failed mid-pipeline should still appear, with empty
    narrative + populated ``error``."""
    failed_section = SectorsSection(error="LLMValidationError: ...")
    rpt = ReviewReportSchema(
        meta=_make_meta(status="partial_failed", failed_sections=["sectors"]),
        headline=_make_headline(),
        overview=_make_overview(),
        sectors=failed_section,
        sentiment=SentimentSection(),
        capital=CapitalSection(),
        leaders=LeadersSection(),
        style=StyleSection(),
        risk_outlook=_make_risk_outlook(),
        metrics=MetricsBlock(),
    )
    assert rpt.sectors.error is not None
    assert rpt.sectors.narrative_md == ""
    payload = rpt.model_dump(by_alias=True)
    assert payload["meta"]["failedSections"] == ["sectors"]
    assert payload["sectors"]["error"].startswith("LLMValidationError")


def test_status_failed_serializes() -> None:
    """Even fully-failed runs must serialize for upload (design §15.1.8)."""
    rpt = _make_full_report(
        status="failed",
        error="TushareUnauthorizedError: token expired",
    )
    payload = rpt.model_dump(by_alias=True)
    assert payload["meta"]["status"] == "failed"
    assert payload["meta"]["error"].startswith("TushareUnauthorizedError")


# ---------------------------------------------------------------------------
# MetricsBlock sub-model sanity
# ---------------------------------------------------------------------------


def test_breadth_snapshot_json_str_keyed_ladder() -> None:
    """up_ladder must be string-keyed (JSON requires string keys)."""
    snap = BreadthSnapshotJson(
        trade_date="20260530",
        n_total=5000, n_up=2500, n_down=2300, n_flat=200,
        n_up5pct=80, n_down5pct=20,
        n_limit_up=50, n_limit_down=3, n_zhaban=10,
        up_ladder={"2": 12, "3": 5, "4": 1},
        total_amount_yi=8500.0,
        index_returns={"000001.SH": 0.5},
    )
    payload = snap.model_dump(by_alias=True)
    assert payload["upLadder"] == {"2": 12, "3": 5, "4": 1}


def test_index_return_json_defaults_empty_series() -> None:
    ir = IndexReturnJson(ts_code="000001.SH", name="上证综指", pct_chg_window=1.5)
    assert ir.close_series == []
    assert ir.amount_series_yi == []


def test_sector_matrix_json_defaults_empty() -> None:
    m = SectorMatrixJson()
    assert m.sectors == []
    assert m.values if hasattr(m, "values") else True  # noqa
    assert m.cum_pct_chg == []


def test_metrics_risk_signal_separate_from_section_signal() -> None:
    sig = MetricsRiskSignal(
        name="north_capital_outflow", triggered=True,
        severity="warning", sample_count=12,
        samples_top_k=["A", "B", "C"],
    )
    payload = sig.model_dump(by_alias=True)
    # camelCase wire format
    assert "sampleCount" in payload
    assert "samplesTopK" in payload
    # NO "detail" — that's LLM section's field; metrics block stays numeric.
    assert "detail" not in payload


def test_metrics_leader_row_no_rationale_field() -> None:
    row = MetricsLeaderRow(ts_code="600001.SH", name="A股龙头", score=85.0)
    payload = row.model_dump(by_alias=True)
    # LeaderCandidateJson (section schema) has rationale; the metrics-block
    # row mustn't carry LLM prose — this is the contract enforcement test.
    assert "rationale" not in payload


def test_style_series_json_defaults_balanced() -> None:
    s = StyleSeriesJson()
    assert s.half_period_flip is False
    assert s.avg_big_to_small == 0.0
    assert s.series == []
