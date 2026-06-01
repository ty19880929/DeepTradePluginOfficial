"""schemas — pydantic round-trip + extra='forbid' + alias coverage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_review.schemas import (
    SCHEMA_VERSION,
    SECTION_ORDER,
    SECTION_SCHEMAS,
    CapitalSection,
    EvidenceItem,
    Finding,
    HeadlineMetric,
    LeadersSection,
    OverviewSection,
    OutlookHypothesis,
    RiskOutlookSection,
    SectorClassification,
    SectorEntry,
    SectorsSection,
    SentimentSection,
    StyleSection,
)


def test_schema_version_is_v10() -> None:
    assert SCHEMA_VERSION == "1.0"


def test_section_order_covers_all_keys() -> None:
    assert set(SECTION_ORDER) == set(SECTION_SCHEMAS)
    # Overview must be first — design §5.4.4 makes theme_tags upstream of others.
    assert SECTION_ORDER[0] == "overview"


def test_evidence_item_disallows_array_value() -> None:
    """Design §15.4 — value MUST be a scalar, not a list/dict."""
    with pytest.raises(ValidationError):
        EvidenceItem(field="f", value=[1, 2, 3], unit="个", interpretation="x")  # type: ignore[arg-type]


def test_evidence_item_camelcase_alias_round_trip() -> None:
    """populate_by_name=True + alias_generator allow both forms."""
    item = EvidenceItem(field="f", value=1.0, unit="亿", interpretation="x")
    dumped = item.model_dump(by_alias=True)
    assert dumped == {"field": "f", "value": 1.0, "unit": "亿", "interpretation": "x"}
    # Round-trip via alias-form JSON (no diff with snake_case here since all
    # field names are single-token; the alias test below uses a multi-token one).
    rt = EvidenceItem.model_validate(dumped)
    assert rt == item


def test_headline_metric_multitoken_alias() -> None:
    """delta_unit ↔ deltaUnit alias round-trip."""
    m = HeadlineMetric(
        label="北向", value=42.0, unit="亿", delta=5.0, delta_unit="亿", interpretation="多",
    )
    dumped = m.model_dump(by_alias=True)
    assert dumped["deltaUnit"] == "亿"
    rt = HeadlineMetric.model_validate({**dumped})
    assert rt.delta_unit == "亿"
    assert rt.label == "北向"


def test_overview_section_requires_three_headline_metrics() -> None:
    """Design Appendix A — headline_metrics min_length=3."""
    finding = Finding(
        headline="x", detail="y",
        evidence=[EvidenceItem(field="f", value=1, unit="个", interpretation="z")],
    )
    with pytest.raises(ValidationError):
        OverviewSection(
            market_tone="震荡分化",
            headline_metrics=[],  # too few
            theme_tags=[],
            findings=[finding],
        )


def test_overview_section_accepts_minimal_valid() -> None:
    metrics = [
        HeadlineMetric(label=f"m{i}", value=i, unit="个") for i in range(3)
    ]
    s = OverviewSection(
        market_tone="震荡分化",
        headline_metrics=metrics,
        theme_tags=["主线1", "主线2"],
    )
    assert s.market_tone == "震荡分化"
    assert s.narrative_md == ""
    assert s.error is None


def test_overview_section_market_tone_is_constrained() -> None:
    with pytest.raises(ValidationError):
        OverviewSection(
            market_tone="非法值",  # type: ignore[arg-type]
            headline_metrics=[
                HeadlineMetric(label="x", value=1, unit="个") for _ in range(3)
            ],
        )


def test_sectors_section_classification_default() -> None:
    s = SectorsSection(provider="ths")
    assert s.classification == SectorClassification()
    assert s.today_top == []


def test_sectors_entry_optional_fields_nullable() -> None:
    e = SectorEntry(name="光模块", pct_chg=3.5)
    assert e.net_inflow_yi is None
    assert e.persistence_days is None


def test_sentiment_section_score_bounds() -> None:
    with pytest.raises(ValidationError):
        SentimentSection(avg_score=101.0)


def test_capital_section_optional_lists() -> None:
    s = CapitalSection()
    assert s.north_summary == []
    assert s.lhb_highlights == []


def test_leaders_section_default_min_score() -> None:
    s = LeadersSection()
    assert s.min_score == 50.0


def test_style_section_default_balanced() -> None:
    s = StyleSection()
    assert s.dominant_style == "balanced"
    assert s.flip_signal is False


def test_risk_outlook_requires_at_least_one_hypothesis() -> None:
    with pytest.raises(ValidationError):
        RiskOutlookSection(hypotheses=[])


def test_risk_outlook_minimal_valid() -> None:
    s = RiskOutlookSection(
        hypotheses=[OutlookHypothesis(
            title="底部承接",
            rationale="北向连续 3 日净流入 + 量能温和放大",
            watch_points=["注意 5 日均线"],
            fail_triggers=["指数失守 60 日"],
        )],
    )
    assert s.overall_risk == "moderate"


def test_section_with_extra_field_rejects() -> None:
    """All section schemas inherit extra='forbid'."""
    with pytest.raises(ValidationError):
        OverviewSection.model_validate({
            "market_tone": "震荡分化",
            "headlineMetrics": [
                {"label": f"m{i}", "value": i, "unit": "个"} for i in range(3)
            ],
            "themeTags": [],
            "unexpectedField": "boom",
        })


def test_camel_alias_in_serialized_payload() -> None:
    """All multi-token fields must use camelCase keys in the JSON wire format."""
    finding = Finding(
        headline="x", detail="y",
        evidence=[EvidenceItem(field="f", value=1, unit="个", interpretation="z")],
    )
    s = OverviewSection(
        market_tone="震荡分化",
        headline_metrics=[
            HeadlineMetric(label="x", value=1, unit="个") for _ in range(3)
        ],
        theme_tags=["t1"],
        narrative_md="abc",
        findings=[finding],
    )
    payload = s.model_dump(by_alias=True)
    # multi-token field names must surface as camelCase
    assert "marketTone" in payload
    assert "headlineMetrics" in payload
    assert "themeTags" in payload
    assert "narrativeMd" in payload
    # snake_case form should NOT appear in by_alias output
    assert "market_tone" not in payload
