"""prompts — hard discipline présent + prev_context injection."""

from __future__ import annotations

import json

import pytest

from market_review.metrics.breadth import BreadthReview, BreadthSnapshot
from market_review.metrics.capital import CapitalReview
from market_review.metrics.leaders import LeaderReview
from market_review.metrics.risk import RiskReview
from market_review.metrics.sectors import SectorEntry, SectorReview
from market_review.metrics.sentiment import SentimentReview, SentimentSnapshot
from market_review.metrics.style import StyleReview
from market_review.prompts import (
    HARD_DISCIPLINE,
    SECTION_SYSTEM_PROMPTS,
    build_capital_user_prompt,
    build_leaders_user_prompt,
    build_overview_user_prompt,
    build_prev_context,
    build_risk_outlook_user_prompt,
    build_sectors_user_prompt,
    build_sentiment_user_prompt,
    build_style_user_prompt,
    system_prompt,
)
from market_review.schemas import HeadlineMetric, OverviewSection, SECTION_ORDER
from market_review.windows import Window


def _make_window() -> Window:
    return Window(
        mode="day", start="20260530", end="20260530",
        trade_dates=("20260530",), anchor="20260530",
    )


def _empty_bundle():
    """Empty PR-3 reviews — enough for prompts to build payloads."""
    return {
        "breadth": BreadthReview(),
        "sentiment": SentimentReview(),
        "capital": CapitalReview(),
        "sectors": SectorReview(),
        "leaders": LeaderReview(),
        "style": StyleReview(),
        "risk": RiskReview(),
    }


def test_hard_discipline_lists_six_rules() -> None:
    """All seven discipline rules from design §5.4.2 are present (rule 7
    added in v0.1.4 to pin the Finding shape against LLM hallucinations
    that placed narrativeMd inside findings[*]).
    """
    for keyword in ("严禁使用外部搜索", "严禁编造数据", "evidence", "四元组",
                    "仅输出 JSON", "1200 中文字", "标量",
                    "headline", "detail", "severity", "extra_forbidden"):
        assert keyword in HARD_DISCIPLINE


def test_system_prompt_appends_discipline() -> None:
    for section in SECTION_ORDER:
        out = system_prompt(section)
        assert "【硬性纪律】" in out
        # The section-specific body must precede the discipline block.
        assert out.endswith(HARD_DISCIPLINE.strip())


def test_system_prompt_unknown_section_raises() -> None:
    with pytest.raises(KeyError):
        system_prompt("unknown")  # type: ignore[arg-type]


def test_each_section_has_system_prompt() -> None:
    assert set(SECTION_SYSTEM_PROMPTS) == set(SECTION_ORDER)


def test_overview_user_prompt_is_valid_json(  # noqa: ARG001
    request: pytest.FixtureRequest,
) -> None:
    """Compact JSON-only output, parses round-trip."""
    bundle = _empty_bundle()
    out = build_overview_user_prompt(window=_make_window(), **bundle)
    decoded = json.loads(out)
    # Top-level summary buckets must be present.
    assert "window" in decoded
    assert "breadthSummary" in decoded
    assert "sectorsSummary" in decoded


def test_build_prev_context_extracts_overview_fields() -> None:
    overview = OverviewSection(
        market_tone="震荡分化",
        headline_metrics=[
            HeadlineMetric(label=f"m{i}", value=i, unit="个") for i in range(3)
        ],
        theme_tags=["主线1", "主线2"],
    )
    ctx = build_prev_context(overview)
    assert ctx == {"marketTone": "震荡分化", "themeTags": ["主线1", "主线2"]}


def test_prev_context_none_yields_empty_dict() -> None:
    assert build_prev_context(None) == {}


def test_sectors_user_prompt_injects_prev_context() -> None:
    prev = {"marketTone": "震荡分化", "themeTags": ["t1"]}
    out = build_sectors_user_prompt(
        window=_make_window(), sectors=SectorReview(), prev_context=prev,
    )
    decoded = json.loads(out)
    assert decoded["prevContext"] == prev


def test_sectors_user_prompt_omits_prev_context_when_empty() -> None:
    out = build_sectors_user_prompt(
        window=_make_window(), sectors=SectorReview(), prev_context={},
    )
    decoded = json.loads(out)
    assert "prevContext" not in decoded


def test_sentiment_user_prompt_serializes_series() -> None:
    snap = SentimentSnapshot(
        trade_date="20260530", median_pct_chg=0.5, mean_pct_chg=0.2,
        pos_ratio=0.6, top_ratio=0.1, crash_ratio=0.05,
        limit_up_intensity=0.02, connection_health=0.8,
        n_lhb=20, north_money_yi=15.0, score_0_100=65.0,
    )
    sentiment = SentimentReview(series=[snap], avg_score=65.0,
                                strongest_day="20260530", weakest_day="20260530")
    out = build_sentiment_user_prompt(
        window=_make_window(), sentiment=sentiment, prev_context={},
    )
    decoded = json.loads(out)
    assert decoded["sentiment"]["series"][0]["trade_date"] == "20260530"


def test_capital_user_prompt_keys_present() -> None:
    out = build_capital_user_prompt(
        window=_make_window(), capital=CapitalReview(), prev_context={},
    )
    decoded = json.loads(out)
    assert "capital" in decoded


def test_leaders_user_prompt_includes_sectors_context() -> None:
    sectors = SectorReview(today_top=[
        SectorEntry(ts_code="X", name="光模块", pct_chg=5.0, persistence_days=1),
    ])
    out = build_leaders_user_prompt(
        window=_make_window(), leaders=LeaderReview(),
        sectors=sectors, prev_context={},
    )
    decoded = json.loads(out)
    assert decoded["sectorsContext"]["todayTop"][0]["name"] == "光模块"


def test_style_and_risk_user_prompts_run_without_crash() -> None:
    out_style = build_style_user_prompt(
        window=_make_window(), style=StyleReview(), prev_context={},
    )
    out_risk = build_risk_outlook_user_prompt(
        window=_make_window(), risk=RiskReview(),
        breadth=BreadthReview(), capital=CapitalReview(), prev_context={},
    )
    assert json.loads(out_style)["style"] is not None
    assert json.loads(out_risk)["risk"] is not None


def test_overview_user_prompt_carries_window_field(  # noqa: ARG001
    request: pytest.FixtureRequest,
) -> None:
    bundle = _empty_bundle()
    decoded = json.loads(build_overview_user_prompt(window=_make_window(), **bundle))
    assert decoded["window"]["anchor"] == "20260530"
    assert decoded["window"]["mode"] == "day"


def test_dumps_are_deterministic_compact() -> None:
    """The JSON wire format is sorted + compact so prompt_hash is stable."""
    bundle = _empty_bundle()
    a = build_overview_user_prompt(window=_make_window(), **bundle)
    b = build_overview_user_prompt(window=_make_window(), **bundle)
    assert a == b
    # No whitespace beyond the compact separators
    assert ": " not in a
    assert ", " not in a
