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
    """All nine discipline rules from design §5.4.2 are present.

    - rule 7 added in v0.1.4 to pin the Finding shape against LLM
      hallucinations that placed narrativeMd inside findings[*].
    - rule 8 added in v0.1.5 to ban hallucinated top-level discriminator
      fields (``section`` / ``sectionName`` / ``type`` / ``kind`` …) after
      qwen-plus emitted ``{"section": "leader_identification", ...}`` in
      LeadersSection.
    - rule 9 added in v0.1.7 to ban prev_context echo (``marketTone`` /
      ``themeTags``) leaking into §2-§7 section top level — qwen-plus had
      mirrored the §1 OverviewSection values back into SentimentSection.
    - rule 3 / 6 strengthened in v0.1.8 after qwen-plus emitted
      ``evidence[*].unit = null`` (categorical evidence with no natural
      unit) and ``evidence[*].value = [..., ...]`` (list of signal names
      crammed into one scalar) in OverviewSection.findings — both rooted
      in an EvidenceItem schema-prompt gap, not free-form hallucination.
    """
    for keyword in ("严禁使用外部搜索", "严禁编造数据", "evidence", "四元组",
                    "仅输出 JSON", "1200 中文字", "标量",
                    "headline", "detail", "severity", "extra_forbidden",
                    "章节标识符", "sectionName", "type", "kind",
                    "prevContext", "marketTone", "themeTags", "语气校准",
                    # v0.1.8 — rule 3 unit branches + rule 6 split-evidence escape hatch
                    "数值型 evidence", "分类型 evidence", "伪单位",
                    "拆成多条 evidence 项"):
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


def test_sectors_system_prompt_excludes_provider_from_allowlist() -> None:
    """v0.1.9 fix — ``provider`` is data-source metadata (MrConfig), not LLM
    output. qwen-plus had filled it with the analyst persona string
    ``"板块轮动分析师"`` because the prompt listed it as required without
    explaining the meaning or supplying a value source. The fix:

    1. Drops ``provider`` from the SECTORS allow-list (8 → 7 fields).
    2. Adds ``provider`` to the 严禁额外加 blacklist alongside
       ``marketTone`` / ``themeTags`` / ``section`` / etc.
    3. Surfaces the reason ("数据源元信息（THS / DC）, 由调用方根据 MrConfig
       填入") so future weak-discipline models know why to skip it.
    """
    sectors_prompt = SECTION_SYSTEM_PROMPTS["sectors"]
    assert "7 个字段" in sectors_prompt
    assert "8 个字段" not in sectors_prompt
    assert "``provider``" in sectors_prompt
    # The blacklist sentence must explicitly forbid provider — search for the
    # 严禁 marker followed by provider somewhere in the same prompt body.
    blacklist_segment = sectors_prompt.split("严禁额外加", maxsplit=1)[-1]
    assert "``provider``" in blacklist_segment
    # The "why" is the part that prevents this exact regression: a future
    # model reading the prompt should learn that provider is config-side
    # metadata, not LLM output.
    assert "MrConfig" in sectors_prompt or "数据源元信息" in sectors_prompt


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


def test_sectors_user_prompt_strips_ts_code_from_entries() -> None:
    """v0.1.10 fix — the internal :class:`metrics.sectors.SectorEntry`
    dataclass carries ``ts_code`` for DB / matrix indexing, but the
    LLM-output :class:`schemas.SectorEntry` deliberately omits it. Prior
    to this fix the prompt builder echoed ``ts_code`` straight into the
    user payload, and qwen-plus replayed it back as ``tsCode`` on every
    today_top / range_top / classification.new_mainline entry (30×
    ``extra_forbidden``). The fix strips ``ts_code`` from each entry in
    all five board lists so the LLM has nothing to mirror.

    ``matrix`` is intentionally left alone — its ``sectors`` axis is a
    parallel list paired with ``sector_names`` + ``values`` and the LLM
    does not confuse it for a SectorEntry array.
    """
    sectors = SectorReview(
        today_top=[
            SectorEntry(ts_code="883424.TI", name="光模块", pct_chg=5.0, persistence_days=2),
        ],
        range_top=[
            SectorEntry(ts_code="883422.TI", name="存储芯片", pct_chg=8.5, persistence_days=3),
        ],
        new_mainline=[
            SectorEntry(ts_code="884282.TI", name="人形机器人", pct_chg=4.2, persistence_days=1),
        ],
        relay=[
            SectorEntry(ts_code="884071.TI", name="AI 算力", pct_chg=3.1, persistence_days=2),
        ],
        fading=[
            SectorEntry(ts_code="700409.TI", name="白酒", pct_chg=-2.0, persistence_days=0),
        ],
    )
    out = build_sectors_user_prompt(
        window=_make_window(), sectors=sectors, prev_context={},
    )
    decoded = json.loads(out)
    sectors_payload = decoded["sectors"]
    for key in ("today_top", "range_top", "new_mainline", "relay", "fading"):
        entries = sectors_payload[key]
        assert entries, f"{key} should be non-empty in this fixture"
        for entry in entries:
            assert "ts_code" not in entry, f"{key} entry leaked ts_code: {entry!r}"
            assert "tsCode" not in entry, f"{key} entry leaked tsCode: {entry!r}"
            # The semantically meaningful fields must still be present.
            assert "name" in entry
            assert "pct_chg" in entry


def test_sectors_system_prompt_forbids_ts_code_in_entries() -> None:
    """v0.1.10 fix — the system prompt must explicitly tell the LLM that
    SectorEntry has no ``tsCode`` field, paired with the prompt-builder
    strip above. Without this prompt-side disclaimer a sufficiently
    confident model could re-introduce ``tsCode`` from earlier turn
    context, the matrix.sectors axis, or upstream prevContext.
    """
    sectors_prompt = SECTION_SYSTEM_PROMPTS["sectors"]
    # The disclaimer must call out both the schema-side absence and the
    # leaderTsCode vs board-ts_code distinction (different concept).
    assert "``tsCode``" in sectors_prompt
    assert "leaderTsCode" in sectors_prompt
    assert "龙头股票" in sectors_prompt
    # Must explicitly warn that even seeing ts_code in input does NOT
    # license echoing it — this is the lesson the LLM keeps missing.
    assert "extra_forbidden" in sectors_prompt


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
