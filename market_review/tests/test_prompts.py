"""prompts — hard discipline présent + prev_context injection."""

from __future__ import annotations

import json

import pytest

from market_review.metrics.breadth import BreadthReview, BreadthSnapshot
from market_review.metrics.capital import (
    CapitalReview,
    HsgtTopRow,
    IndustryFlowRow,
    MktFlowRow,
    NorthFlowRow,
    StockFlowRow,
)
from market_review.metrics.leaders import LeaderReview
from market_review.metrics.risk import RiskReview
from market_review.metrics.sectors import SectorEntry, SectorMatrix, SectorReview
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


def test_sectors_user_prompt_drops_matrix() -> None:
    """v0.1.12 fix — :class:`SectorMatrix` is a (~500 boards) × (window days)
    pct_chg grid. Shipping it into the user prompt verbatim bloated the
    request body to the point where qwen-plus's streaming gateway closed
    the chunked-transfer connection mid-response (``httpx.RemoteProtocolError:
    peer closed connection without sending complete message body``). The
    error escapes the framework's ``(APITimeoutError, APIError)`` transport
    catch (httpx protocol errors don't subclass either), so tenacity does
    NOT retry and the section fails immediately.

    Output schema :class:`SectorsSection` consumes only ``today_top`` /
    ``range_top`` / ``classification`` / ``rotation_commentary`` /
    ``narrative_md`` / ``findings`` / ``error`` — none reference
    ``matrix``, so dropping it is lossless. Pre-aggregated board entries
    already carry the only per-board fact the prompt cares about
    (``pct_chg`` / ``persistence_days``).
    """
    matrix = SectorMatrix(
        sectors=["883424.TI"], sector_names=["光模块"],
        trade_dates=["20260530"], values=[[5.0]],
        cum_pct_chg=[5.0], persistence_days=[1],
    )
    sectors = SectorReview(
        today_top=[SectorEntry(ts_code="883424.TI", name="光模块",
                               pct_chg=5.0, persistence_days=1)],
        matrix=matrix,
    )
    out = build_sectors_user_prompt(
        window=_make_window(), sectors=sectors, prev_context={},
    )
    decoded = json.loads(out)
    assert "matrix" not in decoded["sectors"], decoded["sectors"]


def test_sectors_user_prompt_size_under_budget() -> None:
    """Real-world regression guard: with full THS catalog (~500 boards) ×
    a 5-day window, the serialized user prompt must stay well under the
    8 KB ceiling that v0.1.11 routinely blew past via ``matrix``. 8 KB is
    a generous upper bound — historic non-sectors prompts measure 2–4 KB.
    """
    n_boards, n_days = 500, 5
    trade_dates = [f"2026053{i}" for i in range(n_days)]
    entries = [
        SectorEntry(
            ts_code=f"88{3000 + i:04d}.TI",
            name=f"板块{i:03d}",
            pct_chg=float(i % 10),
            persistence_days=i % 3,
        )
        for i in range(10)
    ]
    matrix = SectorMatrix(
        sectors=[f"88{3000 + i:04d}.TI" for i in range(n_boards)],
        sector_names=[f"板块{i:03d}" for i in range(n_boards)],
        trade_dates=trade_dates,
        values=[[float((i + d) % 10) for d in range(n_days)] for i in range(n_boards)],
        cum_pct_chg=[float(i % 10) for i in range(n_boards)],
        persistence_days=[i % 3 for i in range(n_boards)],
    )
    sectors = SectorReview(
        today_top=entries[:10], range_top=entries[:10],
        new_mainline=entries[:5], relay=entries[5:8], fading=entries[8:],
        matrix=matrix,
    )
    out = build_sectors_user_prompt(
        window=_make_window(), sectors=sectors, prev_context={},
    )
    # Without the matrix drop this same fixture serializes to >50 KB.
    assert len(out.encode("utf-8")) < 8_192, (
        f"sectors user prompt too large: {len(out.encode('utf-8'))} bytes"
    )


def test_sectors_system_prompt_no_longer_references_matrix() -> None:
    """v0.1.12 — once ``matrix`` is dropped from the input payload, the
    system prompt's "+ 板块强度矩阵" reference becomes a contract leak: it
    promises the LLM input data it will not actually receive. The prompt
    now lists only the inputs it really gets: per-board pct_chg + 持续性
    天数.
    """
    sectors_prompt = SECTION_SYSTEM_PROMPTS["sectors"]
    assert "板块强度矩阵" not in sectors_prompt
    assert "板块持续性天数" in sectors_prompt


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


def _make_sentiment_breadth_pair() -> tuple[SentimentReview, BreadthReview]:
    """Matched-date sentiment + breadth fixture for v0.1.11 series tests."""
    sentiment = SentimentReview(
        series=[SentimentSnapshot(
            trade_date="20260530", median_pct_chg=0.5, mean_pct_chg=0.2,
            pos_ratio=0.6, top_ratio=0.1, crash_ratio=0.05,
            limit_up_intensity=0.02, connection_health=0.8,
            n_lhb=20, north_money_yi=15.0, score_0_100=65.0,
        )],
        avg_score=65.0, strongest_day="20260530", weakest_day="20260530",
    )
    breadth = BreadthReview(series=[BreadthSnapshot(
        trade_date="20260530", n_total=5000,
        n_up=3000, n_down=1800, n_flat=200,
        n_up5pct=500, n_down5pct=250,
        n_limit_up=80, n_limit_down=10, n_zhaban=12,
        up_ladder={2: 30, 3: 15, 4: 5}, n_lhb=20,
        total_amount_yi=8000.0, index_returns={},
    )])
    return sentiment, breadth


def test_sentiment_user_prompt_series_matches_output_schema_shape() -> None:
    """v0.1.11 fix — `SentimentSnapshotJson` schema requires per-day fields
    (``nUp`` / ``nDown`` / ``nLimitUp`` / ``nLimitDown`` / ``nZhaban``)
    that live on :class:`BreadthSnapshot`, not :class:`SentimentSnapshot`.
    Prior to this fix, ``build_sentiment_user_prompt`` only fed sentiment
    in, so the LLM saw input fields (``pos_ratio`` / ``connection_health``
    / ``score_0_100`` …) that didn't appear anywhere in the output schema
    and didn't see the counter fields the schema required — qwen-plus
    rationally replayed the input shape as ``series[*]`` (8×
    ``extra_forbidden`` + 6× ``field-required``).

    The fix zips breadth + sentiment by ``trade_date`` and emits exactly
    the 8 keys ``SentimentSnapshotJson`` consumes (snake_case form, since
    ``populate_by_name=True`` lets the LLM either pass through or
    camelCase-flip).
    """
    sentiment, breadth = _make_sentiment_breadth_pair()
    out = build_sentiment_user_prompt(
        window=_make_window(), sentiment=sentiment, breadth=breadth, prev_context={},
    )
    decoded = json.loads(out)
    series = decoded["sentiment"]["series"]
    assert len(series) == 1, series
    entry = series[0]
    # Exactly the 8 keys the output schema consumes, no more, no less.
    assert set(entry) == {
        "trade_date", "score_of_100",
        "n_up", "n_down", "median_pct_chg",
        "n_limit_up", "n_limit_down", "n_zhaban",
    }, sorted(entry)
    assert entry["trade_date"] == "20260530"
    assert entry["score_of_100"] == 65.0
    # Counters must come from breadth — the LLM would have no way to
    # invent these from sentiment alone.
    assert entry["n_up"] == 3000
    assert entry["n_down"] == 1800
    assert entry["n_limit_up"] == 80
    assert entry["n_limit_down"] == 10
    assert entry["n_zhaban"] == 12
    assert entry["median_pct_chg"] == 0.5


def test_sentiment_user_prompt_strips_input_side_ratios() -> None:
    """The internal sentiment ratios (``pos_ratio`` / ``connection_health`` /
    ``score_0_100`` etc.) used to derive the 0-100 score are intermediate
    inputs to the metric — they must NOT appear in the user prompt's
    ``series[*]``, otherwise the LLM mirrors them back into the output
    (the v0.1.11 failure mode). They can still be quoted by the LLM as
    ``findings[*].evidence`` if it wants narrative support, but only via
    ``avgScore`` / day labels at the top level."""
    sentiment, breadth = _make_sentiment_breadth_pair()
    out = build_sentiment_user_prompt(
        window=_make_window(), sentiment=sentiment, breadth=breadth, prev_context={},
    )
    decoded = json.loads(out)
    entry = decoded["sentiment"]["series"][0]
    for forbidden in (
        "pos_ratio", "top_ratio", "crash_ratio",
        "limit_up_intensity", "connection_health",
        "n_lhb", "north_money_yi", "mean_pct_chg",
        "score_0_100",  # the input-side digit-separated form — schema uses score_of_100
    ):
        assert forbidden not in entry, f"sentiment.series[0] leaked {forbidden!r}: {entry!r}"


def test_sentiment_user_prompt_preserves_top_level_aggregates() -> None:
    """avgScore / strongestDay / weakestDay are top-level aggregates the
    LLM copies through verbatim."""
    sentiment, breadth = _make_sentiment_breadth_pair()
    out = build_sentiment_user_prompt(
        window=_make_window(), sentiment=sentiment, breadth=breadth, prev_context={},
    )
    decoded = json.loads(out)
    payload = decoded["sentiment"]
    assert payload["avg_score"] == 65.0
    assert payload["strongest_day"] == "20260530"
    assert payload["weakest_day"] == "20260530"


def test_sentiment_user_prompt_tolerates_missing_breadth_day() -> None:
    """When sentiment has a date breadth doesn't, fall back to zero counts
    rather than crashing. This is rare in practice (breadth is sentiment's
    upstream) but the helper shouldn't KeyError on unaligned series."""
    sentiment = SentimentReview(series=[SentimentSnapshot(
        trade_date="20260530", median_pct_chg=0.1, mean_pct_chg=0.1,
        pos_ratio=0.5, top_ratio=0.05, crash_ratio=0.05,
        limit_up_intensity=0.01, connection_health=0.5,
        n_lhb=10, north_money_yi=None, score_0_100=50.0,
    )])
    out = build_sentiment_user_prompt(
        window=_make_window(), sentiment=sentiment, breadth=BreadthReview(),
        prev_context={},
    )
    decoded = json.loads(out)
    entry = decoded["sentiment"]["series"][0]
    assert entry["n_up"] == 0
    assert entry["n_limit_up"] == 0
    assert entry["n_zhaban"] == 0


def test_sentiment_system_prompt_pins_series_field_list() -> None:
    """v0.1.11 fix — the system prompt must explicitly list the 8 keys
    SentimentSnapshotJson consumes AND blacklist the input-side ratios
    the LLM is tempted to mirror back. Without these the schema-prompt
    gap re-opens the moment someone touches `_serialize_sentiment_for_prompt`.

    Belt-and-suspenders pattern: prompt-builder pre-shapes the input,
    system prompt forbids the wrong shape. Either alone is enough for
    well-behaved models; both together catch qwen-plus.
    """
    sentiment_prompt = SECTION_SYSTEM_PROMPTS["sentiment"]
    # Required series field names — each must be called out explicitly.
    for required in (
        "``tradeDate``", "``scoreOf100``", "``nUp``", "``nDown``",
        "``medianPctChg``", "``nLimitUp``", "``nLimitDown``", "``nZhaban``",
    ):
        assert required in sentiment_prompt, f"missing required key in prompt: {required}"
    # Input-side ratios that previously leaked — must be on the blacklist.
    for forbidden in (
        "``posRatio``", "``connectionHealth``", "``score_0_100``",
    ):
        assert forbidden in sentiment_prompt, f"missing blacklist entry: {forbidden}"
    # The "字母 O 不是数字 0" disambiguator — qwen-plus mirrored back
    # ``score_0_100`` because the input dataclass used the digit-separated
    # form and the output schema uses the of-separated form.
    assert "scoreOf100" in sentiment_prompt
    assert "score_0_100" in sentiment_prompt


def test_hard_discipline_rule_10_pins_error_field_type() -> None:
    """v0.1.11 — alongside the SentimentSnapshotJson 6× field-required +
    8× extra_forbidden, the same qwen-plus response also bombed on
    ``error: []`` (schema is ``str | None``). All 7 sections inherit
    this `error` field via `SectionBase`, so the rule lives in
    HARD_DISCIPLINE not the section-specific prompt.
    """
    for keyword in ("``error``", "``null``", "空数组", "空对象", "空字符串"):
        assert keyword in HARD_DISCIPLINE, f"rule 10 missing keyword: {keyword}"


def test_capital_user_prompt_keys_present() -> None:
    out = build_capital_user_prompt(
        window=_make_window(), capital=CapitalReview(), prev_context={},
    )
    decoded = json.loads(out)
    assert "capital" in decoded


def _make_capital_fixture() -> CapitalReview:
    """CapitalReview with one of every list field populated.

    Covers each shape transformation in ``_serialize_capital_for_prompt``:
    north_series + mkt_series → north_summary[*]; north_top10_anchor →
    north_top10_today (net_amount_yi → net_inflow_yi); industry/concept
    pct_change_avg → pct_chg; stock_top_inflow/outflow → stock_top/bottom.
    """
    return CapitalReview(
        north_series=[NorthFlowRow(trade_date="20260530", north_money_yi=41.47)],
        mkt_series=[MktFlowRow(
            trade_date="20260530", main_net_yi=12.5, retail_net_yi=-3.2,
        )],
        north_top10_anchor=[HsgtTopRow(
            trade_date="20260530", ts_code="688048.SH",
            name="源杰科技", net_amount_yi=1.8,
        )],
        industry_top=[IndustryFlowRow(
            name="半导体", net_inflow_yi=22.5, pct_change_avg=3.4,
        )],
        concept_top=[IndustryFlowRow(
            name="光模块", net_inflow_yi=18.2, pct_change_avg=4.1,
        )],
        stock_top_inflow=[StockFlowRow(
            ts_code="300750.SZ", name="宁德时代", net_inflow_yi=5.6,
        )],
        stock_top_outflow=[StockFlowRow(
            ts_code="600519.SH", name="贵州茅台", net_inflow_yi=-4.9,
        )],
    )


def test_capital_user_prompt_north_summary_matches_output_schema_shape() -> None:
    """v0.1.13 fix — :class:`CapitalDailyRow` (the item schema for
    ``northSummary[*]``) requires 5 fields keyed by ``trade_date`` /
    ``north_money_yi`` / ``main_net_inflow_yi`` / ``margin_balance_yi`` /
    ``margin_delta_yi``. Prior to this fix, ``build_capital_user_prompt``
    only did ``_serialize(capital)``, so the LLM saw ``capital.north_series[*]``
    (input-side key + inner ``trade_date`` / ``north_money_yi``) and had to
    invent the mapping to ``northSummary[*]``. Once renaming is in play,
    qwen-plus rationally generalized "all 5 sibling capital fields use
    ``netInflowYi``, so the 6th does too" → emitted
    ``"northSummary":[{"trade_date":"...","net_inflow_yi":41.47}]``, schema
    rejected with ``extra_forbidden``.

    Fix mirrors v0.1.11 sentiment: zip ``north_series`` + ``mkt_series`` by
    trade_date so the helper emits exactly the 3 keys ``CapitalDailyRow``
    consumes when there's data for them (snake_case, since
    ``populate_by_name=True`` lets the LLM either pass through or
    camelCase-flip). ``margin_*`` has no v0.1 upstream source and is
    omitted entirely (schema allows ``None``).
    """
    capital = _make_capital_fixture()
    out = build_capital_user_prompt(
        window=_make_window(), capital=capital, prev_context={},
    )
    decoded = json.loads(out)
    payload = decoded["capital"]
    assert "north_summary" in payload, payload
    summary = payload["north_summary"]
    assert len(summary) == 1, summary
    entry = summary[0]
    # Exactly the keys CapitalDailyRow consumes when upstream has the data.
    # margin_* omitted (no source); main_net_inflow_yi present (mkt_series has it).
    assert set(entry) == {
        "trade_date", "north_money_yi", "main_net_inflow_yi",
    }, sorted(entry)
    assert entry["trade_date"] == "20260530"
    assert entry["north_money_yi"] == 41.47
    # Renamed from MktFlowRow.main_net_yi → CapitalDailyRow.main_net_inflow_yi.
    assert entry["main_net_inflow_yi"] == 12.5
    # The exact hallucination we're guarding against.
    assert "net_inflow_yi" not in entry, entry


def test_capital_user_prompt_strips_input_side_north_keys() -> None:
    """The internal ``CapitalReview.north_series`` / ``mkt_series`` /
    ``north_top10_anchor`` / ``mkt_series[*].main_net_yi`` /
    ``mkt_series[*].retail_net_yi`` / ``north_top10_anchor[*].net_amount_yi``
    / ``industry_top[*].pct_change_avg`` are intermediate inputs to the
    metric — they must NOT appear in the user prompt under those names,
    otherwise the LLM either mirrors them back (v0.1.10 / v0.1.11 / v0.1.13
    failure mode) or burns budget renaming and rationalizing.

    Pre-shaping renames every input-side key to its output schema name
    before the LLM sees it, so the LLM's only job is verbatim copy.
    ``mkt_series.retail_net_yi`` is dropped entirely — CapitalSection
    schema has no slot for it (mkt 散户 net 数据用于 narrative 推导可由
    derived ``main_net_inflow_yi`` 推出，不再单独透传).
    """
    capital = _make_capital_fixture()
    out = build_capital_user_prompt(
        window=_make_window(), capital=capital, prev_context={},
    )
    decoded = json.loads(out)
    payload = decoded["capital"]
    # Output-side allowlist — only these 6 keys appear in `capital` payload.
    assert set(payload) == {
        "north_summary", "north_top10_today",
        "industry_top", "concept_top",
        "stock_top", "stock_bottom",
    }, sorted(payload)
    # Input-side keys must not leak at the capital top level.
    for forbidden in (
        "north_series", "mkt_series",
        "north_top10_anchor", "stock_top_inflow", "stock_top_outflow",
        "industry_bottom", "concept_bottom",
        "north_total_yi", "lhb_series",
    ):
        assert forbidden not in payload, f"capital payload leaked {forbidden!r}"
    # north_summary[*] must not carry input-side leaf names.
    entry = payload["north_summary"][0]
    for forbidden in (
        "net_inflow_yi", "main_net_yi", "retail_net_yi",
        "net_amount_yi", "north_total_yi",
    ):
        assert forbidden not in entry, f"north_summary[0] leaked {forbidden!r}: {entry!r}"
    # north_top10_today[*] uses CapitalLeader (net_inflow_yi), not the
    # input-side HsgtTopRow.net_amount_yi.
    top10_entry = payload["north_top10_today"][0]
    assert set(top10_entry) == {"ts_code", "name", "net_inflow_yi"}, sorted(top10_entry)
    assert top10_entry["net_inflow_yi"] == 1.8  # was net_amount_yi=1.8
    assert "net_amount_yi" not in top10_entry
    # industry_top[*] uses IndustryFlowRowJson.pct_chg, not pct_change_avg.
    ind_entry = payload["industry_top"][0]
    assert set(ind_entry) == {"name", "net_inflow_yi", "pct_chg"}, sorted(ind_entry)
    assert ind_entry["pct_chg"] == 3.4  # was pct_change_avg=3.4
    assert "pct_change_avg" not in ind_entry


def test_capital_system_prompt_pins_north_summary_field_list() -> None:
    """v0.1.13 fix — the system prompt must explicitly list the 5 keys
    CapitalDailyRow consumes AND explicitly blacklist ``netInflowYi`` in
    ``northSummary[*]`` (the dominant in-section convention and the exact
    hallucination qwen-plus produced).

    Belt-and-suspenders with ``_serialize_capital_for_prompt``: the helper
    pre-shapes the input so the LLM has nothing to mirror; the prompt
    explicitly names the right + wrong keys so a future LLM that ignores
    the input shape still gets corrected. Either alone catches
    well-behaved models; both together catch qwen-plus.

    The capital section sits at the **bottom** of a "local regex" trap:
    five of six neighboring leaves (northTop10Today / industryTop /
    conceptTop / stockTop / stockBottom) all use ``netInflowYi``, but
    ``northSummary[*]`` uniquely uses ``northMoneyYi``. The prompt must
    name this asymmetry out loud — see ``test_sectors_system_prompt_…``
    for the same belt-and-suspenders pattern (v0.1.10) and
    ``test_sentiment_system_prompt_pins_series_field_list`` (v0.1.11).
    """
    capital_prompt = SECTION_SYSTEM_PROMPTS["capital"]
    # Required CapitalDailyRow field names — each must be called out explicitly.
    for required in (
        "``tradeDate``", "``northMoneyYi``", "``mainNetInflowYi``",
        "``marginBalanceYi``", "``marginDeltaYi``",
    ):
        assert required in capital_prompt, f"missing required key in prompt: {required}"
    # The exact hallucination — must be on the blacklist.
    assert "``netInflowYi``" in capital_prompt
    # The "unique exception" disambiguator — qwen-plus generalized "all
    # capital fields use netInflowYi" because the prompt didn't surface
    # that northSummary is the lone exception.
    assert "唯一例外" in capital_prompt or "例外" in capital_prompt
    # The schema name must be referenced so the reason for the exception
    # is anchored to a class, not folklore.
    assert "CapitalDailyRow" in capital_prompt


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
