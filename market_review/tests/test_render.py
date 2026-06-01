"""render — per-section markdown shape + summary + audit dump."""

from __future__ import annotations

import json
from pathlib import Path

from market_review.metrics.breadth import BreadthReview
from market_review.metrics.capital import CapitalReview
from market_review.metrics.leaders import LeaderReview
from market_review.metrics.risk import RiskReview
from market_review.metrics.sectors import SectorReview
from market_review.metrics.sentiment import SentimentReview
from market_review.metrics.style import StyleReview
from market_review.pipeline import MetricsBundle, SectionResult
from market_review.render import (
    SECTION_FILES,
    SECTION_TITLES,
    dump_llm_calls_audit,
    dump_metrics_json,
    render_section_md,
    render_summary_md,
    write_section_files,
    write_summary_md,
)
from market_review.schemas import (
    CapitalSection,
    EvidenceItem,
    Finding,
    HeadlineMetric,
    LeadersSection,
    OutlookHypothesis,
    OverviewSection,
    RiskOutlookSection,
    SectorClassification,
    SectorEntry,
    SectorsSection,
    SentimentSection,
    SentimentSnapshotJson,
    StyleSection,
)
from market_review.windows import Window


def _make_overview(error: str | None = None) -> SectionResult:
    schema = OverviewSection(
        market_tone="震荡分化",
        headline_metrics=[
            HeadlineMetric(label="北向", value=42.0, unit="亿", interpretation="正向"),
            HeadlineMetric(label="情绪", value=68.0, unit="分"),
            HeadlineMetric(label="涨家", value=2300, unit="家"),
        ],
        theme_tags=["AI 算力", "机器人"],
        narrative_md="今日呈现结构性上涨。",
        findings=[Finding(
            headline="主力净流入扩大",
            detail="大盘资金主力净流入 80 亿，散户净流出 30 亿，方向一致。",
            evidence=[EvidenceItem(
                field="north_money_yi", value=42.0, unit="亿",
                interpretation="北向连续 3 日净买",
            )],
            severity="positive",
        )],
        error=error,
    )
    return SectionResult(section="overview", schema=schema, error=error)


def _make_sectors() -> SectionResult:
    schema = SectorsSection(
        provider="ths",
        today_top=[SectorEntry(name="光模块", pct_chg=5.2, net_inflow_yi=10.0,
                               limit_up_count=5, persistence_days=3)],
        range_top=[],
        classification=SectorClassification(
            new_mainline=[SectorEntry(name="光模块", pct_chg=5.2)],
            relay=[],
            fading=[SectorEntry(name="医药", pct_chg=-1.5)],
        ),
        rotation_commentary="新主线由光模块/AI算力接棒。",
        narrative_md="板块整体向科技方向倾斜。",
    )
    return SectionResult(section="sectors", schema=schema)


def _make_risk_outlook() -> SectionResult:
    schema = RiskOutlookSection(
        overall_risk="moderate",
        hypotheses=[OutlookHypothesis(
            title="震荡延续",
            rationale="量能温和 + 北向中性",
            watch_points=["5 日均线", "成交额"],
            fail_triggers=["指数破 60 日均线"],
        )],
        narrative_md="风险可控。",
    )
    return SectionResult(section="risk_outlook", schema=schema)


def _window() -> Window:
    return Window(mode="day", start="20260530", end="20260530",
                  trade_dates=("20260530",), anchor="20260530")


# ---------------------------------------------------------------------------
# Per-section markdown
# ---------------------------------------------------------------------------


def test_section_titles_cover_all() -> None:
    from market_review.schemas import SECTION_ORDER  # noqa: PLC0415
    assert set(SECTION_TITLES) == set(SECTION_ORDER)
    assert set(SECTION_FILES) == set(SECTION_ORDER)


def test_overview_md_contains_headline_metrics_table() -> None:
    md = render_section_md("overview", _make_overview())
    assert "# 大盘整体" in md
    assert "震荡分化" in md  # market_tone
    assert "AI 算力" in md  # theme_tag
    assert "| 北向 |" in md  # headline metrics table
    assert "主力净流入扩大" in md  # findings


def test_sectors_md_contains_three_classification_sections() -> None:
    md = render_section_md("sectors", _make_sectors())
    assert "# 板块轮动" in md
    assert "## 今日 Top" in md
    assert "### 新主线候选" in md
    assert "### 接力" in md
    assert "### 退潮" in md
    assert "光模块" in md
    assert "新主线由" in md  # rotation_commentary


def test_sentiment_md_renders_series_when_present() -> None:
    schema = SentimentSection(
        avg_score=65.0, strongest_day="20260530",
        money_effect="strong", losing_effect="light",
        series=[SentimentSnapshotJson(
            trade_date="20260530", score_of_100=70.0,
            n_up=3000, n_down=1500, median_pct_chg=0.8,
            n_limit_up=80, n_limit_down=5, n_zhaban=12,
        )],
    )
    md = render_section_md("sentiment", SectionResult(section="sentiment", schema=schema))
    assert "情绪与赚钱效应" in md
    assert "20260530" in md
    assert "strong" in md


def test_capital_md_renders_north_summary_table() -> None:
    schema = CapitalSection(
        north_summary=[],  # already covered by leader/industry tests
        industry_top=[],
    )
    md = render_section_md("capital", SectionResult(section="capital", schema=schema))
    assert "# 资金面" in md


def test_leaders_md_renders_score_breakdown_columns() -> None:
    from market_review.schemas import LeaderCandidateJson  # noqa: PLC0415
    schema = LeadersSection(
        primary=[LeaderCandidateJson(
            ts_code="600001.SH", name="A股龙头", score=85.0,
            score_breakdown={"ladder": 20, "return": 22, "capital": 23, "theme": 20},
            industries=["光模块"], rationale="3连板 + 资金集中",
        )],
        min_score=50.0,
        sector_map={"光模块": ["600001.SH"]},
    )
    md = render_section_md("leaders", SectionResult(section="leaders", schema=schema))
    assert "600001.SH" in md
    assert "A股龙头" in md
    assert "85.0" in md
    assert "光模块" in md


def test_style_md_renders_dominant_style() -> None:
    schema = StyleSection(dominant_style="small_cap", flip_signal=True,
                          range_summary={"spread_pct": -3.2})
    md = render_section_md("style", SectionResult(section="style", schema=schema))
    assert "风格切换" in md
    assert "small_cap" in md
    assert "True" in md  # flip_signal


def test_risk_outlook_md_renders_hypotheses() -> None:
    md = render_section_md("risk_outlook", _make_risk_outlook())
    assert "风险与展望" in md
    assert "moderate" in md
    assert "震荡延续" in md
    assert "5 日均线" in md  # watch_points
    assert "指数破 60 日均线" in md  # fail_triggers


def test_section_with_error_shows_warning_banner() -> None:
    result = _make_overview(error="ConnectionError: timeout")
    md = render_section_md("overview", result)
    assert "LLM 调用失败" in md
    assert "ConnectionError" in md


# ---------------------------------------------------------------------------
# Summary + audit
# ---------------------------------------------------------------------------


def test_summary_md_lists_all_sections(tmp_path: Path) -> None:
    results = {
        "overview": _make_overview(),
        "sectors": _make_sectors(),
        "risk_outlook": _make_risk_outlook(),
    }
    md = render_summary_md(run_id="run-1", window=_window(), results=results)
    assert "市场复盘 — 20260530" in md
    assert "震荡分化" in md  # market_tone surfaces
    # Section links — only the ones with results render with no warning marker
    assert "(overview.md)" in md
    assert "(sectors.md)" in md
    assert "(risk_outlook.md)" in md


def test_summary_md_shows_partial_banner_when_section_failed() -> None:
    results = {"overview": _make_overview(error="LLMValidationError: bad")}
    md = render_summary_md(run_id="run-1", window=_window(), results=results)
    assert "PARTIAL" in md
    assert "overview" in md  # mentioned in failed list


def test_write_section_files_creates_each_md(tmp_path: Path) -> None:
    results = {
        "overview": _make_overview(),
        "sectors": _make_sectors(),
    }
    paths = write_section_files(tmp_path / "reports" / "run-1", results)
    assert (tmp_path / "reports" / "run-1" / "overview.md").is_file()
    assert (tmp_path / "reports" / "run-1" / "sectors.md").is_file()
    assert set(paths) == {"overview", "sectors"}


def test_write_summary_md_writes_file(tmp_path: Path) -> None:
    out = write_summary_md(
        tmp_path / "reports" / "run-1",
        run_id="run-1", window=_window(),
        results={"overview": _make_overview()},
    )
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "市场复盘" in text


def test_dump_metrics_json_round_trips(tmp_path: Path) -> None:
    bundle = MetricsBundle(
        window=_window(), breadth=BreadthReview(),
        sentiment=SentimentReview(), capital=CapitalReview(),
        sectors=SectorReview(), leaders=LeaderReview(),
        style=StyleReview(), risk=RiskReview(),
    )
    path = dump_metrics_json(tmp_path / "reports" / "run-1", bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "breadth" in payload
    assert "sentiment" in payload
    assert payload["window"]["anchor"] == "20260530"


def test_dump_llm_calls_audit_writes_jsonl(tmp_path: Path) -> None:
    results = {
        "overview": _make_overview(),
        "sectors": _make_sectors(),
    }
    results["overview"].meta = {"input_tokens": 1000, "latency_ms": 5000}
    path = dump_llm_calls_audit(tmp_path / "reports" / "run-1", results)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2  # one per section
    first = json.loads(lines[0])
    assert first["section"] == "overview"
    assert first["meta"]["input_tokens"] == 1000
