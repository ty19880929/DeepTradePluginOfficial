"""单 LLM 模式 summary.html 渲染单元测试（v0.9 新增）。

覆盖：
* 冒烟测试：完整字段渲染 + 关键字段出现在 HTML 中
* 空 predictions / 空 selected 不抛异常
* HTML 转义：来自 LLM / 用户输入的 ``<``、``"``、``&`` 必须被转义
* partial_failed banner + 失败批次 chips
* write_report() 把 summary.html 落到 reports/<run_id>/ 下，且大小 > 0
* HTML 渲染失败时 write_report 仍能写出 summary.md（错误降级路径）
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from deeptrade.core.run_status import RunStatus

from limit_up_board.data import Round1Bundle, SectorStrength
from limit_up_board.html_report import (
    _close_str,
    _lgb_cell,
    _num,
    _quantile,
    e,
    render_summary_html,
)
from limit_up_board.render import write_report
from limit_up_board.schemas import (
    ContinuationCandidate,
    EvidenceItem,
    FinalRankItem,
    FinalRankingResponse,
    StrongCandidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bundle(*, with_lgb: bool = True, with_filter: bool = False) -> Round1Bundle:
    candidates = [
        {
            "candidate_id": "600519.SH",
            "ts_code": "600519.SH",
            "name": "茅台",
            "close_yuan": 12.5,
            "float_mv_yi": 45.0,
            "lgb_score": 73.0 if with_lgb else None,
            "lgb_decile": 8 if with_lgb else None,
            "industry": "电子",
        },
        {
            "candidate_id": "000001.SZ",
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "close_yuan": 10.0,
            "float_mv_yi": 80.0,
            "lgb_score": 21.0 if with_lgb else None,
            "lgb_decile": 2 if with_lgb else None,
            "industry": "金融",
        },
    ]
    market_summary: dict = {"limit_step_total": 32}
    if with_filter:
        market_summary["candidate_filter_summary"] = {
            "before": 5,
            "after": 2,
            "min_float_mv_yi": 30.0,
            "max_float_mv_yi": 100.0,
            "max_close_yuan": 15.0,
            "dropped_top3": [
                {
                    "ts_code": "600000.SH",
                    "name": "浦发银行",
                    "float_mv_yi": 180.0,
                    "close_yuan": 8.5,
                    "reasons": ["float_mv_yi > 100"],
                }
            ],
        }
    return Round1Bundle(
        trade_date="20260530",
        next_trade_date="20260531",
        candidates=candidates,
        market_summary=market_summary,
        sector_strength=SectorStrength(
            source="limit_cpt_list", data={"top_sectors": []}
        ),
        data_unavailable=[],
        lgb_model_id="20260530_1_demo" if with_lgb else None,
        lgb_predictions=[],
    )


def _make_selected() -> list[StrongCandidate]:
    return [
        StrongCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            selected=True,
            score=80.0,
            strength_level="high",
            rationale="封板早、量价配合好。",
            evidence=[
                EvidenceItem(
                    field="fd_amount_yi",
                    value=1.2,
                    unit="亿",
                    interpretation="封单 1.2 亿，强势封板",
                )
            ],
            risk_flags=["short_history"],
            missing_data=[],
        )
    ]


def _make_predictions() -> list[ContinuationCandidate]:
    return [
        ContinuationCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            rank=1,
            continuation_score=78.0,
            confidence="high",
            prediction="top_candidate",
            rationale="情绪强、模型分高、上量明确。",
            key_evidence=[
                EvidenceItem(
                    field="lgb_score",
                    value=73.0,
                    unit="无",
                    interpretation="分位 8 / 模型分 73",
                )
            ],
            next_day_watch_points=["盘口跟进", "板块联动"],
            failure_triggers=["开盘跌停"],
            missing_data=[],
        ),
        ContinuationCandidate(
            candidate_id="000001.SZ",
            ts_code="000001.SZ",
            name="平安银行",
            rank=2,
            continuation_score=42.0,
            confidence="low",
            prediction="avoid",
            rationale="模型分低、缺乏题材联动。",
            key_evidence=[
                EvidenceItem(
                    field="lgb_score",
                    value=21.0,
                    unit="无",
                    interpretation="模型分 21，分位 2",
                )
            ],
            next_day_watch_points=["不建议跟进"],
            failure_triggers=["开盘溢价"],
            missing_data=[],
        ),
    ]


def _make_final_ranking() -> FinalRankingResponse:
    return FinalRankingResponse(
        stage="final_ranking",
        trade_date="20260530",
        next_trade_date="20260531",
        finalists=[
            FinalRankItem(
                candidate_id="600519.SH",
                ts_code="600519.SH",
                final_rank=1,
                final_prediction="top_candidate",
                final_confidence="high",
                reason_vs_peers="跨批比对：模型分领先、题材独立性强。",
                delta_vs_batch="kept",
            ),
            FinalRankItem(
                candidate_id="000001.SZ",
                ts_code="000001.SZ",
                final_rank=2,
                final_prediction="avoid",
                final_confidence="low",
                reason_vs_peers="跨批比对：分数最低且无溢价空间。",
                delta_vs_batch="kept",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_renders_single_llm_smoke() -> None:
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=True, with_filter=True),
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
        run_id="run-abc",
    )
    assert html.startswith("<!DOCTYPE html>")
    assert "<html lang=\"zh-CN\">" in html
    # Tailwind CDN inserted
    assert "cdn.tailwindcss.com" in html
    # Header
    assert "打板策略报告" in html
    # Trade date in meta card
    assert "20260530" in html
    assert "20260531" in html
    # Run id
    assert "run-abc" in html
    # Candidates surfaced
    assert "茅台" in html
    assert "平安银行" in html
    assert "600519.SH" in html
    # LGB cells render the "73 (d8)" pattern
    assert "73 (d8)" in html
    # Status disclaimer
    assert "免责声明" in html or "策略研究" in html


def test_renders_with_final_ranking_groups_by_final_prediction() -> None:
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=True),
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=_make_final_ranking(),
        run_id="run-fr",
    )
    # final_rank label appears
    assert "全局重排" in html
    # reason_vs_peers surfaced
    assert "跨批比对" in html
    # Both groups present
    assert "重点关注" in html
    assert "回避" in html


def test_renders_empty_predictions_and_selected() -> None:
    """zero candidates / 早退场景必须能渲染不抛异常。"""
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=False),
        selected=[],
        predictions=[],
        final_ranking=None,
    )
    assert "本轮无强势标的" in html
    assert "本轮无候选标的" in html
    # Header still renders
    assert "打板策略报告" in html


def test_renders_no_candidates_at_all() -> None:
    """Step 1 拿到 0 候选 → bundle.candidates 为空时不应渲染数据快照。"""
    bundle = Round1Bundle(
        trade_date="20260530",
        next_trade_date="20260531",
        candidates=[],
        market_summary={},
        sector_strength=SectorStrength(
            source="industry_fallback", data={"top_sectors": []}
        ),
        data_unavailable=["limit_cpt_list"],
        lgb_model_id=None,
        lgb_predictions=[],
    )
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=bundle,
        selected=[],
        predictions=[],
        final_ranking=None,
    )
    # 数据快照不应出现
    assert "数据快照" not in html
    # data_unavailable chip 出现
    assert "limit_cpt_list" in html
    # 板块来源仍渲染
    assert "industry_fallback" in html


# ---------------------------------------------------------------------------
# HTML escaping — 严格防注入
# ---------------------------------------------------------------------------


def test_html_escaping_in_rationale_and_name() -> None:
    """LLM 字符串可能含 <、&、"，HTML 必须全部转义。"""
    selected = [
        StrongCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name='恶意"<script>',
            selected=True,
            score=80.0,
            strength_level="high",
            rationale='含 <script>alert(1)</script> & "引号"',
            evidence=[
                EvidenceItem(
                    field="x",
                    value="<b>raw</b>",
                    unit="无",
                    interpretation="含 <tag> 的解读",
                )
            ],
            risk_flags=["<flag>"],
            missing_data=[],
        )
    ]
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=False),
        selected=selected,
        predictions=[],
        final_ranking=None,
    )
    # 原文 < 不应出现在数据位（Tailwind CDN 的 <script src> 不算）
    assert "<script>alert(1)</script>" not in html
    # 转义后的形式应出现
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;raw&lt;/b&gt;" in html
    assert "&quot;" in html or "&#x27;" in html
    # name 中的引号也被转义
    assert '恶意&quot;&lt;script&gt;' in html


def test_html_escaping_in_evidence_value() -> None:
    """evidence.value 可能是 list 或 number，e() 必须能 stringify 并转义。"""
    predictions = [
        ContinuationCandidate(
            candidate_id="600519.SH",
            ts_code="600519.SH",
            name="茅台",
            rank=1,
            continuation_score=70.0,
            confidence="medium",
            prediction="watchlist",
            rationale="略",
            key_evidence=[
                EvidenceItem(
                    field="risk_tags",
                    value=["<tag1>", "<tag2>"],
                    unit="none",
                    interpretation="标签解读",
                )
            ],
            next_day_watch_points=["x"],
            failure_triggers=["y"],
            missing_data=[],
        )
    ]
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=False),
        selected=[],
        predictions=predictions,
        final_ranking=None,
    )
    assert "<tag1>" not in html
    assert "&lt;tag1&gt;" in html


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------


def test_partial_failed_banner_with_failed_batches() -> None:
    html = render_summary_html(
        status=RunStatus.PARTIAL_FAILED,
        bundle=_make_bundle(with_lgb=False),
        selected=_make_selected(),
        predictions=[],
        final_ranking=None,
        failed_batch_ids=["初筛#2", "预测#3"],
    )
    assert "PARTIAL" in html
    assert "初筛#2" in html
    assert "预测#3" in html
    assert "失败批次" in html


def test_failed_banner() -> None:
    html = render_summary_html(
        status=RunStatus.FAILED,
        bundle=_make_bundle(with_lgb=False),
        selected=[],
        predictions=[],
        final_ranking=None,
    )
    assert "FAILED" in html


def test_cancelled_banner() -> None:
    html = render_summary_html(
        status=RunStatus.CANCELLED,
        bundle=_make_bundle(with_lgb=False),
        selected=[],
        predictions=[],
        final_ranking=None,
    )
    assert "CANCELLED" in html


def test_success_no_banner() -> None:
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=False),
        selected=[],
        predictions=[],
        final_ranking=None,
    )
    assert "PARTIAL" not in html
    assert "FAILED" not in html


# ---------------------------------------------------------------------------
# LGB distribution section
# ---------------------------------------------------------------------------


def test_lgb_distribution_renders_when_scores_present() -> None:
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=True),
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    assert "LGB 评分分布" in html
    # SVG histogram present
    assert "<svg" in html
    assert "<rect" in html
    # p25 / median / p75 labels
    assert "median" in html


def test_lgb_distribution_skipped_when_no_model() -> None:
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=False),
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    assert "LGB 评分分布" not in html


# ---------------------------------------------------------------------------
# Candidate filter section
# ---------------------------------------------------------------------------


def test_candidate_filter_section_renders_drop_top3() -> None:
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=True, with_filter=True),
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    assert "候选筛选" in html
    assert "浦发银行" in html
    assert "float_mv_yi &gt; 100" in html  # 转义后的 >


# ---------------------------------------------------------------------------
# write_report integration
# ---------------------------------------------------------------------------


def test_write_report_writes_summary_html(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-000000000aaa"
    bundle = _make_bundle(with_lgb=True, with_filter=True)
    write_report(
        run_id=run_id,
        status=RunStatus.SUCCESS,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
        reports_root=tmp_path,
    )
    out = tmp_path / run_id / "summary.html"
    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "茅台" in content
    assert len(content) > 1000  # 至少 1KB 才算真的渲染了


def test_write_report_html_failure_does_not_block_markdown(tmp_path: Path) -> None:
    """如果 render_summary_html 抛错，summary.md / JSON 仍必须落盘。"""
    run_id = "00000000-0000-0000-0000-000000000fff"
    bundle = _make_bundle(with_lgb=True)
    with patch(
        "limit_up_board.render.render_summary_html",
        side_effect=RuntimeError("boom"),
    ):
        write_report(
            run_id=run_id,
            status=RunStatus.SUCCESS,
            bundle=bundle,
            selected=_make_selected(),
            predictions=_make_predictions(),
            final_ranking=None,
            reports_root=tmp_path,
        )
    md = tmp_path / run_id / "summary.md"
    html = tmp_path / run_id / "summary.html"
    assert md.is_file()
    assert not html.is_file()  # HTML 抛错后不应落盘


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_e_helper_handles_none_and_nan() -> None:
    assert e(None) == "—"
    assert e(float("nan")) == "—"
    assert e(12.5) == "12.5"
    assert e("normal") == "normal"
    assert e("<>") == "&lt;&gt;"
    assert e('"') == "&quot;"


def test_close_str() -> None:
    assert _close_str(12.345) == "12.35"
    assert _close_str(None) == "—"
    assert _close_str("not a number") == "—"


def test_num_formatting() -> None:
    assert _num(3.14159, 2) == "3.14"
    assert _num(3.14159, 4) == "3.1416"
    assert _num(None) == "—"
    assert _num("abc") == "—"


def test_lgb_cell_score_and_decile() -> None:
    assert _lgb_cell({"lgb_score": 73.0, "lgb_decile": 8}) == "73 (d8)"
    assert _lgb_cell({"lgb_score": 21.4, "lgb_decile": None}) == "21"
    assert _lgb_cell({"lgb_score": None}) == "—"
    assert _lgb_cell({}) == "—"
    assert _lgb_cell(None) == "—"  # type: ignore[arg-type]


def test_quantile() -> None:
    arr = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _quantile(arr, 0.0) == 10.0
    assert _quantile(arr, 1.0) == 50.0
    assert _quantile(arr, 0.5) == 30.0


def test_generated_at_renders_in_footer() -> None:
    ts = datetime(2026, 5, 22, 15, 30, 0, tzinfo=timezone.utc)
    html = render_summary_html(
        status=RunStatus.SUCCESS,
        bundle=_make_bundle(with_lgb=False),
        selected=[],
        predictions=[],
        final_ranking=None,
        generated_at=ts,
    )
    assert "2026-05-22" in html
