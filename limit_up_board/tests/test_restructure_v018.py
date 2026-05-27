"""v0.18 restructure: 强势筛选 → 强势分析（全量进连板预测）+ 确定性全局重排 + 双结论。

Pins the behavioural contract the user asked for:
  * 连板预测 runs over ALL analysed candidates, not just ``selected==true`` —
    removing the funnel-amplification that turned small screening-score noise
    into a wholly different prediction pool across reruns.
  * ``selected`` survives only as an advisory "强势推荐" label.
  * 全局重排 (Step 4.5) is now deterministic (sort by ``-continuation_score, ts_code``),
    so an identical prediction set always yields an identical global order.
  * The report shows BOTH the strong-analysis verdict and the continuation verdict
    per stock (dual conclusion).
"""

from __future__ import annotations

from datetime import datetime, timezone

from deeptrade.core.run_status import RunStatus

from limit_up_board.data import Round1Bundle, SectorStrength
from limit_up_board.pipeline import build_final_ranking_deterministic
from limit_up_board.report.builder import build_strategy_report
from limit_up_board.render import render_summary_md
from limit_up_board.schemas import (
    ContinuationCandidate,
    EvidenceItemStrict,
    StrongCandidate,
)


def _ev() -> EvidenceItemStrict:
    return EvidenceItemStrict(field="fd_amount_yi", value=1.0, unit="亿", interpretation="ok")


def _strong(ts_code: str, *, score: float, selected: bool) -> StrongCandidate:
    return StrongCandidate(
        candidate_id=ts_code,
        ts_code=ts_code,
        name=f"S{ts_code[:3]}",
        selected=selected,
        score=score,
        strength_level="high" if score >= 70 else "medium",
        rationale=f"{ts_code} 强势分析",
        evidence=[_ev()],
        risk_flags=[],
        missing_data=[],
    )


def _pred(ts_code: str, *, rank: int, score: float, pred: str = "watchlist") -> ContinuationCandidate:
    return ContinuationCandidate(
        candidate_id=ts_code,
        ts_code=ts_code,
        name=f"S{ts_code[:3]}",
        rank=rank,
        continuation_score=score,
        confidence="medium",
        prediction=pred,  # type: ignore[arg-type]
        rationale=f"{ts_code} 连板分析",
        key_evidence=[_ev()],
        next_day_watch_points=["w"],
        failure_triggers=["f"],
        missing_data=[],
    )


def _bundle() -> Round1Bundle:
    candidates = [
        {"candidate_id": c, "ts_code": c, "name": f"S{c[:3]}", "close_yuan": 10.0,
         "industry": "测试", "lgb_score": None, "lgb_decile": None}
        for c in ("000001.SZ", "000002.SZ", "000003.SZ")
    ]
    return Round1Bundle(
        trade_date="20260527",
        next_trade_date="20260528",
        candidates=candidates,
        market_summary={},
        sector_strength=SectorStrength(source="unavailable", data={}),
        data_unavailable=[],
        lgb_model_id=None,
    )


# ---------------------------------------------------------------------------
# 确定性全局重排
# ---------------------------------------------------------------------------


def test_deterministic_final_ranking_is_order_invariant() -> None:
    """Same prediction set in any input order → identical final ranking."""
    a = [_pred("000003.SZ", rank=1, score=60.0),
         _pred("000001.SZ", rank=2, score=80.0),
         _pred("000002.SZ", rank=3, score=80.0)]
    b = list(reversed(a))

    fr_a = build_final_ranking_deterministic(a, trade_date="20260527", next_trade_date="20260528")
    fr_b = build_final_ranking_deterministic(b, trade_date="20260527", next_trade_date="20260528")

    order_a = [(f.final_rank, f.ts_code) for f in fr_a.finalists]
    order_b = [(f.final_rank, f.ts_code) for f in fr_b.finalists]
    assert order_a == order_b
    # -score desc, ts_code asc as tie-break → 80/000001, 80/000002, 60/000003
    assert order_a == [(1, "000001.SZ"), (2, "000002.SZ"), (3, "000003.SZ")]
    assert all(f.delta_vs_batch == "kept" for f in fr_a.finalists)


# ---------------------------------------------------------------------------
# 双结论：报告同时展示强势分析 + 连板预测
# ---------------------------------------------------------------------------


def test_report_shows_all_analyzed_and_dual_conclusion() -> None:
    analyzed = [
        _strong("000001.SZ", score=80.0, selected=True),
        _strong("000002.SZ", score=40.0, selected=False),  # NOT recommended
        _strong("000003.SZ", score=55.0, selected=False),
    ]
    predictions = [
        _pred("000001.SZ", rank=1, score=78.0, pred="top_candidate"),
        _pred("000002.SZ", rank=2, score=30.0, pred="avoid"),
        _pred("000003.SZ", rank=3, score=50.0, pred="watchlist"),
    ]
    rpt = build_strategy_report(
        status=RunStatus.SUCCESS,
        bundle=_bundle(),
        selected=[c for c in analyzed if c.selected],
        analyzed=analyzed,
        predictions=predictions,
        final_ranking=None,
        run_id="r",
        generated_at=datetime.now(timezone.utc),
    )
    # 强势分析段覆盖全部 3 只（不再只列 selected）
    assert len(rpt.step2_screening) == 3
    assert rpt.meta.counts.analyzed == 3
    assert rpt.meta.counts.selected == 1  # only 000001 is 强势推荐
    # advisory flag preserved
    by_code = {s.code: s for s in rpt.step2_screening}
    assert by_code["000001.SZ"].strongRecommended is True
    assert by_code["000002.SZ"].strongRecommended is False

    # 连板卡片并列携带强势分析结论
    all_cards = (
        rpt.step4_prediction.top_candidate
        + rpt.step4_prediction.watchlist
        + rpt.step4_prediction.avoid
    )
    assert len(all_cards) == 3
    card1 = next(c for c in all_cards if c.code == "000001.SZ")
    assert card1.strongScore == 80.0
    assert card1.strongLevel == "强"
    assert card1.strongRationale == "000001.SZ 强势分析"
    assert card1.strongRecommended is True
    # 一只非强势推荐的标的仍出现在预测结果中（未被淘汰）
    assert any(c.code == "000002.SZ" for c in all_cards)


def test_debate_per_provider_dump_uses_analyzed(tmp_path) -> None:
    """Guards the v0.18 fix: the debate per-provider report path referenced
    non-existent ``r.r1_result`` / ``r.r2_result`` (latent AttributeError) and
    must now read ``screening_result.analyzed`` / ``prediction_result``."""
    from limit_up_board.pipeline import RoundResult
    from limit_up_board.render import write_report
    from limit_up_board.runner import ProviderDebateResult

    analyzed = [
        _strong("000001.SZ", score=80.0, selected=True),
        _strong("000002.SZ", score=40.0, selected=False),
    ]
    sr = RoundResult(success_batches=1, candidates_in=2)
    sr.analyzed = analyzed
    sr.selected = [c for c in analyzed if c.selected]
    pr = RoundResult(success_batches=1, candidates_in=2)
    pr.predictions = [_pred("000001.SZ", rank=1, score=78.0, pred="top_candidate")]
    pdr = ProviderDebateResult(
        provider="deepseek", screening_result=sr, prediction_result=pr
    )

    write_report(
        run_id="00000000-0000-0000-0000-0000000000de",
        status=RunStatus.SUCCESS,
        bundle=_bundle(),
        selected=sr.selected,
        predictions=pr.predictions,
        final_ranking=None,
        debate_results=[pdr],
        reports_root=tmp_path,
    )
    import json as _json

    dump = (
        tmp_path
        / "00000000-0000-0000-0000-0000000000de"
        / "debate"
        / "deepseek"
        / "round1_strong_targets.json"
    )
    rows = _json.loads(dump.read_text(encoding="utf-8"))
    # all analysed candidates dumped (not just the recommended subset)
    assert {r["ts_code"] for r in rows} == {"000001.SZ", "000002.SZ"}


def test_summary_md_dual_conclusion_columns() -> None:
    analyzed = [
        _strong("000001.SZ", score=80.0, selected=True),
        _strong("000002.SZ", score=40.0, selected=False),
    ]
    predictions = [
        _pred("000001.SZ", rank=1, score=78.0, pred="top_candidate"),
        _pred("000002.SZ", rank=2, score=30.0, pred="avoid"),
    ]
    md = render_summary_md(
        status=RunStatus.SUCCESS,
        bundle=_bundle(),
        selected=[c for c in analyzed if c.selected],
        analyzed=analyzed,
        predictions=predictions,
        final_ranking=None,
    )
    # 强势分析段标题反映全量 + 推荐数
    assert "强势标的分析（分析 2/3 只，强势推荐 1 只 ★" in md
    # 双结论列存在，且两只都出现在连板预测里
    assert "强势分析(分/级)" in md
    assert md.count("000001.SZ") >= 2
    assert md.count("000002.SZ") >= 2
