"""PR-2.3 — analyze_summary.md grows an LGB column + lgb_model_id header line."""

from __future__ import annotations

from pathlib import Path

from deeptrade.core.run_status import RunStatus

from volume_anomaly.data import AnalyzeBundle
from volume_anomaly.render import _fmt_lgb, write_analyze_report
from volume_anomaly.schemas import (
    VADimensionScores,
    VAEvidenceItem,
    VATrendCandidate,
)


def _candidate_dict(ts_code: str, lgb_score: float | None) -> dict:
    return {
        "candidate_id": ts_code,
        "ts_code": ts_code,
        "name": f"Co_{ts_code}",
        "tracked_days": 3,
        "lgb_score": lgb_score,
        "lgb_decile": None if lgb_score is None else 8,
        "lgb_feature_missing": [],
    }


def _pred(ts_code: str, rank: int = 1, prediction: str = "imminent_launch") -> VATrendCandidate:
    return VATrendCandidate(
        candidate_id=ts_code,
        ts_code=ts_code,
        name=f"Co_{ts_code}",
        rank=rank,
        launch_score=72.0,
        confidence="high",
        prediction=prediction,
        pattern="breakout",
        washout_quality="sufficient",
        rationale="test",
        dimension_scores=VADimensionScores(
            washout=70, pattern=70, capital=70, sector=70, historical=70, risk=20
        ),
        key_evidence=[
            VAEvidenceItem(field="anomaly_pct_chg", value=5.0, unit="%", interpretation="ok")
        ],
        risk_flags=[],
        next_session_watch=["watch"],
        invalidation_triggers=["fail"],
        missing_data=[],
    )


def test_summary_includes_lgb_model_id_when_loaded(tmp_path: Path) -> None:
    bundle = AnalyzeBundle(
        trade_date="20260601",
        next_trade_date="20260602",
        candidates=[_candidate_dict("000001.SZ", 68.7), _candidate_dict("000002.SZ", 25.0)],
        lgb_model_id="20260601_1_abc",
    )
    write_analyze_report(
        run_id="r1",
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        predictions=[_pred("000001.SZ", 1), _pred("000002.SZ", 2)],
        market_context_summary=None,
        risk_disclaimer=None,
        reports_root=tmp_path,
    )
    summary = (tmp_path / "r1" / "summary.md").read_text(encoding="utf-8")
    assert "lgb_model_id: `20260601_1_abc`" in summary
    assert "| LGB |" in summary
    # Scores render with 0-decimal precision.
    assert " 69 |" in summary
    assert " 25 |" in summary


def test_summary_renders_disabled_when_no_model(tmp_path: Path) -> None:
    bundle = AnalyzeBundle(
        trade_date="20260601",
        next_trade_date="20260602",
        candidates=[_candidate_dict("000001.SZ", None)],
        lgb_model_id=None,
    )
    write_analyze_report(
        run_id="r2",
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        predictions=[_pred("000001.SZ")],
        market_context_summary=None,
        risk_disclaimer=None,
        reports_root=tmp_path,
    )
    summary = (tmp_path / "r2" / "summary.md").read_text(encoding="utf-8")
    assert "lgb_model_id: `disabled`" in summary
    assert " — |" in summary


def test_fmt_lgb_formats_consistently() -> None:
    assert _fmt_lgb(None) == "—"
    assert _fmt_lgb(0) == "0"
    assert _fmt_lgb(68.7) == "69"
    assert _fmt_lgb(100.0) == "100"
    assert _fmt_lgb("not a number") == "—"  # type: ignore[arg-type]
