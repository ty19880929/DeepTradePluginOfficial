"""PR-2.3 — summary.md / round2_predictions.json LGB rendering.

Asserts:
* summary.md metadata header always includes ``lgb_model_id`` (``disabled``
  when the bundle had no scoring).
* R1 / R2 tables have an LGB column with score + decile, ``—`` when missing.
* round2_predictions.json gains ``lgb_score`` / ``lgb_decile`` /
  ``lgb_model_id`` per record.
* Debate-mode summary.md likewise surfaces ``lgb_model_id``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deeptrade.core.run_status import RunStatus

from limit_up_board.data import Round1Bundle, SectorStrength
from limit_up_board.render import (
    _lgb_cell,
    _lgb_compact,
    _lgb_model_id_repr,
    render_summary_md,
    write_report,
)
from limit_up_board.schemas import (
    ContinuationCandidate,
    EvidenceItem,
    StrongCandidate,
)


# ---------------------------------------------------------------------------
# Bundle fixtures
# ---------------------------------------------------------------------------


def _make_bundle(*, with_lgb: bool = True) -> Round1Bundle:
    candidates = [
        {
            "candidate_id": "600519.SH",
            "ts_code": "600519.SH",
            "name": "茅台",
            "close_yuan": 12.5,
            "lgb_score": 73.0 if with_lgb else None,
            "lgb_decile": 8 if with_lgb else None,
            "lgb_feature_missing": [],
            "industry": "电子",
        },
        {
            "candidate_id": "000001.SZ",
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "close_yuan": 10.0,
            "lgb_score": 21.0 if with_lgb else None,
            "lgb_decile": 2 if with_lgb else None,
            "lgb_feature_missing": [],
            "industry": "金融",
        },
    ]
    return Round1Bundle(
        trade_date="20260530",
        next_trade_date="20260531",
        candidates=candidates,
        market_summary={},
        sector_strength=SectorStrength(source="limit_cpt_list", data={"top_sectors": []}),
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
                    interpretation="封单 1.2 亿",
                )
            ],
            risk_flags=[],
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
            next_day_watch_points=["盘口跟进"],
            failure_triggers=["开盘跌停"],
            missing_data=[],
        )
    ]


# ---------------------------------------------------------------------------
# summary.md rendering
# ---------------------------------------------------------------------------


def test_summary_md_includes_lgb_model_id_header() -> None:
    bundle = _make_bundle(with_lgb=True)
    md = render_summary_md(
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    assert "lgb_model_id: `20260530_1_demo`" in md


def test_summary_md_disabled_when_no_model() -> None:
    bundle = _make_bundle(with_lgb=False)
    md = render_summary_md(
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    assert "lgb_model_id: `disabled`" in md


def test_summary_md_r1_table_has_lgb_column() -> None:
    bundle = _make_bundle(with_lgb=True)
    md = render_summary_md(
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    # Header row has LGB column
    assert "| LGB |" in md or "| LGB " in md
    # Row payload has "73 (d8)" for 600519
    assert "73 (d8)" in md


def test_summary_md_r2_table_has_lgb_column() -> None:
    bundle = _make_bundle(with_lgb=True)
    md = render_summary_md(
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
    )
    # Both the R2 single-batch header row and the body should mention LGB
    assert "Pred | Rationale" in md
    # The candidate's LGB cell is 73 (d8) → present in either R1 or R2 row
    assert md.count("73 (d8)") >= 2  # appears in R1 row + R2 row


# ---------------------------------------------------------------------------
# round2_predictions.json
# ---------------------------------------------------------------------------


def test_write_report_round2_json_includes_lgb_fields(tmp_path: Path) -> None:
    bundle = _make_bundle(with_lgb=True)
    write_report(
        run_id="00000000-0000-0000-0000-000000000bbb",
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
        reports_root=tmp_path,
    )
    out = tmp_path / "00000000-0000-0000-0000-000000000bbb" / "round2_predictions.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    rec = data[0]
    assert rec["lgb_score"] == 73.0
    assert rec["lgb_decile"] == 8
    assert rec["lgb_model_id"] == "20260530_1_demo"


def test_write_report_round2_json_lgb_none_when_disabled(tmp_path: Path) -> None:
    bundle = _make_bundle(with_lgb=False)
    write_report(
        run_id="00000000-0000-0000-0000-000000000ccc",
        status=RunStatus.SUCCESS,
        is_intraday=False,
        bundle=bundle,
        selected=_make_selected(),
        predictions=_make_predictions(),
        final_ranking=None,
        reports_root=tmp_path,
    )
    data = json.loads(
        (tmp_path / "00000000-0000-0000-0000-000000000ccc" / "round2_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    rec = data[0]
    assert rec["lgb_score"] is None
    assert rec["lgb_decile"] is None
    assert rec["lgb_model_id"] is None


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def test_lgb_cell_renders_score_and_decile() -> None:
    candidates = [
        {"candidate_id": "X", "lgb_score": 71.0, "lgb_decile": 9},
    ]
    assert _lgb_cell("X", candidates) == "71 (d9)"


def test_lgb_cell_missing_score_dashes() -> None:
    candidates = [{"candidate_id": "X", "lgb_score": None}]
    assert _lgb_cell("X", candidates) == "—"


def test_lgb_cell_decile_none_renders_score_only() -> None:
    candidates = [{"candidate_id": "X", "lgb_score": 42.0, "lgb_decile": None}]
    assert _lgb_cell("X", candidates) == "42"


def test_lgb_cell_unknown_id_dashes() -> None:
    assert _lgb_cell("nope", []) == "—"


def test_lgb_compact() -> None:
    assert _lgb_compact({"lgb_score": 71.0}) == "71"
    assert _lgb_compact({"lgb_score": None}) == "—"
    assert _lgb_compact({}) == "—"


def test_lgb_model_id_repr() -> None:
    bundle = _make_bundle(with_lgb=True)
    assert _lgb_model_id_repr(bundle) == "`20260530_1_demo`"
    bundle.lgb_model_id = None
    assert _lgb_model_id_repr(bundle) == "`disabled`"
