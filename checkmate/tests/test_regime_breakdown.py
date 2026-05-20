"""Regime breakdown card tests (PR-6.3).

Verifies that build_report aggregates closed positions into per-regime
buckets carrying all 4 metrics (n_trades / total_pnl / win_rate /
avg_hold_days) and that the Rich dashboard's regime breakdown card
renders them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from checkmate import paths
from checkmate.report import build_report, to_html, to_markdown
from checkmate.runtime import CheckmateRuntime
from checkmate.ui.layout import render_regime_breakdown_card


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()
    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())

    rt = CheckmateRuntime(db=db, config=None, tushare=None)  # type: ignore[arg-type]
    yield rt
    db.close()


_TEST_RUN_ID = "00000000-0000-0000-0000-000000000001"


def _seed_backtest_run(db, run_id: str = _TEST_RUN_ID) -> None:
    db.execute(
        """
        INSERT INTO checkmate_backtest_runs
            (run_id, config_hash, code_version, start_date, end_date,
             status, config_json, metrics_json)
        VALUES (?, 'h0', 'v', '20240101', '20240331', 'success', ?, '{}')
        """,
        [run_id, json.dumps({"initial_cash": 1_000_000.0})],
    )


def _seed_position(
    db, run_id: str, *,
    ts_code: str, entry_date: str, exit_date: str,
    entry_price: float, exit_price: float, shares: int = 100,
) -> None:
    db.execute(
        """
        INSERT INTO checkmate_positions
            (ts_code, entry_date, entry_price_raw, entry_price_qfq, shares,
             stop_price, state, risk_R, peak_pnl_R, exit_date, exit_price_raw,
             exit_reason, run_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'closed', ?, NULL, ?, ?, 'hard_stop', ?,
                CURRENT_TIMESTAMP)
        """,
        [ts_code, entry_date, entry_price, entry_price, shares,
         entry_price * 0.95, entry_price - entry_price * 0.95,
         exit_date, exit_price, run_id],
    )


def _seed_regime(db, trade_date: str, regime: str) -> None:
    db.execute(
        """
        INSERT INTO checkmate_regime_daily
            (trade_date, regime, exposure_cap, payload_json, created_at)
        VALUES (?, ?, 1.0, '{}', CURRENT_TIMESTAMP)
        """,
        [trade_date, regime],
    )


# ---------------------------------------------------------------------------
# build_report — aggregates 4 metrics per regime bucket
# ---------------------------------------------------------------------------


def test_by_regime_carries_all_four_metrics(rt) -> None:
    run_id = _TEST_RUN_ID
    _seed_backtest_run(rt.db, run_id)

    # Two positions in 'strong' regime, one win + one loss.
    _seed_regime(rt.db, "20240105", "strong")
    _seed_position(rt.db, run_id, ts_code="A.SH",
                   entry_date="20240105", exit_date="20240115",
                   entry_price=10.0, exit_price=11.0, shares=100)
    _seed_regime(rt.db, "20240108", "strong")
    _seed_position(rt.db, run_id, ts_code="B.SH",
                   entry_date="20240108", exit_date="20240120",
                   entry_price=20.0, exit_price=19.0, shares=100)
    # One position in 'neutral' regime, win.
    _seed_regime(rt.db, "20240201", "neutral")
    _seed_position(rt.db, run_id, ts_code="C.SH",
                   entry_date="20240201", exit_date="20240220",
                   entry_price=15.0, exit_price=16.5, shares=100)

    payload = build_report(rt, run_id)

    assert "strong" in payload.by_regime
    assert "neutral" in payload.by_regime

    strong = payload.by_regime["strong"]
    assert strong["n_trades"] == 2
    assert strong["wins"] == 1
    assert strong["win_rate"] == pytest.approx(0.5)
    # PnL = (11-10)*100 + (19-20)*100 = 100 + (-100) = 0
    assert strong["total_pnl"] == pytest.approx(0.0)
    # Hold days: (15-5)=10  +  (20-8)=12  →  avg 11
    assert strong["avg_hold_days"] == pytest.approx(11.0)

    neutral = payload.by_regime["neutral"]
    assert neutral["n_trades"] == 1
    assert neutral["win_rate"] == pytest.approx(1.0)
    assert neutral["total_pnl"] == pytest.approx(150.0)  # (16.5-15)*100
    assert neutral["avg_hold_days"] == pytest.approx(19.0)  # 20240220-20240201


# ---------------------------------------------------------------------------
# to_markdown / to_html include avg_hold_days
# ---------------------------------------------------------------------------


def test_markdown_renders_avg_hold_days_column(rt) -> None:
    run_id = _TEST_RUN_ID
    _seed_backtest_run(rt.db, run_id)
    _seed_regime(rt.db, "20240105", "strong")
    _seed_position(rt.db, run_id, ts_code="A.SH",
                   entry_date="20240105", exit_date="20240120",
                   entry_price=10.0, exit_price=11.0)
    payload = build_report(rt, run_id)
    md = to_markdown(payload)
    assert "## By regime" in md
    assert "avg_hold_days" in md


def test_html_renders_avg_hold_days_column(rt) -> None:
    run_id = _TEST_RUN_ID
    _seed_backtest_run(rt.db, run_id)
    _seed_regime(rt.db, "20240105", "strong")
    _seed_position(rt.db, run_id, ts_code="A.SH",
                   entry_date="20240105", exit_date="20240120",
                   entry_price=10.0, exit_price=11.0)
    payload = build_report(rt, run_id)
    html = to_html(payload)
    assert ">avg_hold_days<" in html


# ---------------------------------------------------------------------------
# Dashboard layout: render_regime_breakdown_card
# ---------------------------------------------------------------------------


def test_regime_breakdown_card_renders_four_columns() -> None:
    by_regime = {
        "strong":  {"n_trades": 5, "total_pnl": 1234.5, "win_rate": 0.6, "avg_hold_days": 14.5},
        "neutral": {"n_trades": 3, "total_pnl": -100.0, "win_rate": 0.33, "avg_hold_days": 9.0},
    }
    panel = render_regime_breakdown_card(by_regime)
    console = Console(record=True, width=120, force_terminal=True)
    with console.capture() as cap:
        console.print(panel)
    out = cap.get()
    assert "Regime breakdown" in out
    # All 5 columns (regime/n_trades/total_pnl/win_rate/avg_hold) appear
    for col_label in ("regime", "n_trades", "total_pnl", "win_rate", "avg_hold"):
        assert col_label in out
    # Data row values
    assert "strong" in out
    assert "neutral" in out
    assert "1,234.50" in out


def test_regime_breakdown_card_empty_dict_renders_header_only() -> None:
    panel = render_regime_breakdown_card({})
    console = Console(record=True, width=80, force_terminal=True)
    with console.capture() as cap:
        console.print(panel)
    out = cap.get()
    assert "Regime breakdown" in out
    # Empty body has just the header row; no specific regime names
    assert "strong" not in out
    assert "neutral" not in out
