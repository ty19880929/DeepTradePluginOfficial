"""End-to-end smoke test for the ``signals`` orchestrator (PR-3.4).

Pipeline under test: ``scan`` (which populates universe/features/regime) →
``run_signals`` (entry detection + exit detection + risk filter). 10
synthetic ts_codes are crafted so several trigger entry signals, then the
``checkmate_signals`` rows are asserted.

The exit path is exercised by seeding an active position whose stop is
breached on ``trade_date``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkmate import paths
from checkmate.config import RegimeConfig
from checkmate.runtime import CheckmateRuntime
from checkmate.scan import ScanParams, run_scan
from checkmate.signals import SignalsParams, run_signals


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# 10 synthetic ts_codes spanning industries; trend slopes set so several
# clear the 40-day breakout high on trade_date.
SYMBOLS = [
    ("600000.SH", "银行",  8.5,   0.005),
    ("600519.SH", "白酒",  1600.0, 0.50),
    ("000001.SZ", "银行",  12.0,  0.010),
    ("002415.SZ", "电子",  30.0,  0.020),
    ("300750.SZ", "电池",  200.0, 0.10),
    ("600036.SH", "银行",  35.0,  -0.005),
    ("000333.SZ", "家电",  60.0,  0.015),
    ("000651.SZ", "家电",  40.0,  0.000),
    ("600276.SH", "医药",  50.0,  0.030),
    ("600887.SH", "食品",  28.0,  -0.010),
]
END_DATE = "20240329"
N_DAYS = 250


def _make_qfq(ts_code: str, base: float, trend: float) -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    closes = np.array([base + trend * i for i in range(N_DAYS)], dtype=float)
    noise = np.linspace(-0.005, 0.005, N_DAYS) * base
    closes = closes + noise
    highs = closes * 1.015
    lows = closes * 0.985
    pre = np.empty_like(closes)
    pre[0] = closes[0] * 0.99
    pre[1:] = closes[:-1]
    df = pd.DataFrame({
        "ts_code": [ts_code] * N_DAYS,
        "trade_date": dates,
        "open": closes, "high": highs, "low": lows,
        "close": closes, "pre_close": pre,
        "adj_factor": [1.0] * N_DAYS,
    })
    for col in ("open", "high", "low", "close", "pre_close"):
        df[f"{col}_qfq"] = df[col]
    return df


def _make_daily_basic(ts_code: str, amount_qianyuan: float = 500_000.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    return pd.DataFrame({
        "ts_code": [ts_code] * N_DAYS,
        "trade_date": dates,
        "amount": [amount_qianyuan] * N_DAYS,
        "turnover_rate": [2.0] * N_DAYS,
        "total_mv": [1e10] * N_DAYS,
    })


def _plant(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _plant_index_rising(index_code: str) -> None:
    n = 200
    dates = pd.bdate_range(end=END_DATE, periods=n).strftime("%Y%m%d").tolist()
    closes = [1000.0 * (1.0 + 0.002 * i) for i in range(n)]
    df = pd.DataFrame({
        "ts_code": [index_code] * n,
        "trade_date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
    })
    _plant(paths.index_daily_cache_dir() / f"{index_code}.parquet", df)


def _seed_status(db, ts_code: str, industry: str, list_date: str = "20100101") -> None:
    db.execute(
        """
        INSERT INTO checkmate_stock_status_history
            (ts_code, as_of_date, list_status, is_st, name, industry,
             list_date, delist_date, raw_event_json, updated_at)
        VALUES (?, ?, 'L', FALSE, ?, ?, ?, NULL, ?, CURRENT_TIMESTAMP)
        """,
        [ts_code, list_date, ts_code, industry, list_date,
         json.dumps({"src": "test"})],
    )


def _trade_cal_frame() -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    return pd.DataFrame({"cal_date": dates, "is_open": [1] * N_DAYS})


class _StubTushare:
    def call(self, api_name: str, **kwargs):
        if api_name == "trade_cal":
            return _trade_cal_frame()
        return pd.DataFrame()


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()
    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())

    for code, industry, base, trend in SYMBOLS:
        _plant(paths.daily_cache_dir() / f"{code}.parquet",
               _make_qfq(code, base, trend))
        _plant(paths.daily_basic_cache_dir() / f"{code}.parquet",
               _make_daily_basic(code))
        _seed_status(db, code, industry)
    _plant_index_rising(RegimeConfig().index_csi_code)
    _plant_index_rising(RegimeConfig().index_hs300_code)

    rt = CheckmateRuntime(db=db, config=None, tushare=_StubTushare())  # type: ignore[arg-type]
    yield rt
    db.close()


def _seed_position(db, ts_code: str, *, entry_date: str = "20240101",
                   entry_price: float = 10.0, stop_price: float = 9.0,
                   state: str = "holding", peak_pnl_R: float = 0.0) -> None:
    db.execute(
        """
        INSERT INTO checkmate_positions
            (ts_code, entry_date, entry_price_raw, entry_price_qfq, shares,
             stop_price, state, risk_R, peak_pnl_R, run_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, CURRENT_TIMESTAMP)
        """,
        [ts_code, entry_date, entry_price, entry_price, 100,
         stop_price, state, entry_price - stop_price, peak_pnl_R],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_signals_e2e_writes_signals_runs_events(rt) -> None:
    """Full pipeline: scan first, then signals. Assert all expected DB writes."""
    run_scan(rt, ScanParams(trade_date=END_DATE))
    log: list[str] = []
    outcome = run_signals(
        rt,
        SignalsParams(trade_date=END_DATE, portfolio_value=10_000_000.0),
        echo=log.append,
    )
    assert outcome.trade_date == END_DATE
    assert outcome.regime in {"strong", "neutral", "weak", "risk"}

    # checkmate_signals has rows
    sig_rows = rt.db.execute(
        "SELECT ts_code, action, signal_type, explain FROM checkmate_signals "
        "WHERE signal_date = ?",
        [END_DATE],
    ).fetchall()
    assert len(sig_rows) >= 1, "signals table should have at least one row"

    # Every signal carries a non-empty explain JSON
    for ts_code, action, signal_type, explain_json in sig_rows:
        assert explain_json, f"explain empty for {ts_code} {action}"
        payload = json.loads(explain_json)
        assert isinstance(payload, dict)
        if action == "enter":
            # The risk filter augments enter signals with sizing fields.
            assert "shares" in payload and payload["shares"] > 0
            assert "stop_price" in payload
            assert "entry_price" in payload

    # checkmate_runs has the signals-run row
    run_row = rt.db.execute(
        "SELECT mode, status, exit_code FROM checkmate_runs WHERE run_id = ?",
        [outcome.run_id],
    ).fetchone()
    assert run_row == ("signals", "success", 0)

    # checkmate_events has step events
    n_events = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_events WHERE run_id = ?",
        [outcome.run_id],
    ).fetchone()[0]
    # RUN_STARTED + 3×(STEP_STARTED + STEP_FINISHED) + RUN_FINISHED = 8
    assert n_events == 8

    # legacy renderer prints step lines
    joined = "\n".join(log)
    assert "[STEP_STARTED]" in joined
    assert "[RUN_FINISHED]" in joined


def test_signals_fails_when_scan_not_run(rt) -> None:
    """No features_daily rows for trade_date → fast-fail with explanatory event."""
    outcome = run_signals(rt, SignalsParams(trade_date=END_DATE))
    # No proposals possible without features
    assert outcome.n_entry_proposals == 0
    run_row = rt.db.execute(
        "SELECT status, exit_code, error FROM checkmate_runs WHERE run_id = ?",
        [outcome.run_id],
    ).fetchone()
    assert run_row[0] == "failed"
    assert run_row[1] == 2
    assert run_row[2] == "features_missing"


def test_signals_emits_exit_for_breached_stop(rt) -> None:
    """Seeded position with stop=20000 above market → hard_stop exit signal."""
    run_scan(rt, ScanParams(trade_date=END_DATE))
    # Pick a synth stock whose end-of-window close is well below 20000.
    _seed_position(rt.db, "600000.SH", entry_date="20240101",
                   entry_price=20.0, stop_price=20000.0, state="holding")
    outcome = run_signals(
        rt, SignalsParams(trade_date=END_DATE, portfolio_value=10_000_000.0),
    )
    assert outcome.n_exits >= 1

    exits = rt.db.execute(
        "SELECT ts_code, signal_type, explain FROM checkmate_signals "
        "WHERE signal_date = ? AND action = 'exit'",
        [END_DATE],
    ).fetchall()
    assert any(ts == "600000.SH" and st == "hard_stop" for ts, st, _ in exits)


def test_signals_t1_position_suppresses_exit_signal(rt) -> None:
    """Position entered today + price below stop → no Signal (T+1 rule)."""
    run_scan(rt, ScanParams(trade_date=END_DATE))
    _seed_position(rt.db, "600519.SH", entry_date=END_DATE,
                   entry_price=2000.0, stop_price=20000.0, state="holding")
    outcome = run_signals(
        rt, SignalsParams(trade_date=END_DATE, portfolio_value=10_000_000.0),
    )
    exits = rt.db.execute(
        "SELECT ts_code FROM checkmate_signals "
        "WHERE signal_date = ? AND action = 'exit'",
        [END_DATE],
    ).fetchall()
    # No exit signal for 600519.SH even though its raw close is far below stop.
    assert all(r[0] != "600519.SH" for r in exits)


def test_signals_persists_rejected_rows_with_cancel_reason(rt) -> None:
    """Rejected entries are stored as action='rejected' for explain drilldown."""
    run_scan(rt, ScanParams(trade_date=END_DATE))
    # Use the regime=risk override to force everything rejected.
    from checkmate.config import RiskConfig  # noqa: PLC0415

    risk_cfg = RiskConfig(
        regime_entry_caps={"strong": 0, "neutral": 0, "weak": 0, "risk": 0},
    )
    outcome = run_signals(
        rt, SignalsParams(
            trade_date=END_DATE, portfolio_value=10_000_000.0,
            risk_cfg=risk_cfg,
        ),
    )
    rejected = rt.db.execute(
        "SELECT ts_code, explain FROM checkmate_signals "
        "WHERE signal_date = ? AND action = 'rejected'",
        [END_DATE],
    ).fetchall()
    if outcome.n_entry_proposals > 0:
        assert len(rejected) >= 1
        for _, ex_json in rejected:
            ex = json.loads(ex_json)
            assert ex.get("cancel_reason"), "cancel_reason must be set on rejected"
