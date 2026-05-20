"""Exit-rule tests — 5 rules × happy path + T+1 + priority + healthy hold (PR-3.2).

Each scenario builds a :class:`Position` with specific entry / stop / peak
values, hands :func:`evaluate_exit` a synthetic today's close, and asserts
``signal_type`` / ``details.action`` / ``triggered``. The top-level
:func:`detect_exit_signals` is exercised with a tiny in-DB position + planted
daily parquet so the cache-driven path is covered too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from checkmate import paths
from checkmate.config import ExitConfig
from checkmate.runtime import CheckmateRuntime
from checkmate.signals import (
    Position,
    Signal,
    detect_exit_signals,
    evaluate_exit,
    query_active_positions,
)


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


def _pos(
    *,
    ts_code: str = "600000.SH",
    entry_date: str = "20240101",
    entry_price: float = 10.0,
    stop_price: float = 9.0,
    state: str = "holding",
    peak_pnl_R: float | None = 0.0,
    risk_R: float | None = None,
) -> Position:
    return Position(
        ts_code=ts_code, entry_date=entry_date,
        entry_price_raw=entry_price, entry_price_qfq=entry_price,
        shares=100, stop_price=stop_price, state=state,
        risk_R=(entry_price - stop_price) if risk_R is None else risk_R,
        peak_pnl_R=peak_pnl_R,
    )


# ---------------------------------------------------------------------------
# Rule 1 — hard_stop
# ---------------------------------------------------------------------------


def test_hard_stop_triggered_when_close_below_stop() -> None:
    ev = evaluate_exit(
        _pos(stop_price=9.0), "20240301",
        today_close_raw=8.5,
    )
    assert ev.triggered is True
    assert ev.signal_type == "hard_stop"
    assert ev.details["action"] == "exit"
    assert any("close<stop" in h for h in ev.hit)


def test_hard_stop_intraday_low_triggers_even_if_close_above() -> None:
    """Intraday low pierced the stop but close recovered — still an exit."""
    ev = evaluate_exit(
        _pos(stop_price=9.0), "20240301",
        today_close_raw=9.50, today_low_raw=8.80,
    )
    assert ev.triggered is True
    assert ev.signal_type == "hard_stop"


def test_hard_stop_t1_blocked_for_same_day_entry() -> None:
    """A position entered today cannot exit today — T+1 rule."""
    ev = evaluate_exit(
        _pos(entry_date="20240301", stop_price=9.0),
        "20240301",
        today_close_raw=8.0,
    )
    assert ev.triggered is False
    assert ev.signal_type == "t1_blocked"
    assert ev.details["t1_blocked"] is True


# ---------------------------------------------------------------------------
# Rule 2 — risk_regime
# ---------------------------------------------------------------------------


def test_risk_regime_triggers_full_exit() -> None:
    ev = evaluate_exit(
        _pos(stop_price=9.0), "20240301",
        today_close_raw=10.5,  # comfortably above stop
        regime="risk",
    )
    assert ev.triggered is True
    assert ev.signal_type == "risk_regime"
    assert ev.details["action"] == "exit"


def test_risk_regime_is_overridden_by_hard_stop_priority() -> None:
    """Both hard stop AND risk regime fire — hard stop wins (priority 1)."""
    ev = evaluate_exit(
        _pos(stop_price=9.0), "20240301",
        today_close_raw=8.5,
        regime="risk",
    )
    assert ev.signal_type == "hard_stop"


# ---------------------------------------------------------------------------
# Rule 3 — defensive_profit (state transition)
# ---------------------------------------------------------------------------


def test_defensive_profit_transitions_state() -> None:
    """Peak was 3.5R, current 1.0R → retraced 2.5R ≥ 1.5R → defensive."""
    pos = _pos(entry_price=10.0, stop_price=9.0, state="holding", peak_pnl_R=3.5)
    ev = evaluate_exit(pos, "20240301", today_close_raw=11.0, regime="neutral")
    # current pnl = (11 - 10) / 1.0R = 1.0R; peak 3.5; retrace 2.5R
    assert ev.triggered is True
    assert ev.signal_type == "defensive_profit"
    assert ev.details["action"] == "defensive"


def test_defensive_profit_skipped_in_defensive_state() -> None:
    """Once already in 'defensive' state, the rule doesn't double-fire — the
    trailing_stop / hard_stop rules govern the eventual exit."""
    pos = _pos(state="defensive", peak_pnl_R=3.5)
    ev = evaluate_exit(pos, "20240301", today_close_raw=11.0, regime="neutral")
    assert ev.signal_type != "defensive_profit"


def test_defensive_profit_requires_high_enough_peak() -> None:
    """peak=2.5R (< 3.0R threshold) doesn't qualify even with big retrace."""
    pos = _pos(state="holding", peak_pnl_R=2.5)
    ev = evaluate_exit(pos, "20240301", today_close_raw=10.5, regime="neutral")
    # pnl_R = 0.5, peak 2.5, retrace 2.0 >= 1.5 but peak < 3.0 → no defensive
    assert ev.signal_type != "defensive_profit"


# ---------------------------------------------------------------------------
# Rule 4 — trailing_stop
# ---------------------------------------------------------------------------


def test_trailing_stop_triggered_by_pullback_from_peak() -> None:
    """Entry=10, stop=9 (R=1). Peak 6R → peak_price=16. Today=12 < 16*0.85=13.6."""
    pos = _pos(
        entry_price=10.0, stop_price=9.0, state="defensive",
        peak_pnl_R=6.0,
    )
    ev = evaluate_exit(pos, "20240301", today_close_raw=12.0, regime="neutral")
    assert ev.triggered is True
    assert ev.signal_type == "trailing_stop"
    assert ev.details["action"] == "exit"
    assert "trailing_stop" in ev.details
    assert "peak_price" in ev.details


def test_trailing_stop_not_triggered_when_close_above_trail() -> None:
    pos = _pos(entry_price=10.0, stop_price=9.0, state="defensive", peak_pnl_R=6.0)
    # 14 > 16*0.85=13.6 → no trailing exit; nothing else fires either
    ev = evaluate_exit(pos, "20240301", today_close_raw=14.0, regime="neutral")
    assert ev.triggered is False
    assert ev.signal_type == "no_exit"


# ---------------------------------------------------------------------------
# Rule 5 — time_exit
# ---------------------------------------------------------------------------


def test_time_exit_when_held_too_long_with_weak_pnl() -> None:
    """Entry 2023-09-01, today 2024-03-01 → ~182 days > 120; pnl_R=0.5 < 1.0."""
    pos = _pos(entry_date="20230901", entry_price=10.0, stop_price=9.0,
               peak_pnl_R=0.8)
    ev = evaluate_exit(pos, "20240301", today_close_raw=10.5, regime="neutral")
    assert ev.triggered is True
    assert ev.signal_type == "time_exit"


def test_time_exit_blocked_when_pnl_is_healthy() -> None:
    """Same long hold but pnl_R=2.5 stays through time_exit unmolested."""
    pos = _pos(entry_date="20230901", entry_price=10.0, stop_price=9.0,
               peak_pnl_R=3.0)
    ev = evaluate_exit(pos, "20240301", today_close_raw=12.5, regime="neutral")
    # pnl_R = 2.5 >= 1.0 → not time_exit; also trailing peak_price=13 trail=11.05 → 12.5 > 11.05
    assert ev.signal_type != "time_exit"


# ---------------------------------------------------------------------------
# No-exit (everything healthy)
# ---------------------------------------------------------------------------


def test_no_exit_for_healthy_position() -> None:
    pos = _pos(entry_date="20240101", entry_price=10.0, stop_price=9.0,
               state="holding", peak_pnl_R=1.5)
    ev = evaluate_exit(pos, "20240301", today_close_raw=11.0, regime="neutral")
    assert ev.triggered is False
    assert ev.signal_type == "no_exit"
    # Every rule should appear in `missed` for explain transparency.
    missed_text = " ".join(ev.missed)
    for rule in ("hard_stop", "regime!=risk", "trailing_stop", "time_exit"):
        assert rule in missed_text


# ---------------------------------------------------------------------------
# Top-level detect_exit_signals — DB + cache integration
# ---------------------------------------------------------------------------


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()
    from deeptrade.core.db import Database  # noqa: PLC0415
    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())

    class _Stub:
        def call(self, api_name: str, **kwargs):
            return pd.DataFrame()
    rt = CheckmateRuntime(db=db, config=None, tushare=_Stub())  # type: ignore[arg-type]
    yield rt
    db.close()


def _seed_position(db, **overrides) -> None:
    base = dict(
        ts_code="600000.SH", entry_date="20240101",
        entry_price_raw=10.0, entry_price_qfq=10.0,
        shares=100, stop_price=9.0, state="holding",
        risk_R=1.0, peak_pnl_R=0.0, run_id=None,
    )
    base.update(overrides)
    db.execute(
        """
        INSERT INTO checkmate_positions
            (ts_code, entry_date, entry_price_raw, entry_price_qfq, shares,
             stop_price, state, risk_R, peak_pnl_R, run_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            base["ts_code"], base["entry_date"],
            base["entry_price_raw"], base["entry_price_qfq"],
            base["shares"], base["stop_price"], base["state"],
            base["risk_R"], base["peak_pnl_R"], base["run_id"],
        ],
    )


def _plant_today_daily(ts_code: str, *, trade_date: str, close: float, low: float | None = None) -> None:
    p = paths.daily_cache_dir() / f"{ts_code}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "ts_code": [ts_code], "trade_date": [trade_date],
        "open": [close], "high": [close * 1.01],
        "low": [low if low is not None else close * 0.99],
        "close": [close], "pre_close": [close * 0.99],
        "adj_factor": [1.0],
    })
    df.to_parquet(p, index=False)


def test_query_active_positions_excludes_closed(rt) -> None:
    _seed_position(rt.db, ts_code="A.SH", state="holding")
    _seed_position(rt.db, ts_code="B.SH", state="defensive", entry_date="20240102")
    _seed_position(rt.db, ts_code="C.SH", state="closed", entry_date="20240103")
    _seed_position(rt.db, ts_code="D.SH", state="pending", entry_date="20240104")
    out = query_active_positions(rt.db)
    assert {p.ts_code for p in out} == {"A.SH", "B.SH"}


def test_detect_exit_signals_emits_hard_stop_signal(rt) -> None:
    _seed_position(rt.db, ts_code="600000.SH", state="holding", stop_price=9.0,
                   entry_date="20240101")
    _plant_today_daily("600000.SH", trade_date="20240301", close=8.5)
    sigs = detect_exit_signals(rt, "20240301", regime="neutral")
    assert len(sigs) == 1
    sig = sigs[0]
    assert isinstance(sig, Signal)
    assert sig.signal_type == "hard_stop"
    assert sig.action == "exit"
    assert "close<stop" in " ".join(sig.explain["hit"])


def test_detect_exit_signals_emits_defensive_action(rt) -> None:
    _seed_position(rt.db, ts_code="600000.SH", state="holding",
                   peak_pnl_R=3.5, entry_date="20240101", stop_price=9.0)
    _plant_today_daily("600000.SH", trade_date="20240301", close=11.0)
    sigs = detect_exit_signals(rt, "20240301", regime="neutral")
    assert len(sigs) == 1
    assert sigs[0].action == "defensive"
    assert sigs[0].signal_type == "defensive_profit"


def test_detect_exit_signals_suppresses_t1_blocked(rt) -> None:
    """Entered today → no Signal even with a catastrophic close."""
    _seed_position(rt.db, ts_code="600000.SH", state="holding",
                   entry_date="20240301", stop_price=9.0)
    _plant_today_daily("600000.SH", trade_date="20240301", close=8.0)
    sigs = detect_exit_signals(rt, "20240301", regime="neutral")
    assert sigs == []


def test_detect_exit_signals_skips_when_no_active_positions(rt) -> None:
    sigs = detect_exit_signals(rt, "20240301", regime="neutral")
    assert sigs == []


def test_detect_exit_signals_skips_suspended_no_quote(rt) -> None:
    """Position exists but no daily row for trade_date → leave alone."""
    _seed_position(rt.db, ts_code="600000.SH", state="holding", entry_date="20240101")
    # Plant a row for a *different* date — the trade_date lookup fails.
    _plant_today_daily("600000.SH", trade_date="20240228", close=8.0)
    sigs = detect_exit_signals(rt, "20240301", regime="neutral")
    assert sigs == []
