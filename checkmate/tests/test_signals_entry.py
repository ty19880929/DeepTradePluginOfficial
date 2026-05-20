"""Entry-signal tests — happy + edge per rule (PR-3.1).

Each test crafts a tiny qfq / daily_basic frame that isolates one rule,
runs the matching evaluator, and inspects ``EntryEval``. The composite
:func:`detect_entry_signals_for_symbol` is exercised separately to verify
Signal emission and explain JSON shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from checkmate.config import EntryConfig
from checkmate.features import FeaturesRow
from checkmate.signals import (
    Signal,
    _board_pct_cap,
    detect_entry_signals_for_symbol,
    evaluate_breakout,
    evaluate_continuation,
    evaluate_pullback,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_qfq(
    *,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    pre_close: list[float] | None = None,
    ts_code: str = "600000.SH",
    end_date: str = "20240329",
) -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(end=end_date, periods=n).strftime("%Y%m%d").tolist()
    highs_ = highs if highs is not None else [c * 1.01 for c in closes]
    lows_ = lows if lows is not None else [c * 0.99 for c in closes]
    pre_close_ = pre_close if pre_close is not None else [closes[0] * 0.99] + closes[:-1]
    df = pd.DataFrame({
        "ts_code": [ts_code] * n,
        "trade_date": dates,
        "open": closes,
        "high": highs_,
        "low": lows_,
        "close": closes,
        "pre_close": pre_close_,
        "adj_factor": [1.0] * n,
    })
    for col in ("open", "high", "low", "close", "pre_close"):
        df[f"{col}_qfq"] = df[col]
    return df


def _make_daily_basic(
    *,
    amounts: list[float],
    ts_code: str = "600000.SH",
    end_date: str = "20240329",
) -> pd.DataFrame:
    n = len(amounts)
    dates = pd.bdate_range(end=end_date, periods=n).strftime("%Y%m%d").tolist()
    return pd.DataFrame({
        "ts_code": [ts_code] * n,
        "trade_date": dates,
        "amount": amounts,
        "turnover_rate": [1.5] * n,
        "total_mv": [1e10] * n,
    })


# ===========================================================================
# Board cap detection
# ===========================================================================


def test_board_pct_cap_routing() -> None:
    cfg = EntryConfig()
    assert _board_pct_cap("600519.SH", cfg) == (cfg.pct_chg_cap_main_board, "main_board")
    assert _board_pct_cap("000001.SZ", cfg) == (cfg.pct_chg_cap_main_board, "main_board")
    assert _board_pct_cap("300750.SZ", cfg) == (cfg.pct_chg_cap_chinext, "chinext")
    assert _board_pct_cap("688981.SH", cfg) == (cfg.pct_chg_cap_star, "star")


# ===========================================================================
# Breakout — happy + 2 edges (no volume; limit-up)
# ===========================================================================


def _breakout_setup_qfq() -> pd.DataFrame:
    """50-session frame: prior 40d high = 10.20, today 10.50, raw pct_chg ≈ 5%."""
    closes = [10.0 + i * 0.005 for i in range(49)] + [10.50]
    highs = [c + 0.05 for c in closes[:-1]] + [10.55]
    pre = [10.0] + closes[:-1]
    pre[-1] = 10.00  # today's pre_close → pct_chg = 5%
    return _make_qfq(closes=closes, highs=highs, pre_close=pre)


def test_breakout_happy() -> None:
    qfq = _breakout_setup_qfq()
    # 21-day amount window: prior 20d avg = 1.0e8; today = 1.5e8 → ratio 1.5
    amounts = [1.0e8] * 20 + [1.5e8]
    db = _make_daily_basic(amounts=amounts)
    ev = evaluate_breakout("600000.SH", qfq, db, EntryConfig(), trade_date="20240329")
    assert ev.triggered is True
    assert ev.missed == []
    assert any("close>" in h for h in ev.hit)
    assert any("amount_ratio" in h for h in ev.hit)
    assert any("pct_chg<=" in h for h in ev.hit)
    assert ev.details["amount_ratio"] == pytest.approx(1.5, abs=1e-4)


def test_breakout_edge_just_below_amount_threshold() -> None:
    """Volume just shy of 1.2× → not triggered, missed list points at it."""
    qfq = _breakout_setup_qfq()
    amounts = [1.0e8] * 20 + [1.18e8]  # ratio = 1.18 < 1.2
    db = _make_daily_basic(amounts=amounts)
    ev = evaluate_breakout("600000.SH", qfq, db, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert any("amount_ratio" in m for m in ev.missed)


def test_breakout_edge_pct_chg_limit_up_main_board() -> None:
    """A main-board limit-up day (≈10%) blows past the 8% cap → blocked."""
    # Today's pre_close 10.0, close 11.0 → pct_chg = 10%
    closes = [10.0 + i * 0.005 for i in range(49)] + [11.0]
    highs = [c + 0.05 for c in closes[:-1]] + [11.0]  # one-way style; high == close
    pre = [10.0] + closes[:-1]
    pre[-1] = 10.0
    qfq = _make_qfq(closes=closes, highs=highs, pre_close=pre)
    amounts = [1.0e8] * 20 + [1.5e8]
    db = _make_daily_basic(amounts=amounts)
    ev = evaluate_breakout("600000.SH", qfq, db, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert any("pct_chg<=" in m for m in ev.missed), ev.missed


def test_breakout_chinext_allows_higher_pct() -> None:
    """A ChiNext 300xxx stock allows up to 11%."""
    closes = [10.0 + i * 0.005 for i in range(49)] + [11.0]  # 10% advance
    highs = [c + 0.05 for c in closes[:-1]] + [11.0]
    pre = [10.0] + closes[:-1]
    pre[-1] = 10.0
    qfq = _make_qfq(closes=closes, highs=highs, pre_close=pre, ts_code="300999.SZ")
    amounts = [1.0e8] * 20 + [1.5e8]
    db = _make_daily_basic(amounts=amounts, ts_code="300999.SZ")
    ev = evaluate_breakout("300999.SZ", qfq, db, EntryConfig(), trade_date="20240329")
    # ChiNext cap 11% — 10% pct_chg passes.
    assert ev.triggered is True
    assert ev.details["board"] == "chinext"


def test_breakout_close_not_above_lookback_high() -> None:
    """Same close, but prior 40d had a 12.0 spike → not breaking out."""
    closes = [10.0 + i * 0.005 for i in range(49)] + [10.50]
    highs = [c + 0.05 for c in closes[:-1]] + [10.55]
    highs[20] = 12.0  # past spike beats today's close
    pre = [10.0] + closes[:-1]
    pre[-1] = 10.0
    qfq = _make_qfq(closes=closes, highs=highs, pre_close=pre)
    db = _make_daily_basic(amounts=[1.0e8] * 20 + [1.5e8])
    ev = evaluate_breakout("600000.SH", qfq, db, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert any("close>" in m and "high=12" in m for m in ev.missed)


# ===========================================================================
# Pullback
# ===========================================================================


def test_pullback_happy() -> None:
    """80-session series: rising trend, one mild dip to MA20, then re-clears
    the 5d short-term high today."""
    # Build an uptrend with a dip at idx ~70
    base = np.linspace(10.0, 15.0, 80)
    dips = base.copy()
    dips[68:72] = dips[68:72] - 0.7  # dip below the smooth trend
    # Today (idx=-1) clears the 5d platform high
    dips[-1] = dips[-6:-1].max() + 0.20
    qfq = _make_qfq(
        closes=list(dips),
        lows=list(dips * 0.985),
        highs=list(dips * 1.015),
    )
    ev = evaluate_pullback("600000.SH", qfq, EntryConfig(), trade_date="20240329")
    assert ev.triggered, f"missed={ev.missed}"
    assert all(s in ev.hit for s in (
        "close>ma20", "ma20>ma60",
    ))


def test_pullback_edge_downtrend_not_triggered() -> None:
    """Downtrend: ma20 < ma60 → fail trend gate."""
    closes = list(np.linspace(15.0, 10.0, 80))
    qfq = _make_qfq(closes=closes)
    ev = evaluate_pullback("600000.SH", qfq, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert "ma20>ma60" in ev.missed


def test_pullback_edge_never_touched_ma20() -> None:
    """Strong uptrend with a recent gap-up — the last 10 lows sit comfortably
    above MA20 * 1.03 so the ±3% touch test fails."""
    # 60 days at 10 → ma20 / ma60 anchored low; 10 days at 14; final 10 days
    # jump to 17. ma20 = (10*14 + 10*17)/20 = 15.5; touch band = [15.04, 15.97].
    # Last 10 lows = 17 * 0.99 = 16.83 — well above the upper bound → no touch.
    closes = [10.0] * 60 + [14.0] * 10 + [17.0] * 10
    qfq = _make_qfq(closes=closes)
    ev = evaluate_pullback("600000.SH", qfq, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert any("touched_ma20" in m for m in ev.missed)


# ===========================================================================
# Continuation
# ===========================================================================


def _continuation_setup_qfq() -> pd.DataFrame:
    """20 sessions, today's close clears the prior 10-session high."""
    closes = [10.0] * 9 + [10.5] + [9.8] * 9 + [10.6]
    highs = [c * 1.005 for c in closes]
    return _make_qfq(closes=closes, highs=highs)


def test_continuation_happy() -> None:
    qfq = _continuation_setup_qfq()
    row = FeaturesRow(trade_date="20240329", ts_code="600000.SH", rs60_pctile=0.85, score=70.0)
    ev = evaluate_continuation("600000.SH", qfq, row, EntryConfig(), trade_date="20240329")
    assert ev.triggered is True


def test_continuation_edge_rs60_just_below_threshold() -> None:
    qfq = _continuation_setup_qfq()
    row = FeaturesRow(trade_date="20240329", ts_code="600000.SH", rs60_pctile=0.79)
    ev = evaluate_continuation("600000.SH", qfq, row, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert any("rs60_pctile" in m for m in ev.missed)


def test_continuation_missing_rs60_pctile() -> None:
    """rs60_pctile None → cannot evaluate strength → not triggered."""
    qfq = _continuation_setup_qfq()
    row = FeaturesRow(trade_date="20240329", ts_code="600000.SH", rs60_pctile=None)
    ev = evaluate_continuation("600000.SH", qfq, row, EntryConfig(), trade_date="20240329")
    assert ev.triggered is False
    assert any("rs60_pctile" in m and "None" in m for m in ev.missed)


# ===========================================================================
# Composite per-symbol detection
# ===========================================================================


def test_detect_entry_signals_for_symbol_emits_breakout() -> None:
    qfq = _breakout_setup_qfq()
    db = _make_daily_basic(amounts=[1.0e8] * 20 + [1.5e8])
    row = FeaturesRow(
        trade_date="20240329", ts_code="600000.SH",
        score=68.0, rs60_pctile=0.50,
    )
    sigs = detect_entry_signals_for_symbol(row, qfq, db, EntryConfig())
    assert len(sigs) >= 1
    bo = [s for s in sigs if s.signal_type == "breakout"]
    assert len(bo) == 1
    sig = bo[0]
    assert isinstance(sig, Signal)
    assert sig.action == "enter"
    assert sig.score == 68.0
    # explain JSON shape
    assert sig.explain["signal_type"] == "breakout"
    assert "hit" in sig.explain and "missed" in sig.explain
    assert "details" in sig.explain
    assert sig.explain["details"]["board"] == "main_board"


def test_detect_entry_signals_emits_nothing_when_all_fail() -> None:
    """Flat, no volume bump, rs60_pctile None → all three rules fail."""
    closes = [10.0] * 80
    qfq = _make_qfq(closes=closes)
    db = _make_daily_basic(amounts=[1.0e8] * 80)
    row = FeaturesRow(trade_date="20240329", ts_code="600000.SH", rs60_pctile=None)
    sigs = detect_entry_signals_for_symbol(row, qfq, db, EntryConfig())
    assert sigs == []


def test_signal_explain_carries_all_three_keys() -> None:
    qfq = _continuation_setup_qfq()
    db = _make_daily_basic(amounts=[1.0e8] * 19 + [1.5e8])
    row = FeaturesRow(trade_date="20240329", ts_code="600000.SH",
                      rs60_pctile=0.90, score=85.0)
    sigs = detect_entry_signals_for_symbol(row, qfq, db, EntryConfig())
    cont = [s for s in sigs if s.signal_type == "continuation"]
    assert len(cont) == 1
    explain = cont[0].explain
    assert set(explain) == {"signal_type", "hit", "missed", "details"}
    assert explain["details"]["rs60_pctile"] == 0.90
