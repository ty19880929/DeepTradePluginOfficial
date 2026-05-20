"""Executor tests — 8 spec scenarios (PR-4.1).

Each scenario constructs a tiny prices_raw + stk_limit DataFrame for one
trade_date, hands a single ``PendingOrder`` to :func:`simulate_session`, and
checks fills / cancels / deferred / risk_events. Costs are asserted on the
"normal buy fills" path so the breakdown math is locked in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from checkmate.config import ExecutionConfig
from checkmate.executor import (
    Cancel,
    Fill,
    PendingOrder,
    SessionReport,
    compute_costs,
    simulate_session,
)


def _prices(ts_code: str, *, open_: float, high: float, low: float, close: float,
            pre_close: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": ts_code, "trade_date": "20240301",
        "open": open_, "high": high, "low": low, "close": close,
        "pre_close": pre_close,
    }])


def _stk_limit(ts_code: str, *, up_limit: float, down_limit: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": ts_code, "trade_date": "20240301",
        "up_limit": up_limit, "down_limit": down_limit,
    }])


# ===========================================================================
# 1) Normal buy fills + cost decomposition
# ===========================================================================


def test_normal_buy_fills_at_open_with_slippage() -> None:
    order = PendingOrder(
        ts_code="600000.SH", side="buy", shares=1000,
        signal_date="20240229",
        stop_price=9.0, risk_R=1.0, reason_code="breakout",
    )
    prices = _prices("600000.SH", open_=10.0, high=10.5, low=9.9,
                     close=10.3, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)

    assert len(report.fills) == 1
    assert report.cancels == []
    assert report.deferred == []

    fill = report.fills[0]
    assert fill.ts_code == "600000.SH"
    assert fill.shares == 1000
    # Slippage 5bps = 0.05% on a 10.0 open → fill = 10.005
    assert fill.fill_price_raw == pytest.approx(10.005, abs=1e-4)
    assert fill.fill_date == "20240301"
    assert fill.order_date == "20240229"
    # Cost breakdown contains all 5 components (PR-7.3 adds ``impact``).
    cb = fill.cost_breakdown
    assert set(cb) == {"commission", "stamp_tax", "transfer_fee", "slippage", "impact"}
    notional = 1000 * 10.005
    assert cb["commission"] == pytest.approx(max(notional * 0.0003, 5.0), abs=1e-4)
    assert cb["stamp_tax"] == 0.0  # buys are stamp-tax-free
    assert cb["transfer_fee"] == pytest.approx(notional * 0.00002, abs=1e-4)
    assert cb["slippage"] == pytest.approx(notional * 5 / 10000, abs=1e-4)
    # Default impact_model="none" → impact = 0 (back-compat).
    assert cb["impact"] == 0.0


# ===========================================================================
# 2) Limit-up at open → cancel
# ===========================================================================


def test_limit_up_open_cancels_buy() -> None:
    """Open == up_limit → buy can't fill — cancel with reason."""
    order = PendingOrder(
        ts_code="600000.SH", side="buy", shares=1000,
        signal_date="20240229", reason_code="breakout",
    )
    # Open at 11.0 == up_limit 11.0
    prices = _prices("600000.SH", open_=11.0, high=11.0, low=11.0,
                     close=11.0, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    assert report.fills == []
    assert len(report.cancels) == 1
    assert report.cancels[0].cancel_reason == "limit_up_open"


# ===========================================================================
# 3) Limit-down on sell → defer
# ===========================================================================


def test_limit_down_open_defers_sell() -> None:
    order = PendingOrder(
        ts_code="600000.SH", side="sell", shares=1000,
        signal_date="20240229", reason_code="hard_stop",
    )
    prices = _prices("600000.SH", open_=9.0, high=9.0, low=9.0,
                     close=9.0, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    assert report.fills == []
    assert report.cancels == []
    assert len(report.deferred) == 1
    assert report.deferred[0].defer_count == 1
    assert report.deferred[0].ts_code == "600000.SH"


# ===========================================================================
# 4) T+1 — same-day entry sell rejected, risk event recorded
# ===========================================================================


def test_t1_blocks_same_day_sell() -> None:
    """A position opened today can't be sold today — record risk event."""
    order = PendingOrder(
        ts_code="600000.SH", side="sell", shares=1000,
        signal_date="20240301",
        reason_code="hard_stop", same_day_entry=True,
    )
    prices = _prices("600000.SH", open_=8.0, high=8.2, low=7.9,
                     close=7.9, pre_close=10.0)  # crash day
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    assert report.fills == []
    assert len(report.cancels) == 1
    assert report.cancels[0].cancel_reason == "t1_blocked"
    assert len(report.risk_events) == 1
    assert report.risk_events[0].event_type == "t1_sell_blocked"
    assert report.risk_events[0].payload["shares"] == 1000


# ===========================================================================
# 5) Gap-up too large → cancel buy
# ===========================================================================


def test_gap_up_too_large_cancels_buy() -> None:
    """Open / pre_close - 1 > max_gap_up_pct → cancel (don't chase a runaway)."""
    order = PendingOrder(
        ts_code="600000.SH", side="buy", shares=1000,
        signal_date="20240229", reason_code="breakout",
    )
    # 10.0 → 10.8 = +8% gap, exceeds default 5% cap
    prices = _prices("600000.SH", open_=10.8, high=11.0, low=10.7,
                     close=10.9, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    assert report.fills == []
    assert len(report.cancels) == 1
    assert "gap_up_too_large" in report.cancels[0].cancel_reason


def test_gap_up_below_threshold_fills_normally() -> None:
    """+4% gap < 5% cap → fill (boundary check on the gap rule)."""
    order = PendingOrder(
        ts_code="600000.SH", side="buy", shares=100,
        signal_date="20240229", reason_code="breakout",
    )
    prices = _prices("600000.SH", open_=10.4, high=10.6, low=10.3,
                     close=10.5, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    assert len(report.fills) == 1


# ===========================================================================
# 6) Partial fill — v0.1 doesn't support; record as TODO
# ===========================================================================


def test_partial_fill_unsupported_in_v01() -> None:
    """v0.1 fills the full order or cancels — there's no halfway state.

    This is a documentation test that locks the behaviour: a 1M-share order
    on a tiny stock still tries to fill at one open price, no slippage model
    adjustment. The "real" partial-fill model lands in v0.4 / Iter-7.
    """
    order = PendingOrder(
        ts_code="600000.SH", side="buy", shares=10_000_000,  # unrealistic size
        signal_date="20240229", reason_code="breakout",
    )
    prices = _prices("600000.SH", open_=10.0, high=10.1, low=9.9,
                     close=10.0, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    # All-or-nothing: it fills the full size.
    assert len(report.fills) == 1
    assert report.fills[0].shares == 10_000_000
    # TODO(v0.4): replace with a partial-fill assertion once Iter-7 lands.


# ===========================================================================
# 7) Ex-dividend day — fill uses raw post-ex-div open
# ===========================================================================


def test_ex_dividend_day_uses_raw_post_div_price() -> None:
    """Stock drops 0.5 yuan on ex-div day. Sell at the post-div open works
    fine — executor is exchange-truth, so it uses raw prices verbatim."""
    order = PendingOrder(
        ts_code="600000.SH", side="sell", shares=500,
        signal_date="20240229", reason_code="trailing_stop",
    )
    # Pre-div close 10.0, post-div pre_close adjusted to 9.5 by exchange.
    # The exchange recomputes up/down limits off the adjusted pre_close.
    prices = _prices("600000.SH", open_=9.55, high=9.6, low=9.5,
                     close=9.55, pre_close=9.5)
    limits = _stk_limit("600000.SH", up_limit=10.45, down_limit=8.55)
    report = simulate_session([order], "20240301", prices, limits)
    assert len(report.fills) == 1
    fill = report.fills[0]
    # 9.55 with -5bps slippage on a sell = 9.55 * (1 - 0.0005) ≈ 9.5452
    assert fill.fill_price_raw == pytest.approx(9.5452, abs=1e-3)


# ===========================================================================
# 8) Multi-day limit-down queue
# ===========================================================================


def test_multi_day_limit_down_eventually_cancels() -> None:
    """A sell deferred ``cfg.max_defer_days`` times in a row gets cancelled."""
    cfg = ExecutionConfig(max_defer_days=3)
    # First attempt
    order = PendingOrder(
        ts_code="600000.SH", side="sell", shares=500,
        signal_date="20240229", reason_code="hard_stop",
    )
    prices = _prices("600000.SH", open_=9.0, high=9.0, low=9.0,
                     close=9.0, pre_close=10.0)
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)

    current = order
    for day_idx in range(1, 4):  # 3 sessions of limit-down → deferred each
        report = simulate_session([current], f"2024030{day_idx}", prices, limits, cfg=cfg)
        assert report.fills == []
        assert report.cancels == []
        assert len(report.deferred) == 1
        assert report.deferred[0].defer_count == day_idx
        current = report.deferred[0]

    # 4th session still locked → defer_count would become 4 > max 3 → cancel
    report = simulate_session([current], "20240304", prices, limits, cfg=cfg)
    assert report.deferred == []
    assert len(report.cancels) == 1
    assert "limit_down_wedged" in report.cancels[0].cancel_reason
    assert report.cancels[0].defer_count == 4


# ===========================================================================
# Edge: suspended stock (no quote)
# ===========================================================================


def test_suspended_no_quote_cancels() -> None:
    """No row for ts_code in prices_raw → cancel as suspended."""
    order = PendingOrder(
        ts_code="600000.SH", side="buy", shares=1000,
        signal_date="20240229", reason_code="breakout",
    )
    prices = pd.DataFrame(columns=["ts_code", "trade_date", "open", "high",
                                    "low", "close", "pre_close"])
    limits = _stk_limit("600000.SH", up_limit=11.0, down_limit=9.0)
    report = simulate_session([order], "20240301", prices, limits)
    assert report.fills == []
    assert len(report.cancels) == 1
    assert report.cancels[0].cancel_reason == "suspended_no_quote"


# ===========================================================================
# Cost helper sanity
# ===========================================================================


def test_compute_costs_commission_floor() -> None:
    """Tiny notional → commission falls back to the 5 yuan floor."""
    cfg = ExecutionConfig()
    cb = compute_costs("buy", shares=100, fill_price=1.0, cfg=cfg)
    # notional=100, commission = max(100*0.0003=0.03, 5.0) = 5.0
    assert cb["commission"] == pytest.approx(5.0)


def test_compute_costs_sell_includes_stamp_tax() -> None:
    cfg = ExecutionConfig()
    cb_buy = compute_costs("buy", shares=1000, fill_price=10.0, cfg=cfg)
    cb_sell = compute_costs("sell", shares=1000, fill_price=10.0, cfg=cfg)
    assert cb_buy["stamp_tax"] == 0.0
    assert cb_sell["stamp_tax"] == pytest.approx(1000 * 10.0 * 0.001)
