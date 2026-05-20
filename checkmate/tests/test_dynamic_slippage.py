"""Dynamic-slippage tests (PR-7.1).

Three layers:
1. Curve math — `dynamic_slippage_bps` returns expected values at and
   between breakpoints, with proper clamping outside the curve.
2. Cost / fill price — `compute_costs` and `_slip_price` route through
   `_effective_slippage_bps` so dynamic mode picks the right bps per order.
3. Back-compat — fixed mode (the v0.1 default) keeps the old behaviour;
   PendingOrder.amount_20d_avg defaulting to None means existing tests
   exercise the legacy path.
"""

from __future__ import annotations

import pandas as pd
import pytest

from checkmate.config import ExecutionConfig
from checkmate.executor import (
    PendingOrder,
    _slip_price,
    compute_costs,
    dynamic_slippage_bps,
    simulate_session,
)


# ---------------------------------------------------------------------------
# Curve math
# ---------------------------------------------------------------------------


@pytest.fixture
def dynamic_cfg() -> ExecutionConfig:
    return ExecutionConfig(slippage_model="dynamic")


class TestCurve:
    def test_micro_cap_clamps_at_floor(self, dynamic_cfg) -> None:
        # 100万/day — below 1e6 breakpoint → clamp at 30 bps
        assert dynamic_slippage_bps(1e5, dynamic_cfg) == pytest.approx(30.0)

    def test_breakpoint_1e6_is_30bps(self, dynamic_cfg) -> None:
        assert dynamic_slippage_bps(1e6, dynamic_cfg) == pytest.approx(30.0)

    def test_breakpoint_1e7_is_20bps(self, dynamic_cfg) -> None:
        assert dynamic_slippage_bps(1e7, dynamic_cfg) == pytest.approx(20.0)

    def test_breakpoint_1e8_is_10bps(self, dynamic_cfg) -> None:
        assert dynamic_slippage_bps(1e8, dynamic_cfg) == pytest.approx(10.0)

    def test_breakpoint_1e9_is_5bps(self, dynamic_cfg) -> None:
        assert dynamic_slippage_bps(1e9, dynamic_cfg) == pytest.approx(5.0)

    def test_breakpoint_1e10_is_2bps(self, dynamic_cfg) -> None:
        assert dynamic_slippage_bps(1e10, dynamic_cfg) == pytest.approx(2.0)

    def test_mega_cap_clamps_at_ceiling(self, dynamic_cfg) -> None:
        # 1万亿/day — above 1e10 → clamp at 2 bps (the lowest)
        assert dynamic_slippage_bps(1e12, dynamic_cfg) == pytest.approx(2.0)

    def test_interpolates_between_breakpoints(self, dynamic_cfg) -> None:
        """3.16e8 ≈ log10 8.5 → halfway between (8, 10bps) and (9, 5bps) = 7.5 bps."""
        v = dynamic_slippage_bps(10 ** 8.5, dynamic_cfg)
        assert v == pytest.approx(7.5, rel=1e-3)

    def test_monotonic_decrease(self, dynamic_cfg) -> None:
        """Bigger liquidity → lower slippage all the way up."""
        vals = [dynamic_slippage_bps(10 ** k, dynamic_cfg) for k in range(5, 12)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1], f"non-monotonic at k={i+5}: {vals}"

    def test_custom_curve(self) -> None:
        cfg = ExecutionConfig(
            slippage_model="dynamic",
            slippage_bps_curve=((6.0, 50.0), (9.0, 5.0)),
        )
        # Halfway in log-space: log10 7.5
        assert dynamic_slippage_bps(10 ** 7.5, cfg) == pytest.approx(27.5, rel=1e-3)


# ---------------------------------------------------------------------------
# compute_costs / _slip_price routing
# ---------------------------------------------------------------------------


class TestComputeCostsRouting:
    def test_dynamic_uses_amount_to_pick_bps(self) -> None:
        cfg = ExecutionConfig(slippage_model="dynamic")
        # 1e10 → 2 bps; notional 100 * 100 = 10_000 → slippage = 2 yuan
        cb_mega = compute_costs("buy", 100, 100.0, cfg, amount_20d_avg=1e10)
        assert cb_mega["slippage"] == pytest.approx(2.0)
        # 1e6 → 30 bps; same notional → slippage = 30 yuan
        cb_micro = compute_costs("buy", 100, 100.0, cfg, amount_20d_avg=1e6)
        assert cb_micro["slippage"] == pytest.approx(30.0)

    def test_dynamic_without_amount_falls_back_to_fixed(self) -> None:
        cfg = ExecutionConfig(slippage_model="dynamic", slippage_bps=5)
        cb = compute_costs("buy", 100, 100.0, cfg, amount_20d_avg=None)
        # 5 bps on 10_000 notional → 5 yuan
        assert cb["slippage"] == pytest.approx(5.0)

    def test_fixed_mode_ignores_amount(self) -> None:
        cfg = ExecutionConfig(slippage_model="fixed", slippage_bps=5)
        cb = compute_costs("buy", 100, 100.0, cfg, amount_20d_avg=1e6)
        # Despite "micro-cap" amount, fixed mode keeps 5 bps
        assert cb["slippage"] == pytest.approx(5.0)

    def test_slip_price_buys_drift_up_with_dynamic(self) -> None:
        cfg = ExecutionConfig(slippage_model="dynamic")
        # Micro-cap (1e6 → 30 bps): 10.0 * (1 + 0.003) = 10.030
        p = _slip_price("buy", 10.0, cfg, amount_20d_avg=1e6)
        assert p == pytest.approx(10.030, abs=1e-4)
        # Mega-cap (1e10 → 2 bps): 10.0 * (1 + 0.0002) = 10.002
        p2 = _slip_price("buy", 10.0, cfg, amount_20d_avg=1e10)
        assert p2 == pytest.approx(10.002, abs=1e-4)

    def test_slip_price_sells_drift_down(self) -> None:
        cfg = ExecutionConfig(slippage_model="dynamic")
        p = _slip_price("sell", 10.0, cfg, amount_20d_avg=1e6)
        # 10.0 * (1 - 0.003) = 9.970
        assert p == pytest.approx(9.970, abs=1e-4)


# ---------------------------------------------------------------------------
# simulate_session end-to-end with dynamic mode
# ---------------------------------------------------------------------------


def _prices(ts_code: str, *, open_: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": ts_code, "trade_date": "20240301",
        "open": open_, "high": open_ * 1.02, "low": open_ * 0.98,
        "close": open_, "pre_close": open_ * 0.99,
    }])


def _stk_limit(ts_code: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "ts_code": ts_code, "trade_date": "20240301",
        "up_limit": 100.0, "down_limit": 1.0,
    }])


def test_simulate_session_uses_dynamic_slippage_from_pending_amount() -> None:
    """Two orders for the same symbol but different amount → different fill prices."""
    cfg = ExecutionConfig(slippage_model="dynamic")
    micro = PendingOrder(
        ts_code="A.SH", side="buy", shares=100,
        signal_date="20240229", reason_code="breakout",
        amount_20d_avg=1e6,  # micro-cap → 30 bps
    )
    mega = PendingOrder(
        ts_code="B.SH", side="buy", shares=100,
        signal_date="20240229", reason_code="breakout",
        amount_20d_avg=1e10,  # mega-cap → 2 bps
    )
    prices = pd.concat([_prices("A.SH"), _prices("B.SH")], ignore_index=True)
    limits = pd.concat([_stk_limit("A.SH"), _stk_limit("B.SH")], ignore_index=True)
    report = simulate_session([micro, mega], "20240301", prices, limits, cfg)
    assert len(report.fills) == 2
    by_code = {f.ts_code: f for f in report.fills}
    # Micro-cap pays more bps → higher fill price; mega-cap pays less.
    assert by_code["A.SH"].fill_price_raw > by_code["B.SH"].fill_price_raw
    # Specific values: 10 * 1.003 vs 10 * 1.0002
    assert by_code["A.SH"].fill_price_raw == pytest.approx(10.030, abs=1e-4)
    assert by_code["B.SH"].fill_price_raw == pytest.approx(10.002, abs=1e-4)
    # And cost_breakdown.slippage differs accordingly
    assert by_code["A.SH"].cost_breakdown["slippage"] > by_code["B.SH"].cost_breakdown["slippage"]


def test_simulate_session_fixed_mode_back_compat() -> None:
    """Default (slippage_model='fixed') ignores amount_20d_avg — old tests pass unchanged."""
    cfg = ExecutionConfig()  # fixed mode, 5 bps
    o = PendingOrder(
        ts_code="A.SH", side="buy", shares=100,
        signal_date="20240229", reason_code="breakout",
        amount_20d_avg=1e6,  # would be 30 bps if dynamic — but cfg is fixed
    )
    report = simulate_session(
        [o], "20240301", _prices("A.SH"), _stk_limit("A.SH"), cfg,
    )
    assert len(report.fills) == 1
    # 10 * (1 + 5/10000) = 10.005
    assert report.fills[0].fill_price_raw == pytest.approx(10.005, abs=1e-4)


def test_pending_order_amount_defaults_to_none() -> None:
    """Existing tests that don't set amount_20d_avg keep working."""
    o = PendingOrder(
        ts_code="A.SH", side="buy", shares=100,
        signal_date="20240229", reason_code="breakout",
    )
    assert o.amount_20d_avg is None
