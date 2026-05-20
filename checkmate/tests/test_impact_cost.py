"""Market-impact cost tests (PR-7.3).

Layered coverage:
1. ``impact_bps`` — square-root model math: zero by default, scales with
   sqrt(participation), zero below ``impact_min_participation``.
2. ``compute_costs`` — ``impact`` is always present in the dict (5 keys);
   value is 0 in the default ``"none"`` mode; non-zero under ``"sqrt"``.
3. ``_slip_price`` — fills shift further when impact is enabled.
4. ``simulate_session`` — end-to-end through the executor: big participation
   sees its fill price move past a tiny participation order's.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from checkmate.config import ExecutionConfig
from checkmate.executor import (
    PendingOrder,
    _slip_price,
    compute_costs,
    impact_bps,
    simulate_session,
)


# ---------------------------------------------------------------------------
# impact_bps — square-root model
# ---------------------------------------------------------------------------


@pytest.fixture
def sqrt_cfg() -> ExecutionConfig:
    """sqrt model with the default coefficient + min participation."""
    return ExecutionConfig(impact_model="sqrt")


class TestImpactBpsSqrt:
    def test_none_model_returns_zero(self) -> None:
        cfg = ExecutionConfig()  # default impact_model="none"
        assert impact_bps(1e8, 1e8, cfg) == 0.0

    def test_missing_amount_returns_zero(self, sqrt_cfg) -> None:
        assert impact_bps(1e6, None, sqrt_cfg) == 0.0
        assert impact_bps(1e6, 0.0, sqrt_cfg) == 0.0

    def test_below_min_participation_returns_zero(self, sqrt_cfg) -> None:
        # 0.2% participation < 0.5% default min
        assert impact_bps(2_000.0, 1_000_000.0, sqrt_cfg) == 0.0

    def test_one_percent_participation_yields_10bps(self, sqrt_cfg) -> None:
        # participation = 1% → sqrt(0.01) * 100 = 10 bps
        bps = impact_bps(1_000_000.0, 100_000_000.0, sqrt_cfg)
        assert bps == pytest.approx(10.0, rel=1e-9)

    def test_25_percent_participation_yields_50bps(self, sqrt_cfg) -> None:
        # participation = 25% → sqrt(0.25) * 100 = 50 bps
        bps = impact_bps(2.5e7, 1e8, sqrt_cfg)
        assert bps == pytest.approx(50.0, rel=1e-9)

    def test_100_percent_participation_yields_100bps(self, sqrt_cfg) -> None:
        # participation = 1.0 → sqrt(1.0) * 100 = 100 bps
        bps = impact_bps(1e8, 1e8, sqrt_cfg)
        assert bps == pytest.approx(100.0, rel=1e-9)

    def test_coefficient_scales_linearly(self) -> None:
        cfg = ExecutionConfig(impact_model="sqrt", impact_coefficient=2.0)
        # Same 1% participation but 2× coefficient → 20 bps
        assert impact_bps(1_000_000.0, 1e8, cfg) == pytest.approx(20.0)

    def test_monotonic_in_participation(self, sqrt_cfg) -> None:
        amts = (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
        values = [impact_bps(p * 1e9, 1e9, sqrt_cfg) for p in amts]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], f"non-monotonic at i={i}: {values}"


# ---------------------------------------------------------------------------
# compute_costs — impact key always present
# ---------------------------------------------------------------------------


class TestComputeCostsImpactKey:
    def test_cost_dict_has_five_keys(self) -> None:
        cb = compute_costs("buy", 100, 10.0, ExecutionConfig())
        assert set(cb) == {"commission", "stamp_tax", "transfer_fee", "slippage", "impact"}

    def test_default_none_mode_impact_is_zero(self) -> None:
        cb = compute_costs("buy", 100, 10.0, ExecutionConfig(),
                            amount_20d_avg=1e6)
        assert cb["impact"] == 0.0

    def test_sqrt_mode_impact_nonzero(self) -> None:
        cfg = ExecutionConfig(impact_model="sqrt")
        # 1000 shares × 100 yuan = 100_000 notional; daily 1e6 → 10% participation
        # → bps = sqrt(0.10) * 100 ≈ 31.62 → impact = 100_000 * 31.62/10_000 = 316.22
        cb = compute_costs("buy", 1000, 100.0, cfg, amount_20d_avg=1e6)
        assert cb["impact"] == pytest.approx(316.2278, abs=1e-3)

    def test_sqrt_mode_below_min_participation_impact_zero(self) -> None:
        cfg = ExecutionConfig(impact_model="sqrt")
        # 100 shares × 10 yuan = 1_000 notional; daily 1e9 → 0.0001% participation
        cb = compute_costs("buy", 100, 10.0, cfg, amount_20d_avg=1e9)
        assert cb["impact"] == 0.0


# ---------------------------------------------------------------------------
# _slip_price — impact stacks with slippage
# ---------------------------------------------------------------------------


class TestSlipPriceWithImpact:
    def test_buy_with_impact_drifts_higher_than_pure_slippage(self) -> None:
        """Slippage-only vs slippage+impact: with the same fixed slippage_bps,
        adding impact pushes the buy fill price further up."""
        cfg_no_imp = ExecutionConfig()  # impact_model="none"
        cfg_imp = ExecutionConfig(impact_model="sqrt")
        # 1% participation → 10 bps impact
        kwargs = {"amount_20d_avg": 1e8, "order_value": 1e6}
        p_no = _slip_price("buy", 100.0, cfg_no_imp, **kwargs)
        p_imp = _slip_price("buy", 100.0, cfg_imp, **kwargs)
        # 5 bps slippage vs (5 + 10) bps stack
        assert p_no == pytest.approx(100.0 * 1.0005, abs=1e-6)
        assert p_imp == pytest.approx(100.0 * 1.0015, abs=1e-6)

    def test_sell_with_impact_drifts_lower(self) -> None:
        cfg = ExecutionConfig(impact_model="sqrt")
        p = _slip_price("sell", 100.0, cfg,
                         amount_20d_avg=1e8, order_value=1e6)
        # Same magnitudes but sells lose: 100 * (1 - 0.0015) = 99.85
        assert p == pytest.approx(99.85, abs=1e-6)

    def test_no_order_value_zeroes_impact(self) -> None:
        cfg = ExecutionConfig(impact_model="sqrt")
        # order_value=None or 0 → impact zero, only slippage applies
        p = _slip_price("buy", 100.0, cfg, amount_20d_avg=1e8)
        # default slippage_bps=5 → 100 * 1.0005
        assert p == pytest.approx(100.05, abs=1e-6)


# ---------------------------------------------------------------------------
# simulate_session — end-to-end behaviour
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


def test_simulate_session_big_order_pays_more_impact() -> None:
    """Same symbol, two order sizes → bigger order has higher fill price
    (more impact) and a non-zero cost_breakdown.impact field."""
    cfg = ExecutionConfig(impact_model="sqrt")
    small = PendingOrder(
        ts_code="A.SH", side="buy", shares=100,  # 100 shares × 10 = 1_000 notional
        signal_date="20240229", reason_code="x",
        amount_20d_avg=1_000_000.0,  # 0.1% participation → below 0.5% min, impact=0
    )
    big = PendingOrder(
        ts_code="B.SH", side="buy", shares=20_000,  # 20_000 × 10 = 200_000 notional
        signal_date="20240229", reason_code="x",
        amount_20d_avg=1_000_000.0,  # 20% participation → sqrt(0.2)*100 ≈ 44.7 bps
    )
    prices = pd.concat([_prices("A.SH"), _prices("B.SH")], ignore_index=True)
    limits = pd.concat([_stk_limit("A.SH"), _stk_limit("B.SH")], ignore_index=True)
    report = simulate_session([small, big], "20240301", prices, limits, cfg)
    assert len(report.fills) == 2

    by_code = {f.ts_code: f for f in report.fills}
    small_fill = by_code["A.SH"]
    big_fill = by_code["B.SH"]

    # Big order's fill price > small order's fill price.
    assert big_fill.fill_price_raw > small_fill.fill_price_raw
    # cost_breakdown.impact: small=0, big>0
    assert small_fill.cost_breakdown["impact"] == 0.0
    assert big_fill.cost_breakdown["impact"] > 0.0
    # Specific big impact: notional ≈ 200_000 × 1.00497 (fill price), bps ≈ 44.7,
    # so impact_dollar ≈ 200_000 × 44.7/10_000 ≈ 894 yuan.
    assert big_fill.cost_breakdown["impact"] == pytest.approx(894.0, rel=0.05)


def test_simulate_session_default_mode_keeps_impact_zero() -> None:
    """Back-compat: without impact_model='sqrt', the cost_breakdown has the
    new ``impact`` key but always at 0.0."""
    cfg = ExecutionConfig()  # impact_model="none" by default
    o = PendingOrder(
        ts_code="A.SH", side="buy", shares=1000,
        signal_date="20240229", reason_code="x",
        amount_20d_avg=100.0,  # extreme participation, but no impact in "none" mode
    )
    report = simulate_session(
        [o], "20240301", _prices("A.SH"), _stk_limit("A.SH"), cfg,
    )
    assert len(report.fills) == 1
    cb = report.fills[0].cost_breakdown
    assert "impact" in cb
    assert cb["impact"] == 0.0
