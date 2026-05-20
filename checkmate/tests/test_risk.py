"""Risk-module tests (PR-3.3).

Pure-function tests — no DB, no Tushare. Cover the three spec acceptance
points (board lot rounding / industry cap rejects the 4th / regime=weak
limits new opens) plus per-cap units.
"""

from __future__ import annotations

import pytest

from checkmate.config import RiskConfig
from checkmate.risk import (
    PositionView,
    ProposedEntry,
    SizedOrder,
    apply_portfolio_constraints,
    size_position,
)


# ===========================================================================
# size_position
# ===========================================================================


def test_size_position_rounds_down_to_lot() -> None:
    """risk_dollars=10_000, per_share_risk=1.0 → 10_000 raw shares → 10_000 (already a multiple of 100)."""
    n = size_position(entry_price=10.0, stop_price=9.0, portfolio_value=1_000_000.0)
    assert n == 10_000


def test_size_position_truncates_non_lot_remainder() -> None:
    """raw=15_050 → 15_000 (truncate to 100-share lot)."""
    cfg = RiskConfig(risk_per_trade=0.01)
    # 1% of 1.505e6 = 15_050; per_share risk = 1.0 → 15_050 raw → 15_000 lot
    n = size_position(entry_price=10.0, stop_price=9.0,
                      portfolio_value=1_505_000.0, cfg=cfg)
    assert n == 15_000


def test_size_position_zero_when_stop_above_entry() -> None:
    assert size_position(entry_price=10.0, stop_price=11.0,
                         portfolio_value=1e6) == 0
    assert size_position(entry_price=10.0, stop_price=10.0,
                         portfolio_value=1e6) == 0


def test_size_position_zero_for_empty_portfolio() -> None:
    assert size_position(entry_price=10.0, stop_price=9.0, portfolio_value=0.0) == 0
    assert size_position(entry_price=10.0, stop_price=9.0, portfolio_value=-1.0) == 0


def test_size_position_uses_risk_per_trade_setting() -> None:
    """0.5% per trade halves the share count vs the default 1%."""
    n_default = size_position(entry_price=10.0, stop_price=9.0,
                              portfolio_value=1_000_000.0,
                              cfg=RiskConfig(risk_per_trade=0.01))
    n_half = size_position(entry_price=10.0, stop_price=9.0,
                           portfolio_value=1_000_000.0,
                           cfg=RiskConfig(risk_per_trade=0.005))
    assert n_half == n_default // 2


def test_size_position_zero_shares_when_risk_budget_tiny() -> None:
    """A wide stop on a small portfolio yields fewer than 100 raw shares."""
    # Budget=100, per-share risk=10 → 10 raw shares < 100-lot → 0
    n = size_position(entry_price=20.0, stop_price=10.0, portfolio_value=10_000.0)
    assert n == 0


# ===========================================================================
# apply_portfolio_constraints — single position cap
# ===========================================================================


def test_single_weight_cap_rejects_oversize_order() -> None:
    """Single proposal that would consume >10% of the portfolio is dropped."""
    # entry=100, stop=99 (R=1). risk_budget=1% of 1M = 10_000 → 10_000 shares
    # → value = 1_000_000 → 100% of portfolio → fails single cap.
    props = [ProposedEntry(ts_code="A.SH", entry_price=100.0, stop_price=99.0,
                           industry="X", score=80.0)]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="neutral",
    )
    assert len(orders) == 1
    assert orders[0].accepted is False
    assert "single_weight_cap" in orders[0].cancel_reason


# ===========================================================================
# apply_portfolio_constraints — industry cap
# ===========================================================================


def test_industry_cap_drops_fourth_candidate() -> None:
    """Three same-industry proposals fit; the fourth pushes the bucket past 30%."""
    # Each order ≈ 10% (right at the single-weight cap). 3 × 10% = 30%
    # industry — exactly at the cap; a 4th would tip over.
    # Override both the daily cap AND the regime entry cap so the industry
    # constraint is the only thing limiting acceptance.
    cfg = RiskConfig(
        max_new_entries_per_day=10,
        regime_entry_caps={"strong": 10, "neutral": 10, "weak": 10, "risk": 0},
    )
    props = [
        ProposedEntry(ts_code=f"{i}.SH", entry_price=10.0, stop_price=9.0,
                      industry="电子", score=90.0 - i)
        for i in range(4)
    ]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="strong",
        cfg=cfg,
    )
    accepted = [o for o in orders if o.accepted]
    rejected = [o for o in orders if not o.accepted]
    assert len(accepted) == 3
    assert len(rejected) == 1
    assert "industry_cap" in rejected[0].cancel_reason
    # The lowest-score proposal is the one that got dropped.
    assert rejected[0].ts_code == "3.SH"


def test_industry_cap_respects_existing_positions() -> None:
    """20% of the industry already held → only ~10% headroom → only 1 of 2 fits."""
    existing = [
        PositionView(ts_code="OLD.SH", industry="电子", market_value=200_000.0),  # 20%
    ]
    props = [
        ProposedEntry(ts_code="A.SH", entry_price=10.0, stop_price=9.0,
                      industry="电子", score=90.0),
        ProposedEntry(ts_code="B.SH", entry_price=10.0, stop_price=9.0,
                      industry="电子", score=80.0),
    ]
    cfg = RiskConfig(
        max_new_entries_per_day=10,
        regime_entry_caps={"strong": 10, "neutral": 10, "weak": 10, "risk": 0},
    )
    orders = apply_portfolio_constraints(
        props, current_positions=existing,
        portfolio_value=1_000_000.0, regime="strong",
        cfg=cfg,
    )
    accepted = [o for o in orders if o.accepted]
    assert len(accepted) == 1
    assert accepted[0].ts_code == "A.SH"  # higher score wins


# ===========================================================================
# apply_portfolio_constraints — regime caps
# ===========================================================================


def test_regime_risk_blocks_all_new_entries() -> None:
    props = [
        ProposedEntry(ts_code=f"{i}.SH", entry_price=10.0, stop_price=9.0,
                      industry="X", score=90.0)
        for i in range(5)
    ]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="risk",
    )
    assert all(o.accepted is False for o in orders)
    assert all(o.cancel_reason == "regime_cap" for o in orders)


def test_regime_weak_caps_at_one_new_entry() -> None:
    props = [
        ProposedEntry(ts_code=f"{i}.SH", entry_price=10.0, stop_price=9.0,
                      industry=f"S{i}", score=90.0 - i)
        for i in range(3)
    ]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="weak",
    )
    accepted = [o for o in orders if o.accepted]
    assert len(accepted) == 1
    rejected = [o for o in orders if not o.accepted]
    assert len(rejected) == 2
    assert all(o.cancel_reason == "regime_cap" for o in rejected)


def test_regime_neutral_caps_at_two() -> None:
    props = [
        ProposedEntry(ts_code=f"{i}.SH", entry_price=10.0, stop_price=9.0,
                      industry=f"S{i}", score=90.0 - i)
        for i in range(4)
    ]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="neutral",
    )
    accepted = [o for o in orders if o.accepted]
    assert len(accepted) == 2


def test_regime_strong_uses_default_daily_cap() -> None:
    props = [
        ProposedEntry(ts_code=f"{i}.SH", entry_price=10.0, stop_price=9.0,
                      industry=f"S{i}", score=90.0 - i)
        for i in range(5)
    ]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="strong",
    )
    accepted = [o for o in orders if o.accepted]
    # strong cap = 3 = default max_new_entries_per_day
    assert len(accepted) == 3


# ===========================================================================
# apply_portfolio_constraints — proposal ordering
# ===========================================================================


def test_proposals_evaluated_in_score_desc_order() -> None:
    """Higher-score proposals consume the cap budget first."""
    props = [
        ProposedEntry(ts_code="LOW.SH",  entry_price=10.0, stop_price=9.0,
                      industry="X", score=10.0),
        ProposedEntry(ts_code="HIGH.SH", entry_price=10.0, stop_price=9.0,
                      industry="X", score=90.0),
    ]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="weak",   # cap = 1
    )
    accepted = [o for o in orders if o.accepted]
    assert len(accepted) == 1
    assert accepted[0].ts_code == "HIGH.SH"


# ===========================================================================
# Order shape sanity
# ===========================================================================


def test_sized_order_carries_explain_through() -> None:
    explain = {"hit": ["close>40d_high"], "missed": []}
    props = [ProposedEntry(ts_code="A.SH", entry_price=10.0, stop_price=9.0,
                           industry="X", score=80.0, signal_type="breakout",
                           explain=explain)]
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=1_000_000.0, regime="strong",
    )
    assert len(orders) == 1 and orders[0].accepted is True
    assert orders[0].signal_type == "breakout"
    assert orders[0].explain == explain
    # 1% of 1M / 1.0 R-per-share = 10_000 shares → weight 0.10 (right at cap)
    assert orders[0].shares == 10_000
    assert orders[0].weight == pytest.approx(0.10)


def test_zero_shares_proposal_gets_cancel_reason() -> None:
    """Stop tighter than risk budget allows yields 0 shares (post-lot floor)."""
    props = [ProposedEntry(ts_code="A.SH", entry_price=10.0, stop_price=0.0,
                           industry="X", score=70.0)]
    # Stop at 0 → per-share risk=10. 1% of 10_000 = 100. 100/10 = 10 raw < 100 lot.
    orders = apply_portfolio_constraints(
        props, current_positions=[],
        portfolio_value=10_000.0, regime="strong",
    )
    assert orders[0].accepted is False
    assert orders[0].cancel_reason == "zero_shares"
