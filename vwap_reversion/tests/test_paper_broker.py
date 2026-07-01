"""PaperBroker 撮合账务（P2 验收项）。纯 stdlib。"""

from __future__ import annotations

import pytest

from vwap_reversion.execution.base import ExecutionRejected
from vwap_reversion.execution.paper import PaperBroker
from vwap_reversion.schemas import Side


def broker(cash: float = 10_000.0) -> PaperBroker:
    # fee 5bps / slip 10bps —— 取整数好手算
    return PaperBroker(cash=cash, fee_bps=5.0, slippage_bps=10.0)


def test_buy_math_exact() -> None:
    b = broker()
    fill = b.fill(Side.BUY, 100, 1.0, ts=1)
    assert fill.price == pytest.approx(1.001)            # 1.0 × (1 + 10bps)
    assert fill.fee == pytest.approx(1.001 * 100 * 5e-4)  # 0.05005
    assert fill.slippage_cost == pytest.approx(0.1)       # 0.001 × 100
    assert b.position == 100
    assert b.cash == pytest.approx(10_000 - 100.1 - 0.05005)
    assert fill.cash_after == pytest.approx(b.cash)
    assert fill.position_after == 100


def test_min_fee_per_trade_floor() -> None:
    b = PaperBroker(
        cash=10_000.0,
        fee_bps=2.5,
        min_fee_per_trade=5.0,
        slippage_bps=0.0,
    )
    fill = b.fill(Side.BUY, 100, 1.0, ts=1)
    assert fill.fee == pytest.approx(5.0)
    assert b.cash == pytest.approx(10_000.0 - 100.0 - 5.0)


def test_percentage_fee_applies_above_floor() -> None:
    b = PaperBroker(
        cash=100_000.0,
        fee_bps=2.5,
        min_fee_per_trade=5.0,
        slippage_bps=0.0,
    )
    fill = b.fill(Side.BUY, 10_000, 3.0, ts=1)
    assert fill.fee == pytest.approx(30_000.0 * 2.5 / 10_000.0)


def test_sell_math_exact() -> None:
    b = broker()
    b.fill(Side.BUY, 100, 1.0, ts=1)
    fill = b.fill(Side.SELL, 100, 1.01, ts=2)
    assert fill.price == pytest.approx(1.01 * (1 - 1e-3))  # 1.00899
    assert fill.fee == pytest.approx(1.00899 * 100 * 5e-4)
    assert b.position == 0
    # 现金守恒：初始 − 买成本 + 卖回款
    expected = 10_000 - (100.1 + 0.05005) + (100.899 - 0.0504495)
    assert b.cash == pytest.approx(expected)


def test_insufficient_cash_rejected() -> None:
    b = PaperBroker(cash=50.0, fee_bps=5.0, slippage_bps=10.0)
    with pytest.raises(ExecutionRejected, match="现金不足"):
        b.fill(Side.BUY, 100, 1.0, ts=1)
    assert b.cash == 50.0 and b.position == 0  # 原子性：拒单不动账


def test_insufficient_position_rejected() -> None:
    b = broker()
    with pytest.raises(ExecutionRejected, match="持仓不足"):
        b.fill(Side.SELL, 100, 1.0, ts=1)


@pytest.mark.parametrize("qty", [0, -100, 150, 99])
def test_qty_must_be_positive_lot(qty: int) -> None:
    with pytest.raises(ExecutionRejected, match="100 整数倍"):
        broker().fill(Side.BUY, qty, 1.0, ts=1)


def test_seed_position_no_fee_no_cash_change() -> None:
    b = broker()
    b.seed_position(1000)
    assert b.position == 1000
    assert b.cash == 10_000.0
    # 底仓可以直接卖（做 T 高抛腿）
    fill = b.fill(Side.SELL, 100, 1.0, ts=1)
    assert fill.position_after == 900
