from __future__ import annotations

import pytest

from vwap_reversion.execution.base import ExecutionRejected
from vwap_reversion.execution.order_paper import (
    OrderIntent,
    OrderStatus,
    PaperOrderBroker,
)
from vwap_reversion.schemas import Side


def broker(**kw) -> PaperOrderBroker:
    base = dict(
        cash=10_000.0,
        fee_bps=5.0,
        base_slippage_bps=10.0,
        max_participation_rate=0.10,
        impact_slippage_coef=0.0,
    )
    base.update(kw)
    return PaperOrderBroker(**base)


def intent(side: Side = Side.BUY, qty: int = 300, ts: int = 1) -> OrderIntent:
    return OrderIntent(
        side=side,
        qty=qty,
        ref_price=1.0,
        ts=ts,
        price_protect_bps=20.0,
        client_order_id="ord-1",
    )


def test_submit_and_partial_fill_then_full_fill() -> None:
    b = broker()
    order = b.submit_order(intent(qty=300))
    assert order.status is OrderStatus.ACCEPTED

    events = b.process_bar(ts=2, last=1.0, interval_vol=1000)
    assert len(events) == 1
    assert events[0].status is OrderStatus.PARTIALLY_FILLED
    assert events[0].fill is not None and events[0].fill.qty == 100
    assert b.position == 100
    assert b.orders()[0].remaining_qty == 200

    b.process_bar(ts=3, last=1.0, interval_vol=2000)
    order = b.orders()[0]
    assert order.status is OrderStatus.FILLED
    assert order.remaining_qty == 0
    assert b.position == 300


def test_limit_price_must_be_marketable() -> None:
    b = broker()
    b.submit_order(intent(qty=100))
    events = b.process_bar(ts=2, last=1.1, interval_vol=10_000)
    assert events == []
    assert b.orders()[0].status is OrderStatus.ACCEPTED


def test_cancel_order() -> None:
    b = broker()
    order = b.submit_order(intent(qty=100))
    event = b.cancel_order(order.order_id, ts=2)
    assert event.status is OrderStatus.CANCELED
    assert b.process_bar(ts=3, last=1.0, interval_vol=10_000) == []


def test_order_expires() -> None:
    b = broker()
    b.submit_order(OrderIntent(Side.BUY, 100, 1.0, 1, 20.0, ttl_seconds=5))
    events = b.process_bar(ts=7, last=1.0, interval_vol=10_000)
    assert events[0].status is OrderStatus.EXPIRED


def test_sell_requires_position_and_account_snapshot() -> None:
    b = broker()
    with pytest.raises(ExecutionRejected, match="持仓不足"):
        b.submit_order(intent(side=Side.SELL, qty=100))
    b.seed_position(200)
    b.submit_order(intent(side=Side.SELL, qty=100))
    events = b.process_bar(ts=2, last=1.0, interval_vol=10_000)
    assert events[0].status is OrderStatus.FILLED
    snap = b.account(mark_price=1.0)
    assert snap.position == 100
    assert snap.equity == pytest.approx(b.cash + 100)


@pytest.mark.parametrize("qty", [0, 50, -100])
def test_rejects_bad_qty(qty: int) -> None:
    with pytest.raises(ExecutionRejected):
        broker().submit_order(intent(qty=qty))
