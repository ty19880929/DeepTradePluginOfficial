from __future__ import annotations

from vwap_reversion.execution.adapter import ShadowBrokerAdapter
from vwap_reversion.execution.order_paper import OrderIntent, OrderStatus
from vwap_reversion.schemas import Side


def test_shadow_broker_accepts_without_real_fill() -> None:
    broker = ShadowBrokerAdapter(cash=100_000, position=1000)
    order = broker.submit_order(
        "159518.SZ",
        OrderIntent(Side.BUY, 100, 1.0, 1, 20.0, client_order_id="shadow-1"),
    )
    assert order.status is OrderStatus.ACCEPTED
    assert order.remaining_qty == 100
    assert broker.account(mark_price=1.0).position == 1000
    events = broker.poll()
    assert events[0].message == "shadow accepted"
    assert broker.poll() == []


def test_shadow_broker_reconcile_clean_and_cancel() -> None:
    broker = ShadowBrokerAdapter(cash=100_000, position=1000, code="159518.SZ")
    event = broker.cancel_order("missing", ts=2)
    assert event.status is OrderStatus.CANCELED
    assert broker.positions()[0].available_qty == 1000
    assert broker.reconcile() == []
