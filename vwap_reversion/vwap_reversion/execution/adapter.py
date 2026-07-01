"""Broker adapter boundary for shadow/live execution.

Concrete securities-broker SDKs vary in login, order, callback, and query
contracts. The strategy should depend only on this narrow boundary so that
paper, shadow, and live modes can share signal/risk code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .order_paper import AccountSnapshot, BrokerEvent, OrderIntent, OrderStatus, PaperOrder


@dataclass(frozen=True)
class BrokerPosition:
    code: str
    qty: int
    available_qty: int
    cost_price: float | None = None


class BrokerAdapter(Protocol):
    def submit_order(self, code: str, intent: OrderIntent) -> PaperOrder: ...

    def cancel_order(self, order_id: str, *, ts: int) -> BrokerEvent: ...

    def poll(self) -> list[BrokerEvent]: ...

    def positions(self) -> list[BrokerPosition]: ...

    def account(self, *, mark_price: float) -> AccountSnapshot: ...

    def reconcile(self) -> list[str]:
        """Return human-readable reconciliation warnings. Empty means clean."""
        ...


class ShadowBrokerAdapter:
    """Dry-run adapter: accepts strategy orders but never sends them to a broker."""

    def __init__(self, *, cash: float, position: int = 0, code: str = "") -> None:
        self._cash = float(cash)
        self._position = int(position)
        self._code = code
        self._orders: dict[str, PaperOrder] = {}
        self._events: list[BrokerEvent] = []

    def submit_order(self, code: str, intent: OrderIntent) -> PaperOrder:
        order = PaperOrder(
            order_id=intent.client_order_id or f"shadow-{len(self._orders) + 1}",
            side=intent.side,
            qty=intent.qty,
            remaining_qty=intent.qty,
            limit_price=intent.ref_price,
            ref_price=intent.ref_price,
            submitted_ts=intent.ts,
            ttl_seconds=intent.ttl_seconds,
            status=OrderStatus.ACCEPTED,
        )
        self._code = code
        self._orders[order.order_id] = order
        self._events.append(
            BrokerEvent(intent.ts, order.order_id, OrderStatus.ACCEPTED, "shadow accepted")
        )
        return order

    def cancel_order(self, order_id: str, *, ts: int) -> BrokerEvent:
        order = self._orders.get(order_id)
        if order is not None:
            order.status = OrderStatus.CANCELED
        event = BrokerEvent(ts, order_id, OrderStatus.CANCELED, "shadow canceled")
        self._events.append(event)
        return event

    def poll(self) -> list[BrokerEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def positions(self) -> list[BrokerPosition]:
        return [
            BrokerPosition(
                code=self._code,
                qty=self._position,
                available_qty=self._position,
            )
        ] if self._code else []

    def account(self, *, mark_price: float) -> AccountSnapshot:
        return AccountSnapshot(
            cash=self._cash,
            position=self._position,
            equity=self._cash + self._position * mark_price,
        )

    def reconcile(self) -> list[str]:
        return []
