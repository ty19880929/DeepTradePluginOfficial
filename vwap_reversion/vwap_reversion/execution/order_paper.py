"""Order-level paper broker for the vwap-reversion live-dry-run path.

The existing :mod:`paper` broker is intentionally synchronous and remains the
stable default for strategy parity tests. This module models the lifecycle that
a real broker adapter will expose: accepted orders, partial fills, cancellation,
expiry, account snapshots, and position snapshots.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from enum import Enum

from ..schemas import Side
from .base import ExecutionRejected, Fill


class OrderStatus(Enum):
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class OrderIntent:
    side: Side
    qty: int
    ref_price: float
    ts: int
    price_protect_bps: float
    ttl_seconds: int = 120
    client_order_id: str | None = None


@dataclass
class PaperOrder:
    order_id: str
    side: Side
    qty: int
    remaining_qty: int
    limit_price: float
    ref_price: float
    submitted_ts: int
    ttl_seconds: int
    status: OrderStatus


@dataclass(frozen=True)
class BrokerEvent:
    ts: int
    order_id: str
    status: OrderStatus
    message: str
    fill: Fill | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    position: int
    equity: float


class PaperOrderBroker:
    def __init__(
        self,
        *,
        cash: float,
        fee_bps: float,
        base_slippage_bps: float,
        max_participation_rate: float = 0.10,
        impact_slippage_coef: float = 2.0,
        position: int = 0,
    ) -> None:
        if cash < 0 or position < 0:
            raise ValueError(f"cash/position 不可为负（cash={cash}, position={position}）")
        if not (0 < max_participation_rate <= 1):
            raise ValueError("max_participation_rate 必须落在 (0, 1]")
        if fee_bps < 0 or base_slippage_bps < 0 or impact_slippage_coef < 0:
            raise ValueError("fee/slippage/impact 参数不可为负")
        self._cash = float(cash)
        self._position = int(position)
        self._fee_bps = float(fee_bps)
        self._base_slippage_bps = float(base_slippage_bps)
        self._max_participation_rate = float(max_participation_rate)
        self._impact_slippage_coef = float(impact_slippage_coef)
        self._orders: dict[str, PaperOrder] = {}
        self._events: list[BrokerEvent] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position(self) -> int:
        return self._position

    def submit_order(self, intent: OrderIntent) -> PaperOrder:
        if intent.qty <= 0 or intent.qty % 100 != 0:
            raise ExecutionRejected(f"qty 必须为正的 100 整数倍（{intent.qty}）")
        if intent.ref_price <= 0:
            raise ExecutionRejected(f"ref_price 非法（{intent.ref_price}）")
        if intent.price_protect_bps < 0:
            raise ExecutionRejected("price_protect_bps 不可为负")
        if intent.side is Side.SELL and self._position < intent.qty:
            raise ExecutionRejected(f"持仓不足：需卖 {intent.qty}，仅持 {self._position}")

        protect = intent.price_protect_bps / 10_000.0
        limit_price = (
            intent.ref_price * (1.0 + protect)
            if intent.side is Side.BUY
            else intent.ref_price * (1.0 - protect)
        )
        order = PaperOrder(
            order_id=intent.client_order_id or str(uuid.uuid4()),
            side=intent.side,
            qty=intent.qty,
            remaining_qty=intent.qty,
            limit_price=limit_price,
            ref_price=intent.ref_price,
            submitted_ts=intent.ts,
            ttl_seconds=intent.ttl_seconds,
            status=OrderStatus.ACCEPTED,
        )
        self._orders[order.order_id] = order
        self._events.append(BrokerEvent(intent.ts, order.order_id, order.status, "accepted"))
        return order

    def cancel_order(self, order_id: str, *, ts: int) -> BrokerEvent:
        order = self._orders.get(order_id)
        if order is None:
            raise ExecutionRejected(f"unknown order_id: {order_id}")
        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED}:
            event = BrokerEvent(ts, order_id, order.status, "already terminal")
            self._events.append(event)
            return event
        order.status = OrderStatus.CANCELED
        event = BrokerEvent(ts, order_id, OrderStatus.CANCELED, "canceled")
        self._events.append(event)
        return event

    def process_bar(self, *, ts: int, last: float, interval_vol: float) -> list[BrokerEvent]:
        if last <= 0:
            raise ExecutionRejected(f"last 非法（{last}）")
        events: list[BrokerEvent] = []
        for order in list(self._orders.values()):
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED}:
                continue
            if ts - order.submitted_ts > order.ttl_seconds:
                order.status = OrderStatus.EXPIRED
                event = BrokerEvent(ts, order.order_id, OrderStatus.EXPIRED, "expired")
                self._events.append(event)
                events.append(event)
                continue
            if not self._marketable(order, last):
                continue

            fill_qty = min(order.remaining_qty, self._fillable_qty(interval_vol))
            if fill_qty <= 0:
                continue
            fill = self._apply_fill(order.side, fill_qty, last, interval_vol, ts)
            order.remaining_qty -= fill_qty
            order.status = (
                OrderStatus.FILLED if order.remaining_qty == 0 else OrderStatus.PARTIALLY_FILLED
            )
            event = BrokerEvent(ts, order.order_id, order.status, "fill", fill=fill)
            self._events.append(event)
            events.append(event)
        return events

    def poll(self) -> list[BrokerEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def account(self, *, mark_price: float) -> AccountSnapshot:
        return AccountSnapshot(
            cash=self._cash,
            position=self._position,
            equity=self._cash + self._position * mark_price,
        )

    def orders(self) -> list[PaperOrder]:
        return list(self._orders.values())

    def seed_position(self, qty: int) -> None:
        if qty < 0:
            raise ValueError(f"底仓不可为负（{qty}）")
        self._position += int(qty)

    def _marketable(self, order: PaperOrder, last: float) -> bool:
        if order.side is Side.BUY:
            return last <= order.limit_price
        return last >= order.limit_price

    def _fillable_qty(self, interval_vol: float) -> int:
        raw = max(0.0, interval_vol) * self._max_participation_rate
        return int(raw // 100) * 100

    def _apply_fill(
        self, side: Side, qty: int, ref_price: float, interval_vol: float, ts: int
    ) -> Fill:
        slip_bps = self._base_slippage_bps + self._impact_slippage_coef * math.sqrt(
            qty / max(interval_vol, 1.0)
        )
        slip = ref_price * slip_bps / 10_000.0
        if side is Side.BUY:
            px = ref_price + slip
            notional = px * qty
            fee = notional * self._fee_bps / 10_000.0
            if self._cash < notional + fee:
                raise ExecutionRejected(f"现金不足：需 {notional + fee:.2f}，仅有 {self._cash:.2f}")
            self._cash -= notional + fee
            self._position += qty
        else:
            if self._position < qty:
                raise ExecutionRejected(f"持仓不足：需卖 {qty}，仅持 {self._position}")
            px = ref_price - slip
            notional = px * qty
            fee = notional * self._fee_bps / 10_000.0
            self._cash += notional - fee
            self._position -= qty
        return Fill(
            ts=ts,
            side=side,
            qty=qty,
            price=px,
            fee=fee,
            slippage_cost=abs(px - ref_price) * qty,
            cash_after=self._cash,
            position_after=self._position,
        )
