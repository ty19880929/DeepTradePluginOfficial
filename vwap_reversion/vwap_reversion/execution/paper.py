"""PaperBroker — 模拟撮合（设计 §10.1 / §13 成本模型）.

撮合假设：
* 成交价 = ``ref_price × (1 ± slippage_bps/1e4)``（买加卖减，保守偏不利）
* 佣金   = ``max(成交额 × fee_bps/1e4, min_fee_per_trade)``（ETF 无印花税）
* 现金/持仓硬约束：买入需 ``cash ≥ 成交额+佣金``；卖出需 ``position ≥ qty``。
  违约抛 :class:`ExecutionRejected`（调用方落 suppressed 信号，不崩 run）。

纯账务对象，不持有策略状态（leg/盈亏归 TradingSession 管）。
"""

from __future__ import annotations

from ..schemas import Side
from .base import ExecutionRejected, Fill


class PaperBroker:
    def __init__(
        self,
        *,
        cash: float,
        fee_bps: float,
        slippage_bps: float,
        min_fee_per_trade: float = 0.0,
        position: int = 0,
    ) -> None:
        if cash < 0 or position < 0:
            raise ValueError(f"cash/position 不可为负（cash={cash}, position={position}）")
        if min_fee_per_trade < 0:
            raise ValueError(f"min_fee_per_trade 不可为负（{min_fee_per_trade}）")
        self._cash = float(cash)
        self._position = int(position)
        self._fee_bps = float(fee_bps)
        self._min_fee = float(min_fee_per_trade)
        self._slip_bps = float(slippage_bps)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position(self) -> int:
        return self._position

    def seed_position(self, qty: int) -> None:
        """底仓做 T：把既有底仓注入账户（不产生成交/费用，设计 §2.3）。"""
        if qty < 0:
            raise ValueError(f"底仓不可为负（{qty}）")
        self._position += int(qty)

    def fill(self, side: Side, qty: int, ref_price: float, ts: int) -> Fill:
        if qty <= 0 or qty % 100 != 0:
            raise ExecutionRejected(f"qty 必须为正的 100 整数倍（{qty}）")
        if ref_price <= 0:
            raise ExecutionRejected(f"ref_price 非法（{ref_price}）")

        slip = ref_price * self._slip_bps / 1e4
        if side is Side.BUY:
            px = ref_price + slip
            notional = px * qty
            fee = self._fee(notional)
            if self._cash < notional + fee:
                raise ExecutionRejected(
                    f"现金不足：需 {notional + fee:.2f}，仅有 {self._cash:.2f}"
                )
            self._cash -= notional + fee
            self._position += qty
        else:
            if self._position < qty:
                raise ExecutionRejected(
                    f"持仓不足：需卖 {qty}，仅持 {self._position}"
                )
            px = ref_price - slip
            notional = px * qty
            fee = self._fee(notional)
            self._cash += notional - fee
            self._position -= qty

        return Fill(
            ts=ts, side=side, qty=qty, price=px, fee=fee,
            slippage_cost=abs(px - ref_price) * qty,
            cash_after=self._cash, position_after=self._position,
        )

    def _fee(self, notional: float) -> float:
        return max(notional * self._fee_bps / 10_000.0, self._min_fee)
