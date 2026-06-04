"""Broker 抽象（设计 §5 execution/）.

P2 仅有 :class:`~vwap_reversion.execution.paper.PaperBroker`；P5 接真实
执行（QMT 等）时实现同一协议，TradingSession 不感知差异。

约定：``fill`` 是同步撮合 —— paper 模式立即成交；真实 broker 实现时由
适配层把异步回报封装成同步语义（或在 P5 扩展协议）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schemas import Side


class ExecutionRejected(Exception):
    """撮合前置检查失败（现金/持仓不足）。调用方据此记 suppressed 信号。"""


@dataclass(frozen=True)
class Fill:
    """一次成交回报。``price`` 为含滑点的实际成交价。"""

    ts: int
    side: Side
    qty: int
    price: float
    fee: float
    slippage_cost: float     # |fill_price − ref_price| × qty（滑点成本，元）
    cash_after: float
    position_after: int


class Broker(Protocol):
    @property
    def cash(self) -> float: ...

    @property
    def position(self) -> int: ...

    def fill(self, side: Side, qty: int, ref_price: float, ts: int) -> Fill:
        """以 ref_price 为基准撮合。失败抛 :class:`ExecutionRejected`。"""
        ...
