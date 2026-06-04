"""EventRenderer 协议（设计 §11，仿 limit_up_board/ui/protocol.py）.

daemon 只面向这个协议 emit 事件；两个实现：
* :class:`~vwap_reversion.ui.dashboard.RichDashboardRenderer` —— Live 双面板
* :class:`~vwap_reversion.ui.legacy.LegacyStreamRenderer` —— 行式输出

约定的 payload["kind"]（dashboard 据此分流到面板）：
* ``sample``  → 指标区刷新 + 执行记录面板
* ``trade``   → 交易记录面板（P2 起）
* 其余        → 执行记录面板
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from ..clock import MarketClock


@dataclass(frozen=True)
class RunMeta:
    """渲染器开场需要的运行元信息（header 区）。"""

    run_id: str
    code: str
    trade_date: str
    mode: str                # paper / backtest
    params_summary: str      # 一行参数摘要（k 值 / 轮询间隔等）


class EventRenderer(Protocol):
    def start(self, meta: RunMeta, clock: MarketClock) -> None: ...

    def handle(self, event: StrategyEvent) -> None: ...

    def finish(self) -> None: ...
