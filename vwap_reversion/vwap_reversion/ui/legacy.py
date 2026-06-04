"""行式渲染器 —— 每事件一行 stdout（非 TTY / CI / --no-dashboard 的降级路径）。

格式：``  {glyph} [{event_type}] {上海时间} {message}``（时间按市场时区显示）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .protocol import RunMeta

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from ..clock import MarketClock

_GLYPHS = {"info": "·", "warn": "⚠", "error": "✘"}


class LegacyStreamRenderer:
    def __init__(self) -> None:
        self._clock: MarketClock | None = None

    def start(self, meta: RunMeta, clock: MarketClock) -> None:
        self._clock = clock
        print(
            f"▶ vwap-reversion {meta.mode} · {meta.code} · {meta.trade_date} · "
            f"run_id={meta.run_id}"
        )
        print(f"  params: {meta.params_summary}")

    def handle(self, event: StrategyEvent) -> None:
        glyph = _GLYPHS.get(str(event.level.value), "·")
        hhmmss = self._clock.now().strftime("%H:%M:%S") if self._clock else "--:--:--"
        print(f"  {glyph} [{event.type.value}] {hhmmss} {event.message}")

    def finish(self) -> None:
        print("■ run finished")
