"""RiskManager — 风控闸门（设计 §13）.

开仓闸门（任一命中即抑制，返回 suppressed_by 标签）：
* ``circuit_breaker`` — 日亏熔断已触发（当日只平不开）
* ``eod_halt``        — 已过 EOD 强平时刻
* ``max_trades``      — 当日成交笔数达上限（按成交 fill 计数，开平各算一笔）
* ``cooldown``        — 距上次平腿不足 cooldown_seconds

平仓闸门：仅 ``revert_exit`` 受 ``min_holding`` 防抖；止损/熔断/EOD 永远放行
（止损被防抖拦住是本末倒置）。

被抑制的信号仍由 TradingSession 落 vwr_signals（suppressed_by 标注），
便于复盘「漏单/废单」。
"""

from __future__ import annotations

from .engine.signal import LegState


class RiskManager:
    def __init__(
        self,
        *,
        max_trades_per_day: int,
        min_holding_seconds: int,
        cooldown_seconds: int,
    ) -> None:
        self.max_trades_per_day = max_trades_per_day
        self.min_holding_seconds = min_holding_seconds
        self.cooldown_seconds = cooldown_seconds
        self.fills_today = 0
        self.last_close_ts: int | None = None
        self.circuit_broken = False
        self.eod_halted = False

    # ---- 闸门 -------------------------------------------------------------

    def check_open(self, ts: int) -> str | None:
        if self.circuit_broken:
            return "circuit_breaker"
        if self.eod_halted:
            return "eod_halt"
        # 开腿必然伴随将来一次平腿 → 留出余量，fills 已达上限-1 时不再开新腿，
        # 保证任何已开腿都还有名额平掉（不会被 max_trades 锁死在持仓里）。
        if self.fills_today >= self.max_trades_per_day - 1:
            return "max_trades"
        if (
            self.last_close_ts is not None
            and ts - self.last_close_ts < self.cooldown_seconds
        ):
            return "cooldown"
        return None

    def check_close(self, reason: str, leg: LegState, ts: int) -> str | None:
        if reason == "revert_exit" and ts - leg.entry_ts < self.min_holding_seconds:
            return "min_holding"
        return None

    # ---- 状态推进 ----------------------------------------------------------

    def note_fill(self) -> None:
        self.fills_today += 1

    def note_close(self, ts: int) -> None:
        self.last_close_ts = ts
