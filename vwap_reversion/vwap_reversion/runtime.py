"""VwrRuntime — context bundle the plugin's daemon / backtest run against.

与 ``LubRuntime`` 同构但更轻：本策略不调用 LLM（无 ``LLMManager``）、
单标的单进程（无 debate-mode worker 隔离问题，``ConfigService`` 直接共享）。
新增 :class:`vwap_reversion.clock.MarketClock` 字段 —— 所有「现在几点 / 今天
是哪个交易日」一律经由它（设计 §4），禁止散落 ``datetime.now()``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from .clock import MarketClock

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.config import ConfigService
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient

PLUGIN_ID = "vwap-reversion"


@dataclass
class VwrRuntime:
    """Services bundle for run / backtest / report."""

    db: Database
    config: ConfigService
    clock: MarketClock
    plugin_id: str = PLUGIN_ID
    run_id: str | None = None
    tushare: TushareClient | None = None

    def emit(
        self,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        payload: dict[str, object] | None = None,
        **extra: object,
    ) -> StrategyEvent:
        # 与 LubRuntime.emit 同款双风格入参：payload= 显式 dict 或 ad-hoc kwargs。
        full: dict[str, object] = dict(payload or {})
        if extra:
            full.update(extra)
        return StrategyEvent(type=event_type, level=level, message=message, payload=full)


def build_tushare_client(rt: VwrRuntime, *, event_cb: Any = None) -> TushareClient:
    """Construct a TushareClient bound to this plugin."""
    from deeptrade.core.tushare_client import TushareClient, TushareSDKTransport  # noqa: PLC0415

    token = rt.config.get("tushare.token")
    if not token:
        raise RuntimeError("tushare.token not configured; run `deeptrade config set-tushare`")
    cfg = rt.config.get_app_config()
    transport = TushareSDKTransport(str(token))
    return TushareClient(
        rt.db,
        transport,
        plugin_id=rt.plugin_id,
        rps=cfg.tushare_rps,
        event_cb=event_cb,
    )
