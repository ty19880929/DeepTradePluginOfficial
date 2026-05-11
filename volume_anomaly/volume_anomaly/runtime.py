"""VaRuntime — context bundle the volume-anomaly plugin's pipeline runs against.

v0.6 — ``llm: DeepSeekClient`` field removed. ``llms: LLMManager`` is the
new framework hand-off; runner / pipeline pull a per-provider ``LLMClient``
via ``rt.llms.get_client(name, plugin_id=, run_id=)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.config import ConfigService
    from deeptrade.core.db import Database
    from deeptrade.core.llm_manager import LLMManager
    from deeptrade.core.tushare_client import TushareClient

PLUGIN_ID = "volume-anomaly"


@dataclass
class VaRuntime:
    db: Database
    config: ConfigService
    llms: LLMManager
    plugin_id: str = PLUGIN_ID
    run_id: str | None = None
    is_intraday: bool = False
    tushare: TushareClient | None = None

    def emit(
        self,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        **payload: object,
    ) -> StrategyEvent:
        return StrategyEvent(type=event_type, level=level, message=message, payload=dict(payload))


def build_tushare_client(rt: VaRuntime, *, intraday: bool = False, event_cb: Any = None):
    from deeptrade.core.tushare_client import TushareClient, TushareSDKTransport

    token = rt.config.get("tushare.token")
    if not token:
        raise RuntimeError("tushare.token not configured; run `deeptrade config set-tushare`")
    cfg = rt.config.get_app_config()
    return TushareClient(
        rt.db,
        TushareSDKTransport(str(token)),
        plugin_id=rt.plugin_id,
        rps=cfg.tushare_rps,
        intraday=intraday,
        event_cb=event_cb,
    )


def pick_llm_provider(rt: VaRuntime) -> str | None:
    """Pick which configured LLM provider to use for this run.

    Returning None defers the choice to the framework default
    (``LLMProviderConfig.is_default`` resolved by ``LLMManager.get_client``).
    Kept as a plugin-side hook so a future revision can reintroduce a
    plugin-specific override without touching the runner.
    """
    return None
