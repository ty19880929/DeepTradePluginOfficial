"""ApwRuntime — context bundle for the accumulation-probe-washout pipeline.

Aligned with VaRuntime / LubRuntime conventions: a small dataclass holding
``Database`` + ``ConfigService`` + ``LLMManager`` + optional ``TushareClient``
+ optional ``lgb_scorer`` (always ``None`` in v0.1; field is reserved so v0.2
can wire LGB without breaking runtime construction).
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

PLUGIN_ID = "accumulation-probe-washout"


@dataclass
class ApwRuntime:
    db: Database
    config: ConfigService
    llms: LLMManager
    plugin_id: str = PLUGIN_ID
    run_id: str | None = None
    tushare: TushareClient | None = None
    # v0.6.0 — LightGBM scorer. Constructed lazily by ``execute_analyze`` via
    # :func:`lgb.scorer.build_lgb_scorer`; ``None`` when disabled by either
    # ``ApwConfig.lgb_enabled = False`` or the ``--no-lgb`` one-shot flag.
    lgb_scorer: Any | None = None

    def emit(
        self,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        payload: dict[str, object] | None = None,
        **extra: object,
    ) -> StrategyEvent:
        full: dict[str, object] = dict(payload or {})
        if extra:
            full.update(extra)
        return StrategyEvent(type=event_type, level=level, message=message, payload=full)


def build_tushare_client(rt: ApwRuntime, *, event_cb: Any = None):
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
        event_cb=event_cb,
    )


def open_worker_runtime(parent_rt: ApwRuntime, llm_provider: str | None = None) -> ApwRuntime:
    """Build a per-worker runtime for debate-mode (v0.2).

    Critical invariant (mirrors VA): each worker gets its own Database +
    LLMManager, but ``ConfigService`` is **shared with the parent on purpose** —
    SecretStore's keyring probe is racy under per-worker construction.

    v0.1 single-provider path never invokes this; API shape is reserved so the
    v0.2 debate PR doesn't have to refactor the runtime.
    """
    from deeptrade.core.db import Database  # noqa: PLC0415
    from deeptrade.core.llm_manager import LLMManager  # noqa: PLC0415
    from deeptrade.core import paths  # noqa: PLC0415

    db = Database(paths.db_path())
    return ApwRuntime(
        db=db,
        config=parent_rt.config,  # shared on purpose
        llms=LLMManager(db, parent_rt.config),
        plugin_id=parent_rt.plugin_id,
        run_id=parent_rt.run_id,
        tushare=parent_rt.tushare,
    )
