"""MrRuntime — services bundle the plugin's pipeline runs against (design §2.3).

Replaces the framework's removed ``StrategyContext``. The plugin owns its
runtime: it builds db / config / tushare itself and obtains LLM clients
on-demand from the framework's :class:`LLMManager`.

PR-2 scope: only the dataclass + a thin ``build_tushare_client`` helper +
``emit()`` glue for :class:`StrategyEvent` construction.

PR-4..6 will add fingerprinting / concept-repo plumbing as runner.py /
pipeline.py come online.

**Not included** (per design §2.3): no ``_FrozenConfigService`` /
``ProviderConfigSnapshot`` machinery — market-review will never run in
debate mode (design §1.3 "永不引入"). Workers don't fan out; everything
runs on the main thread.
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
    from deeptrade.plugins_api import ConceptRepository

PLUGIN_ID = "market-review"


@dataclass
class MrRuntime:
    """Services bundle for the market-review pipeline.

    ``llms`` is the framework's LLMManager — call
    ``rt.llms.get_client(name, plugin_id=rt.plugin_id, run_id=rt.run_id, ...)``
    to obtain a per-provider client. The pipeline runs 7 sections through
    the same provider (design §5.4); we still go through the manager so the
    framework's audit trail captures every call.
    """

    db: Database
    config: ConfigService
    llms: LLMManager
    plugin_id: str = PLUGIN_ID
    run_id: str | None = None
    tushare: TushareClient | None = None
    # PR-4+ — framework v0.14.0 ths_concept_* snapshot, injected by ctx.
    concept_repo: ConceptRepository | None = None
    # PR-5 — 64-char sha256 of normalized inputs (window + metrics blob +
    # plugin version), filled by runner.py before LLM stage. Mirrors lub's
    # ``LubRuntime.input_fingerprint`` contract.
    input_fingerprint: str | None = None

    def emit(
        self,
        event_type: EventType,
        message: str,
        *,
        level: EventLevel = EventLevel.INFO,
        payload: dict[str, object] | None = None,
        **extra: object,
    ) -> StrategyEvent:
        """Convenience constructor for :class:`StrategyEvent`.

        Same shape as ``LubRuntime.emit``: ``payload`` is merged with ad-hoc
        kwargs into ``StrategyEvent.payload``. Without the explicit
        ``payload=`` slot, ad-hoc kwargs would otherwise get buried under a
        literal ``"payload"`` key (lub bug-fix carried over verbatim).
        """
        full: dict[str, object] = dict(payload or {})
        if extra:
            full.update(extra)
        return StrategyEvent(type=event_type, level=level, message=message, payload=full)


def build_tushare_client(
    rt: MrRuntime,
    *,
    event_cb: Any = None,
) -> TushareClient:
    """Construct a :class:`TushareClient` bound to this plugin.

    Mirrors ``limit_up_board.runtime.build_tushare_client``: pulls the token
    from ``rt.config`` (env / keyring routed through ConfigService) and
    wires up an SDK transport. Raises :class:`RuntimeError` if no token is
    configured — that's a user-config issue, not a runtime fault, and
    failing fast keeps the "should I run sync" decision close to the user.
    """
    from deeptrade.core.tushare_client import (  # noqa: PLC0415
        TushareClient,
        TushareSDKTransport,
    )

    token = rt.config.get("tushare.token")
    if not token:
        raise RuntimeError(
            "tushare.token 未配置；运行 `deeptrade config set-tushare` 设置 token"
        )
    cfg = rt.config.get_app_config()
    transport = TushareSDKTransport(str(token))
    return TushareClient(
        rt.db,
        transport,
        plugin_id=rt.plugin_id,
        rps=cfg.tushare_rps,
        event_cb=event_cb,
    )
