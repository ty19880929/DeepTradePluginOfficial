"""LubRuntime — context bundle the plugin's pipeline runs against.

Replaces the old framework-provided ``StrategyContext``. The plugin owns its
own runtime now: it constructs db / config / tushare itself, and obtains LLM
clients on-demand from the framework's :class:`LLMManager`.

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

PLUGIN_ID = "limit-up-board"


@dataclass
class LubRuntime:
    """Services bundle the plugin's run() / sync_data() use.

    ``llms`` is the framework's LLMManager — call
    ``rt.llms.get_client(name, plugin_id=rt.plugin_id, run_id=rt.run_id, ...)``
    to obtain a per-provider client. The plugin may use multiple providers
    in the same run.
    """

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


def build_tushare_client(
    rt: LubRuntime,
    *,
    intraday: bool = False,
    event_cb: Any = None,
) -> TushareClient:
    """Construct a TushareClient bound to this plugin."""
    from deeptrade.core.tushare_client import TushareClient, TushareSDKTransport

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
        intraday=intraday,
        event_cb=event_cb,
    )


def open_worker_runtime(
    plugin_id: str,
    run_id: str,
    *,
    config: ConfigService,
    is_intraday: bool = False,
) -> tuple[Database, LubRuntime]:
    """Construct an isolated runtime for a debate-mode worker thread.

    Each worker gets its own DuckDB connection + ``LLMManager`` so that
    concurrent ``LLMClient.complete_json`` calls don't share the lock /
    audit-write bookkeeping. Same-process multiple ``duckdb.connect()`` calls
    against the same file share the underlying DB instance, so writes still
    land in the same physical file (run history visible across all workers).

    The ``ConfigService`` (and its ``SecretStore``) is **shared** with the
    main thread on purpose: ``SecretStore`` probes the OS keyring at
    construction time with a side-effecting set/get/delete round-trip, and
    running that probe per worker is both wasteful and racy — concurrent
    workers overwrite each other's probe key, causing false negatives that
    silently demote keyring-stored secrets to "no api_key set".

    Sharing implies that ``ConfigService`` reads (e.g. ``get_app_config``)
    are issued against the **main thread's** ``Database._conn`` from worker
    threads. ``Database.fetchone`` / ``fetchall`` MUST therefore hold their
    write lock across the full execute+fetch round-trip; otherwise concurrent
    workers race on the connection's result set and trigger a native heap
    corruption (Windows 0xC0000374). See ``deeptrade.core.db``.

    The worker MUST close ``db`` when done.
    """
    from deeptrade.core import paths
    from deeptrade.core.db import Database
    from deeptrade.core.llm_manager import LLMManager

    db = Database(paths.db_path())
    rt = LubRuntime(
        db=db,
        config=config,
        llms=LLMManager(db, config),
        plugin_id=plugin_id,
        run_id=run_id,
        is_intraday=is_intraday,
    )
    return db, rt


def pick_llm_provider(rt: LubRuntime) -> str | None:
    """Pick which configured LLM provider to use for this run.

    Returning None defers the choice to the framework default
    (``LLMProviderConfig.is_default`` resolved by ``LLMManager.get_client``).
    The plugin keeps this hook so a future revision can reintroduce a
    plugin-specific override (e.g. ``limit-up-board.default_llm``) without
    touching the runner.
    """
    return None
