"""LubRuntime — context bundle the plugin's pipeline runs against.

Replaces the old framework-provided ``StrategyContext``. The plugin owns its
own runtime now: it constructs db / config / tushare itself, and obtains LLM
clients on-demand from the framework's :class:`LLMManager`.

v0.6 — ``llm: DeepSeekClient`` field removed. ``llms: LLMManager`` is the
new framework hand-off; runner / pipeline pull a per-provider ``LLMClient``
via ``rt.llms.get_client(name, plugin_id=, run_id=)``.

v0.13.0 (P1-4) — debate-mode worker threads no longer share the main thread's
``ConfigService`` (and therefore no longer reach across into the main thread's
``Database._conn`` via ``ConfigService.get``). Each worker gets a
:class:`_FrozenConfigService` built from a :class:`ProviderConfigSnapshot`
that the main thread captures once before fan-out. See ``open_worker_runtime``
and ``docs/limit-up-board/debate_isolation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.config import AppConfig, ConfigService
    from deeptrade.core.db import Database
    from deeptrade.core.llm_manager import LLMManager
    from deeptrade.core.tushare_client import TushareClient

    from .lgb.scorer import LgbScorer

PLUGIN_ID = "limit-up-board"


@dataclass
class LubRuntime:
    """Services bundle the plugin's run() / sync_data() use.

    ``llms`` is the framework's LLMManager — call
    ``rt.llms.get_client(name, plugin_id=rt.plugin_id, run_id=rt.run_id, ...)``
    to obtain a per-provider client. The plugin may use multiple providers
    in the same run.

    v0.5+ — ``lgb_scorer`` is constructed once by ``LubRunner.execute`` (lazy
    booster load) and shared by every batch in the run, including debate-mode
    worker threads (LightGBM ``Booster.predict`` is thread-safe; the scorer
    holds no mutable state once loaded — see lightgbm_design.md §7.2).
    """

    db: Database
    config: ConfigService
    llms: LLMManager
    plugin_id: str = PLUGIN_ID
    run_id: str | None = None
    tushare: TushareClient | None = None
    lgb_scorer: LgbScorer | None = None
    # P1-I/K: deterministic 64-char sha256 of this run's business inputs
    # (Round1Bundle + LubConfig + LGB model + schema/prompt versions). Set
    # by :class:`LubRunner` at the end of Step 1; surfaced in summary.md and
    # available for the future framework LLM replay cache key. Stays ``None``
    # for sync-only runs (no LLM stage to feed it into).
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
        # Accept both styles: ``payload={...}`` (explicit dict) and ad-hoc
        # kwargs (``providers=[...]``). Without the explicit ``payload=`` slot
        # the kwargs path would bury the caller's dict under a literal
        # ``"payload"`` key, breaking every dashboard handler that reads
        # ``ev.payload.get("min_float_mv_yi")`` etc.
        full: dict[str, object] = dict(payload or {})
        if extra:
            full.update(extra)
        return StrategyEvent(type=event_type, level=level, message=message, payload=full)


def build_tushare_client(
    rt: LubRuntime,
    *,
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
        event_cb=event_cb,
    )


# ---------------------------------------------------------------------------
# v0.13.0 (P1-4) — Worker config isolation: ProviderConfigSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProviderSnapshot:
    """Single LLM provider entry, captured at snapshot-build time."""

    name: str
    base_url: str
    model: str
    timeout: int
    is_default: bool
    api_key: str  # may be "" when the user hasn't configured one


@dataclass(frozen=True)
class ProviderConfigSnapshot:
    """Immutable subset of ``AppConfig`` + secrets that workers need.

    Built once on the main thread by :func:`build_provider_config_snapshot`
    and handed to each worker. Workers never read from the main thread's
    ``ConfigService`` / ``SecretStore`` / ``Database`` afterwards — the
    snapshot is the entire source of truth.
    """

    providers: dict[str, _ProviderSnapshot]
    default_provider: str | None
    app_timezone: str
    app_log_level: str
    app_profile: str
    tushare_rps: float
    tushare_timeout: int
    tushare_max_retries: int
    llm_audit_full_payload: bool


def build_provider_config_snapshot(config: ConfigService) -> ProviderConfigSnapshot:
    """Capture :class:`ConfigService` state into a thread-safe snapshot.

    Runs on the main thread (single ``SecretStore`` keyring probe still happens
    inside ``ConfigService.get`` for each ``llm.<name>.api_key`` lookup) and
    returns a frozen dataclass that worker threads use via
    :class:`_FrozenConfigService`.
    """
    cfg = config.get_app_config()
    providers: dict[str, _ProviderSnapshot] = {}
    for name, p in cfg.llm_providers.items():
        api_key = config.get(f"llm.{name}.api_key") or ""
        providers[name] = _ProviderSnapshot(
            name=name,
            base_url=p.base_url,
            model=p.model,
            timeout=p.timeout,
            is_default=p.is_default,
            api_key=str(api_key),
        )
    return ProviderConfigSnapshot(
        providers=providers,
        default_provider=config.get_default_llm_provider(),
        app_timezone=cfg.app_timezone,
        app_log_level=cfg.app_log_level,
        app_profile=cfg.app_profile,
        tushare_rps=cfg.tushare_rps,
        tushare_timeout=cfg.tushare_timeout,
        tushare_max_retries=cfg.tushare_max_retries,
        llm_audit_full_payload=cfg.llm_audit_full_payload,
    )


class _FrozenConfigService:
    """Read-only :class:`ConfigService` surrogate built from a snapshot.

    Implements the subset that ``LLMManager`` actually calls
    (``get_app_config``, ``get(\"llm.<name>.api_key\")``,
    ``get_default_llm_provider``). Workers never touch the main thread's
    ``Database._conn`` / ``SecretStore`` keyring.

    Write operations (``set``, ``set_llm_provider``, ...) raise
    :class:`RuntimeError`; workers must not mutate global config.
    """

    def __init__(self, snapshot: ProviderConfigSnapshot) -> None:
        self._snap = snapshot

    def get_app_config(self) -> AppConfig:
        from deeptrade.core.config import AppConfig, LLMProviderConfig  # noqa: PLC0415

        providers = {
            name: LLMProviderConfig(
                base_url=p.base_url,
                model=p.model,
                timeout=p.timeout,
                is_default=p.is_default,
            )
            for name, p in self._snap.providers.items()
        }
        return AppConfig(
            app_timezone=self._snap.app_timezone,
            app_log_level=self._snap.app_log_level,
            app_profile=self._snap.app_profile,
            tushare_rps=self._snap.tushare_rps,
            tushare_timeout=self._snap.tushare_timeout,
            tushare_max_retries=self._snap.tushare_max_retries,
            llm_providers=providers,
            llm_audit_full_payload=self._snap.llm_audit_full_payload,
        )

    def get(self, key: str) -> Any:
        # Secret-routed: llm.<name>.api_key 是 worker 唯一会访问的 secret 键。
        if key.startswith("llm.") and key.endswith(".api_key"):
            name = key[len("llm."): -len(".api_key")]
            p = self._snap.providers.get(name)
            return p.api_key if p and p.api_key else None
        # Non-secret AppConfig accessors are surfaced via get_app_config(),
        # but a handful of callers do ``config.get(\"app.tushare_rps\")`` style
        # reads — fall back to the snapshot's fields.
        flat = {
            "app.timezone": self._snap.app_timezone,
            "app.log_level": self._snap.app_log_level,
            "app.profile": self._snap.app_profile,
            "tushare.rps": self._snap.tushare_rps,
            "tushare.timeout": self._snap.tushare_timeout,
            "tushare.max_retries": self._snap.tushare_max_retries,
            "llm.audit_full_payload": self._snap.llm_audit_full_payload,
        }
        if key in flat:
            return flat[key]
        return None

    def get_default_llm_provider(self) -> str | None:
        return self._snap.default_provider

    def list_all(self) -> dict[str, Any]:  # pragma: no cover — LLMManager 不调用
        out: dict[str, Any] = {}
        for name, p in self._snap.providers.items():
            out[f"llm.{name}.base_url"] = p.base_url
            out[f"llm.{name}.model"] = p.model
            out[f"llm.{name}.is_default"] = p.is_default
        return out

    def set(self, *args, **kwargs):  # noqa: ANN001, ARG002
        raise RuntimeError("_FrozenConfigService is read-only; workers must not write config")

    def set_llm_provider(self, *args, **kwargs):  # noqa: ANN001, ARG002
        raise RuntimeError("_FrozenConfigService is read-only; workers must not write config")

    def delete(self, *args, **kwargs):  # noqa: ANN001, ARG002
        raise RuntimeError("_FrozenConfigService is read-only; workers must not write config")

    def delete_llm_provider(self, *args, **kwargs):  # noqa: ANN001, ARG002
        raise RuntimeError("_FrozenConfigService is read-only; workers must not write config")

    def source_of(self, key: str) -> str:  # pragma: no cover
        return "snapshot"


def open_worker_runtime(
    plugin_id: str,
    run_id: str,
    *,
    config_snapshot: ProviderConfigSnapshot,
    lgb_scorer: LgbScorer | None = None,
) -> tuple[Database, LubRuntime]:
    """Construct an isolated runtime for a debate-mode worker thread.

    v0.13.0 (P1-4) — workers no longer share the main thread's
    ``ConfigService``. ``config_snapshot`` is captured once on the main
    thread via :func:`build_provider_config_snapshot`; each worker wraps it
    in :class:`_FrozenConfigService` for ``LLMManager`` to consume.

    Each worker gets its own DuckDB connection + ``LLMManager``. Multiple
    ``duckdb.connect()`` calls against the same file share the underlying
    DB instance in-process, so plugin-owned table writes (e.g. event /
    stage_results inserts) still land in the same physical file.

    ``lgb_scorer`` is passed straight through (read-only reference share). The
    main thread loads the booster once; debate workers see the same object and
    rely on LightGBM's thread-safe ``Booster.predict``. See lightgbm_design.md
    §7.2 for the concurrency contract.

    The worker MUST close ``db`` when done.
    """
    from deeptrade.core import paths  # noqa: PLC0415
    from deeptrade.core.db import Database  # noqa: PLC0415
    from deeptrade.core.llm_manager import LLMManager  # noqa: PLC0415

    db = Database(paths.db_path())
    frozen_config = _FrozenConfigService(config_snapshot)
    rt = LubRuntime(
        db=db,
        config=frozen_config,  # type: ignore[arg-type]
        llms=LLMManager(db, frozen_config),  # type: ignore[arg-type]
        plugin_id=plugin_id,
        run_id=run_id,
        lgb_scorer=lgb_scorer,
    )
    return db, rt


def pick_llm_provider(rt: LubRuntime, override: str | None = None) -> str | None:
    """Pick which configured LLM provider to use for this run.

    ``override`` (v0.6.8+) — CLI ``--llm`` value when the user pinned a
    provider for this run; takes precedence and is returned verbatim. When
    None, defers the choice to the framework default
    (``LLMProviderConfig.is_default`` resolved by ``LLMManager.get_client``).
    The plugin keeps this hook so a future revision can reintroduce a
    plugin-specific persisted default (e.g. ``limit-up-board.default_llm``)
    without touching the runner.
    """
    return override
