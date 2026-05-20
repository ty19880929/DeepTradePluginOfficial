"""CheckmateRuntime — context bundle for the Checkmate pipeline.

Aligned with ApwRuntime / VaRuntime / LubRuntime conventions, but **without
the ``llms`` field** — Checkmate v0.1 is LLM-free per development_plan §1
(``permissions.llm: false``). The runtime exposes:

* ``db``         — framework :class:`~deeptrade.core.db.Database`
* ``config``     — :class:`~deeptrade.core.config.ConfigService` (namespace ``checkmate.*``)
* ``tushare``    — optional :class:`~deeptrade.core.tushare_client.TushareClient`
* ``plugin_id`` / ``run_id``
* ``backtest_cache_dir`` — ``~/.deeptrade/checkmate/backtests/`` parquet
  checkpoint root (Iter-4 BacktestRunner consumes; reserved here so callers
  don't have to compute the path twice).

Iter-0 PR-0.3 ships the dataclass + ``open_runtime`` helper only; the runner
that actually consumes it lands in Iter-1+.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.config import ConfigService
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient

PLUGIN_ID = "checkmate"


def _default_backtest_cache_dir() -> Path:
    # Lazy-import paths so this module is safe to import in environments
    # where the framework isn't installed (e.g. very early CI smoke tests).
    from deeptrade.core import paths  # noqa: PLC0415

    root = Path(paths.user_data_dir()) if hasattr(paths, "user_data_dir") else Path.home() / ".deeptrade"
    return root / "checkmate" / "backtests"


@dataclass
class CheckmateRuntime:
    db: "Database"
    config: "ConfigService"
    plugin_id: str = PLUGIN_ID
    run_id: str | None = None
    tushare: "TushareClient | None" = None
    backtest_cache_dir: Path | None = None
    # Reserved for v0.5+ LightGBM scoring; always None in v0.1.
    lgb_scorer: Any | None = None


def open_runtime() -> tuple["Database", CheckmateRuntime]:
    """Build a single-thread runtime tied to the framework's default DB path.

    The caller owns the returned :class:`Database` and is responsible for
    closing it (``try / finally``). The runtime itself is plain data and does
    not need explicit teardown.
    """
    from deeptrade.core import paths  # noqa: PLC0415
    from deeptrade.core.config import ConfigService  # noqa: PLC0415
    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(paths.db_path())
    cfg = ConfigService(db)
    rt = CheckmateRuntime(
        db=db,
        config=cfg,
        backtest_cache_dir=_default_backtest_cache_dir(),
    )
    return db, rt


def build_tushare_client(rt: CheckmateRuntime, *, event_cb: Any = None) -> "TushareClient":
    """Construct a TushareClient bound to ``rt.db`` and the framework config.

    Mirrors ApwRuntime.build_tushare_client; not invoked in Iter-0 (the CLI
    stubs short-circuit before any data call), but the import-site smoke test
    of ``runtime.py`` already exercises this path's compile-time correctness.
    """
    from deeptrade.core.tushare_client import TushareClient, TushareSDKTransport  # noqa: PLC0415

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
