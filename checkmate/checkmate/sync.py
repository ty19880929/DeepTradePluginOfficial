"""``sync`` orchestrator — prepare local caches and survivorship snapshots.

Iter-1 PR-1.2 deliverable. The ``deeptrade checkmate sync`` CLI is a thin
shell over :func:`run_sync`, which:

1. Loads (or refreshes) the trade calendar parquet.
2. Pulls ``stock_basic`` (L+D+P) and ``namechange``, derives event rows via
   :func:`checkmate.data.build_status_history_rows`, upserts them into
   ``checkmate_stock_status_history``.
3. For each symbol in scope (``--symbols`` subset or, by default, every L
   ts_code in ``stock_basic``), pulls ``daily`` + ``adj_factor`` and
   ``daily_basic`` into per-symbol parquet caches.

What ``sync`` does NOT do (matches development_plan §11 contract):
  * No per-trade-date ``stk_limit`` pre-pull — that's executor-on-demand and
    would burn ~3000 API calls for a 12y window.
  * No ``index_daily`` pre-pull — Iter-2 ``regime`` pulls on demand.

Tushare's default RPS is 200/min (free tier) or 500/min (Pro). We deliberately
issue calls serially — `concurrent.futures` here would either trip the rate
limiter or require a token-bucket layer that the framework doesn't ship in
the plugin-facing TushareClient. Iter-3+ may revisit once we have a proper
client-side rate-limit primitive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from . import data
from .calendar import load_trade_calendar
from .runtime import CheckmateRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class SyncParams:
    start: str  # YYYY-MM-DD or YYYYMMDD
    end: str | None = None  # default: today
    symbols: list[str] | None = None  # None → all listed (status=L)
    force_refresh: bool = False


@dataclass
class SyncOutcome:
    n_symbols: int
    n_status_rows: int
    started_at: datetime
    finished_at: datetime
    cached_symbols: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_date(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    raise ValueError(f"unrecognised date: {s!r}")


def _default_end() -> str:
    return datetime.now().strftime("%Y%m%d")


def _resolve_symbol_set(
    stock_basic, user_symbols: list[str] | None,
) -> list[str]:
    """Return the ts_code list to fetch per-symbol data for.

    ``user_symbols`` (CLI ``--symbols``) takes precedence and is returned
    verbatim. Otherwise we restrict to currently-listed symbols (``L``) —
    delisted names aren't worth daily-pulling because a back-test's universe
    builder will reach them via the status snapshot and pull on demand if
    they're needed.
    """
    if user_symbols:
        return list(user_symbols)
    if stock_basic is None or stock_basic.empty:
        return []
    listed = stock_basic[stock_basic["list_status"] == "L"]
    return sorted(listed["ts_code"].astype(str).tolist())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_sync(
    rt: CheckmateRuntime,
    params: SyncParams,
    *,
    echo: Callable[[str], None] = print,
) -> SyncOutcome:
    """Execute the full sync sequence against ``rt.tushare`` + ``rt.db``.

    ``echo`` is the legacy-stream progress sink. Iter-5 will wrap this in a
    proper :class:`EventRenderer`; for now it's a callable so tests can pass
    a list-append capture.
    """
    if rt.tushare is None:
        raise RuntimeError("CheckmateRuntime.tushare is None; build_tushare_client first")

    started_at = datetime.now()
    start = _norm_date(params.start) or "20140101"
    end = _norm_date(params.end) or _default_end()
    echo(f"[sync] window={start}..{end} force_refresh={params.force_refresh}")

    # 1) Trade calendar
    echo("[sync] step=trade_cal")
    load_trade_calendar(rt.tushare, refresh=params.force_refresh)

    # 2) Stock basic + namechange → status history
    echo("[sync] step=stock_basic+namechange")
    stock_basic = data.fetch_stock_basic_all_statuses(
        rt.tushare, force_refresh=params.force_refresh,
    )
    namechange = data.fetch_namechange_history(
        rt.tushare, force_refresh=params.force_refresh,
    )
    rows = data.build_status_history_rows(stock_basic, namechange)
    n_status = data.upsert_status_history(rt.db, rows)
    echo(f"[sync] status_history rows_written={n_status}")

    # 3) Per-symbol daily / daily_basic caches
    symbols = _resolve_symbol_set(stock_basic, params.symbols)
    echo(f"[sync] step=daily n_symbols={len(symbols)}")
    cached: list[str] = []
    errors: list[str] = []
    for i, ts_code in enumerate(symbols, start=1):
        try:
            data.fetch_daily_raw(
                rt.tushare, ts_code, start, end,
                force_refresh=params.force_refresh,
            )
            data.fetch_daily_basic(
                rt.tushare, ts_code, start, end,
                force_refresh=params.force_refresh,
            )
            cached.append(ts_code)
            if i % 100 == 0:
                echo(f"[sync] progress {i}/{len(symbols)}")
        except Exception as exc:  # noqa: BLE001
            # Bad symbols / API hiccups shouldn't kill the whole sync —
            # collect, surface at the end, exit non-zero.
            logger.warning("sync failed for %s: %s", ts_code, exc)
            errors.append(f"{ts_code}: {exc}")

    finished_at = datetime.now()
    elapsed = (finished_at - started_at).total_seconds()
    echo(
        f"[sync] done n_symbols={len(cached)} n_status_rows={n_status} "
        f"errors={len(errors)} elapsed={elapsed:.1f}s"
    )
    return SyncOutcome(
        n_symbols=len(cached),
        n_status_rows=n_status,
        started_at=started_at,
        finished_at=finished_at,
        cached_symbols=cached,
        errors=errors,
    )


def parse_symbols(arg: str | None) -> list[str] | None:
    """CLI helper: split ``--symbols 600519.SH,000001.SZ`` into a list."""
    if not arg:
        return None
    out = [s.strip() for s in arg.split(",") if s.strip()]
    return out or None
