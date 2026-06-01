"""Data layer for market-review (design §5.2).

Single entry point: :func:`sync_window(rt, window, *, force_sync=False)`
which orchestrates Tushare calls + materializes each frame into the
``mr_*`` tables declared in ``deeptrade_plugin.yaml``.

**Scope** — PR-2 covers exactly the rows in design §5.2's sync matrix that
have a concrete ``mr_*``落表 target. Three matrix rows are deferred to
later PRs because their target sets aren't known until later metric work:

- ``ths_daily`` + ``dc_index`` — board catalog needs ``ConceptRepository``;
  PR-3 ``sectors.py`` adds the sync alongside its own metrics.
- ``cyq_perf`` — design §5.2 explicitly limits this to "龙头候选集（成本控制）",
  so PR-3 ``leaders.py`` issues the per-candidate calls.

APIs declared ``required`` in yaml but **not** in §5.2's落表 matrix
(``moneyflow_dc``, ``kpl_concept``, ``kpl_list``, ``index_weight``) are
caller-issued from later PRs (capital / sectors / style metrics). They
remain in yaml so the framework grants permission, and
``check_registry`` allows declared-but-unused entries.

Fetch patterns (per §5.2 column "调用粒度"):

- **One-shot**: ``stock_basic``, ``trade_cal``.
- **Per-trade-date all-market**: ``daily`` / ``daily_basic`` / ``moneyflow``.
  Mandatory loop — Tushare caps single responses at ~6000 rows, so an
  all-market range query silently truncates (lub v0.18 design note).
- **Per-trade-date scalar/small**: ``stock_st`` / ``suspend_d`` /
  ``limit_step`` / ``limit_cpt_list`` / ``top_list`` / ``top_inst`` /
  ``ths_hot`` / ``dc_hot`` / ``margin`` / ``block_trade``.
- **Range**: ``limit_list_d`` / ``limit_list_ths`` /
  ``moneyflow_hsgt`` / ``hsgt_top10`` /
  ``moneyflow_mkt_dc`` / ``moneyflow_ind_ths`` / ``moneyflow_cnt_ths``.
- **Per-index range**: ``index_daily`` / ``index_dailybasic`` over the
  8 wide-base indices in :data:`WIDE_BASE_INDICES`.

The :func:`fetch_latest_trade_date` helper mirrors lub's contract: probe
Tushare ``index_daily(000001.SH)`` to ground "today" in published market
data instead of the local clock (design §3.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.tushare_client import TushareClient

    from .runtime import MrRuntime
    from .windows import Window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade-date probe (design §3.3)
# ---------------------------------------------------------------------------

LATEST_TRADE_DATE_PROBE_INDEX = "000001.SH"


def fetch_latest_trade_date(
    tushare: TushareClient,
    *,
    index_code: str = LATEST_TRADE_DATE_PROBE_INDEX,
    lookback_days: int = 60,
) -> str:
    """Return the most recent published ``trade_date`` per ``index_daily``.

    T is derived from market data (the latest ``trade_date`` Tushare has
    published) rather than ``datetime.now()``, so a wrong host clock /
    timezone doesn't silently anchor T to a wrong "today". The local clock
    only bounds the probe window (``[now - lookback_days, now + 1d]``);
    skew exceeding ``lookback_days`` raises — the correct failure mode.

    ``force_sync=True`` is mandatory: ``index_daily`` is classified
    ``trade_day_immutable``, so without it a stale cached window would be
    served after the next trading day publishes.
    """
    now = datetime.now()
    start_date = (now - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = (now + timedelta(days=1)).strftime("%Y%m%d")
    df = tushare.call(
        "index_daily",
        params={"ts_code": index_code, "start_date": start_date, "end_date": end_date},
        force_sync=True,
    )
    if df is None or df.empty or "trade_date" not in df.columns:
        raise RuntimeError(
            f"index_daily({index_code}) probe returned no rows over "
            f"{start_date}..{end_date}; cannot resolve latest trade date. "
            "传入 --trade-date <YYYYMMDD> 覆盖，或检查 Tushare token / 网络。"
        )
    return str(df["trade_date"].astype(str).max())


# ---------------------------------------------------------------------------
# Index basket (design §5.2 — "9 个宽基")
# ---------------------------------------------------------------------------

# 8 wide-base indices the 8000-credit Tushare tier reliably serves.
# 万得全A (``881001.WI``) is gated behind the WIND data package and would
# silently degrade the风格 axis if cached as null; we'd rather omit it
# explicitly than fail half-way. PR-3 ``style.py`` works with 8 indices.
WIDE_BASE_INDICES: tuple[str, ...] = (
    "000001.SH",   # 上证综指
    "399001.SZ",   # 深证成指
    "399006.SZ",   # 创业板指
    "000688.SH",   # 科创50
    "899050.BJ",   # 北证50
    "000300.SH",   # 沪深300
    "000905.SH",   # 中证500
    "000852.SH",   # 中证1000
)


# ---------------------------------------------------------------------------
# Sync orchestration
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Per-table row counters + per-API empty-day list returned by
    :func:`sync_window`. Informational only — pipeline / metrics never read
    these; tests + dashboard / events do.

    ``empty_days[api]`` records the trade_dates that returned empty *despite
    being open* — design §11.2 considers these legitimate data-dimension
    gaps (北向 on节假日, kpl ranks on stale days, etc.). v0.1 keeps the run
    as ``success`` and the affected section just elides that field.
    """

    rows_materialized: dict[str, int] = field(default_factory=dict)
    empty_days: dict[str, list[str]] = field(default_factory=dict)
    duration_seconds: float = 0.0


def sync_window(
    rt: MrRuntime,
    window: Window,
    *,
    force_sync: bool = False,
) -> SyncResult:
    """Fetch + materialize PR-2's API set over ``window``.

    Mutates only ``mr_*`` tables. Reads ``rt.tushare`` (caller must have
    built it via :func:`market_review.runtime.build_tushare_client`).
    Tushare cache is honored unless ``force_sync=True`` is passed.

    Raises whatever Tushare raises — design §11.1 makes every API failure
    fatal for v0.1 (no degradation path). Empty responses are NOT failures;
    they appear in :attr:`SyncResult.empty_days` only.
    """
    if rt.tushare is None:
        raise RuntimeError(
            "MrRuntime.tushare is None; call build_tushare_client(rt) first"
        )
    tushare = rt.tushare

    started = datetime.now()
    res = SyncResult()
    open_days = list(window.trade_dates)
    start, end = window.start, window.end

    # --- One-shot (no date arg) ---------------------------------------------
    _one_shot(tushare, "stock_basic", "mr_stock_basic", ["ts_code"], res, force_sync=force_sync)
    _one_shot(
        tushare,
        "trade_cal",
        "mr_trade_cal",
        ["exchange", "cal_date"],
        res,
        force_sync=force_sync,
    )

    # --- Per-day all-market (mandatory loop — Tushare row cap) --------------
    _per_day(
        tushare, "daily", "mr_daily",
        key_cols=["ts_code", "trade_date"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "daily_basic", "mr_daily_basic",
        key_cols=["ts_code", "trade_date"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "moneyflow", "mr_moneyflow",
        key_cols=["ts_code", "trade_date"],
        open_days=open_days, res=res, force_sync=force_sync,
    )

    # --- Per-day scalar / small frames --------------------------------------
    _per_day(
        tushare, "stock_st", "mr_stock_st",
        key_cols=["ts_code", "trade_date"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "suspend_d", "mr_suspend_d",
        key_cols=["ts_code", "trade_date", "suspend_type"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "limit_step", "mr_limit_step",
        key_cols=["trade_date", "ts_code"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "limit_cpt_list", "mr_limit_cpt_list",
        key_cols=["trade_date", "ts_code"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "top_list", "mr_top_list",
        key_cols=["trade_date", "ts_code", "reason"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "top_inst", "mr_top_inst",
        key_cols=["trade_date", "ts_code", "exalter", "side", "reason"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "ths_hot", "mr_hot",
        key_cols=["trade_date", "source", "rank"],
        open_days=open_days, res=res, force_sync=force_sync,
        transform=_transform_hot("ths"),
    )
    _per_day(
        tushare, "dc_hot", "mr_hot",
        key_cols=["trade_date", "source", "rank"],
        open_days=open_days, res=res, force_sync=force_sync,
        transform=_transform_hot("dc"),
    )
    _per_day(
        tushare, "margin", "mr_margin",
        key_cols=["trade_date", "exchange_id"],
        open_days=open_days, res=res, force_sync=force_sync,
    )
    _per_day(
        tushare, "block_trade", "mr_block_trade",
        key_cols=["trade_date", "ts_code", "buyer", "seller"],
        open_days=open_days, res=res, force_sync=force_sync,
    )

    # --- Range (single API call covering start..end) ------------------------
    _range(
        tushare, "limit_list_d", "mr_limit_list_d",
        key_cols=["trade_date", "ts_code", "limit"],
        start=start, end=end, res=res, force_sync=force_sync,
    )
    _range(
        tushare, "limit_list_ths", "mr_limit_ths",
        key_cols=["trade_date", "ts_code", "limit_type"],
        start=start, end=end, res=res, force_sync=force_sync,
    )
    _range(
        tushare, "moneyflow_hsgt", "mr_moneyflow_hsgt",
        key_cols=["trade_date"],
        start=start, end=end, res=res, force_sync=force_sync,
    )
    _range(
        tushare, "hsgt_top10", "mr_hsgt_top10",
        key_cols=["trade_date", "ts_code", "market_type"],
        start=start, end=end, res=res, force_sync=force_sync,
    )
    _range(
        tushare, "moneyflow_mkt_dc", "mr_moneyflow_mkt",
        key_cols=["trade_date"],
        start=start, end=end, res=res, force_sync=force_sync,
    )
    _range(
        tushare, "moneyflow_ind_ths", "mr_moneyflow_ind_ths",
        key_cols=["trade_date", "name"],
        start=start, end=end, res=res, force_sync=force_sync,
    )
    _range(
        tushare, "moneyflow_cnt_ths", "mr_moneyflow_cnt_ths",
        key_cols=["trade_date", "ts_code"],
        start=start, end=end, res=res, force_sync=force_sync,
    )

    # --- Per-index range ----------------------------------------------------
    for code in WIDE_BASE_INDICES:
        _per_index_range(
            tushare, "index_daily", "mr_index_daily",
            key_cols=["ts_code", "trade_date"],
            ts_code=code, start=start, end=end,
            res=res, force_sync=force_sync,
        )
        _per_index_range(
            tushare, "index_dailybasic", "mr_index_dailybasic",
            key_cols=["ts_code", "trade_date"],
            ts_code=code, start=start, end=end,
            res=res, force_sync=force_sync,
        )

    res.duration_seconds = (datetime.now() - started).total_seconds()
    return res


def sync_sector_quotes(
    rt: MrRuntime,
    window: Window,
    *,
    force_sync: bool = False,
) -> SyncResult:
    """Fetch + materialize ``ths_daily`` (per-day full board catalog) and
    ``dc_index`` (anchor-day only) — §5.2 board-quote rows that PR-2 deferred.

    Split out from :func:`sync_window` because sectors data is consumed only
    by :mod:`market_review.metrics.sectors`; runs that only need breadth /
    sentiment / capital / risk metrics can skip this step.

    ``ths_daily`` is the primary source: it returns every同花顺 board's
    daily line. ``dc_index`` is fetched only for ``window.anchor`` per
    design §5.2 (东财 mirror exists for互验 spot-checks; persisting full
    window doubles cost for negligible value).

    Tushare's ``ths_daily`` doesn't accept ``trade_date``; we issue a
    one-day range query (``start_date=end_date=d``) per open day.
    """
    if rt.tushare is None:
        raise RuntimeError(
            "MrRuntime.tushare is None; call build_tushare_client(rt) first"
        )
    tushare = rt.tushare

    started = datetime.now()
    res = SyncResult()
    open_days = list(window.trade_dates)

    for d in open_days:
        df = tushare.call(
            "ths_daily",
            params={"start_date": d, "end_date": d},
            force_sync=force_sync,
        )
        if df is None or df.empty:
            res.empty_days.setdefault("ths_daily", []).append(d)
            continue
        # ths_daily occasionally elides ``trade_date`` when the day-range
        # is collapsed; back-fill from the loop variable so the PK
        # (ts_code, trade_date) is always populated.
        if "trade_date" not in df.columns:
            df = df.assign(trade_date=d)
        else:
            df = df.copy()
            df["trade_date"] = df["trade_date"].astype(str)
        n = tushare.materialize(
            "mr_ths_daily", df, key_cols=["ts_code", "trade_date"],
        )
        res.rows_materialized["mr_ths_daily"] = (
            res.rows_materialized.get("mr_ths_daily", 0) + int(n)
        )

    # 东财 boards — anchor day only (design §5.2 "仅末日").
    df = tushare.call(
        "dc_index",
        params={"start_date": window.anchor, "end_date": window.anchor},
        force_sync=force_sync,
    )
    if df is not None and not df.empty:
        if "trade_date" not in df.columns:
            df = df.assign(trade_date=window.anchor)
        else:
            df = df.copy()
            df["trade_date"] = df["trade_date"].astype(str)
        n = tushare.materialize(
            "mr_dc_index", df, key_cols=["ts_code", "trade_date"],
        )
        res.rows_materialized["mr_dc_index"] = (
            res.rows_materialized.get("mr_dc_index", 0) + int(n)
        )

    res.duration_seconds = (datetime.now() - started).total_seconds()
    return res


# ---------------------------------------------------------------------------
# Fetch helpers — small functions, all converge on ``tushare.materialize``
# ---------------------------------------------------------------------------


def _one_shot(
    tushare: TushareClient,
    api: str,
    table: str,
    key_cols: list[str],
    res: SyncResult,
    *,
    force_sync: bool,
) -> None:
    df = tushare.call(api, force_sync=force_sync)
    if df is None or df.empty:
        return
    n = tushare.materialize(table, df, key_cols=key_cols)
    res.rows_materialized[table] = res.rows_materialized.get(table, 0) + int(n)


def _per_day(
    tushare: TushareClient,
    api: str,
    table: str,
    *,
    key_cols: list[str],
    open_days: list[str],
    res: SyncResult,
    force_sync: bool,
    transform: Callable[..., pd.DataFrame] | None = None,
) -> None:
    for d in open_days:
        df = tushare.call(api, trade_date=d, force_sync=force_sync)
        if df is None or df.empty:
            res.empty_days.setdefault(api, []).append(d)
            continue
        if transform is not None:
            df = transform(df, trade_date=d)
            if df is None or df.empty:
                res.empty_days.setdefault(api, []).append(d)
                continue
        n = tushare.materialize(table, df, key_cols=key_cols)
        res.rows_materialized[table] = res.rows_materialized.get(table, 0) + int(n)


def _range(
    tushare: TushareClient,
    api: str,
    table: str,
    *,
    key_cols: list[str],
    start: str,
    end: str,
    res: SyncResult,
    force_sync: bool,
) -> None:
    df = tushare.call(
        api,
        params={"start_date": start, "end_date": end},
        force_sync=force_sync,
    )
    if df is None or df.empty:
        return
    n = tushare.materialize(table, df, key_cols=key_cols)
    res.rows_materialized[table] = res.rows_materialized.get(table, 0) + int(n)


def _per_index_range(
    tushare: TushareClient,
    api: str,
    table: str,
    *,
    key_cols: list[str],
    ts_code: str,
    start: str,
    end: str,
    res: SyncResult,
    force_sync: bool,
) -> None:
    df = tushare.call(
        api,
        params={"ts_code": ts_code, "start_date": start, "end_date": end},
        force_sync=force_sync,
    )
    if df is None or df.empty:
        return
    n = tushare.materialize(table, df, key_cols=key_cols)
    res.rows_materialized[table] = res.rows_materialized.get(table, 0) + int(n)


# ---------------------------------------------------------------------------
# Per-API transforms (column renames, source injection)
# ---------------------------------------------------------------------------


def _transform_hot(source: str) -> Callable[..., pd.DataFrame]:
    """Map ``{ths,dc}_hot`` frames into the ``mr_hot`` schema.

    Tushare returns ``ts_name`` + ``hot`` columns and (in some SDK versions)
    omits ``trade_date`` because the caller already pinned it. mr_hot
    expects ``name`` + ``hot_value`` + ``source`` + ``trade_date``; we
    synthesize each missing piece and coerce ``rank`` to ``int`` so the
    composite PK doesn't reject string ranks.
    """

    def _t(df: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
        out = df.copy()
        if "trade_date" not in out.columns:
            out["trade_date"] = trade_date
        else:
            out["trade_date"] = out["trade_date"].astype(str)
        if "ts_name" in out.columns and "name" not in out.columns:
            out = out.rename(columns={"ts_name": "name"})
        if "hot" in out.columns and "hot_value" not in out.columns:
            out = out.rename(columns={"hot": "hot_value"})
        out["source"] = source
        if "rank" in out.columns:
            out["rank"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
            out = out[out["rank"].notna()]
            out["rank"] = out["rank"].astype(int)
        keep_cols = [
            c for c in ("trade_date", "source", "rank", "ts_code", "name", "hot_value")
            if c in out.columns
        ]
        return out[keep_cols]

    return _t
