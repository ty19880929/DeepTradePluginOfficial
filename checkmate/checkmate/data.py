"""Tushare thin wrappers + local parquet cache for Checkmate.

Design (development_plan §1, §11):

* **Double-track adjustment.** Cache stores raw daily merged with
  ``adj_factor`` (one frame per ts_code). ``fetch_daily_raw`` returns the raw
  columns unchanged; ``fetch_daily_qfq`` derives a front-adjusted view on the
  fly using ``adj_factor[t] / latest_adj_factor`` — the same approach
  Tushare's own qfq endpoint takes, but here we keep both sides materialised
  side-by-side so the executor can use raw + ``stk_limit`` for真实交易所
  口径 while features/signals consume qfq.

* **Long-period caches.** ``sync`` populates these parquet files once;
  subsequent ``scan`` / ``signals`` / ``backtest`` calls just slice the cache.
  Cache extension policy: if a request window isn't fully covered by the
  cache, re-fetch the full window and union with the existing rows (last
  write wins, keyed by trade_date).

* **Survivorship snapshots (PR-1.2).** ``fetch_namechange_history`` +
  ``fetch_stock_basic_all_statuses`` pull the raw upstream frames;
  ``build_status_history_rows`` derives one row per (ts_code, change-date)
  event; ``upsert_status_history`` / ``query_status_as_of`` are the DB shim
  that ``universe.py`` (PR-1.3) consumes for "what was tradable on date X".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import paths

logger = logging.getLogger(__name__)

# Columns we expect from Tushare's ``daily`` endpoint. Anything missing on a
# given pull is left untouched — the merged frame may carry extra columns
# without disrupting downstream consumers.
_DAILY_PRICE_COLS = ("open", "high", "low", "close", "pre_close")
_DAILY_REQUIRED_COLS = ("ts_code", "trade_date") + _DAILY_PRICE_COLS


# ---------------------------------------------------------------------------
# Cache primitives
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("corrupt parquet cache at %s (%s); ignoring", path, exc)
        return None


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows with ``start <= trade_date <= end``, sorted ascending."""
    if df.empty:
        return df
    s = df.copy()
    s["trade_date"] = s["trade_date"].astype(str)
    s = s[(s["trade_date"] >= start) & (s["trade_date"] <= end)]
    return s.sort_values("trade_date").reset_index(drop=True)


def _covers_range(df: pd.DataFrame | None, start: str, end: str) -> bool:
    """True iff the cached frame has at least one row at ``start`` and at ``end``.

    We don't try to verify every intermediate trade day is present — Tushare's
    ``daily`` is already gap-free for a continuous symbol, and missing edges
    are what users care about (a partial window means we likely skipped a
    sync). A symbol that simply didn't trade on those bookend days (newly
    listed / suspended) still re-fetches and merges harmlessly.
    """
    if df is None or df.empty:
        return False
    dates = df["trade_date"].astype(str)
    return bool((dates >= start).any() and (dates <= end).any() and (dates.min() <= start) and (dates.max() >= end))


def _union_by_trade_date(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge fresh rows into existing cache, last-write-wins per trade_date."""
    fresh = fresh.copy()
    if "trade_date" in fresh.columns:
        fresh["trade_date"] = fresh["trade_date"].astype(str)
    if existing is None or existing.empty:
        return fresh.drop_duplicates("trade_date", keep="last").sort_values("trade_date").reset_index(drop=True)
    existing = existing.copy()
    existing["trade_date"] = existing["trade_date"].astype(str)
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined = combined.drop_duplicates("trade_date", keep="last")
    return combined.sort_values("trade_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Daily (raw + qfq, single cache)
# ---------------------------------------------------------------------------


def _daily_cache_path(ts_code: str, *, cache_root: Path | None = None) -> Path:
    root = cache_root or paths.daily_cache_dir()
    return root / f"{ts_code}.parquet"


def _fetch_daily_merged(
    tushare: Any,
    ts_code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """One-shot pull of ``daily`` + ``adj_factor`` for ``ts_code`` over [start, end]."""
    daily = tushare.call("daily", ts_code=ts_code, start_date=start, end_date=end)
    if daily is None:
        daily = pd.DataFrame()
    daily = daily.copy()
    if not daily.empty:
        missing = set(_DAILY_REQUIRED_COLS) - set(daily.columns)
        if missing:
            raise RuntimeError(
                f"tushare daily({ts_code}) missing columns {sorted(missing)}; "
                "schema drift — bump pinned tushare version or extend wrapper."
            )
        daily["trade_date"] = daily["trade_date"].astype(str)

    adj = tushare.call("adj_factor", ts_code=ts_code, start_date=start, end_date=end)
    if adj is None:
        adj = pd.DataFrame()
    if not adj.empty:
        adj = adj.copy()
        if "trade_date" not in adj.columns or "adj_factor" not in adj.columns:
            raise RuntimeError(
                f"tushare adj_factor({ts_code}) missing trade_date/adj_factor columns"
            )
        adj["trade_date"] = adj["trade_date"].astype(str)
        adj = adj[["ts_code", "trade_date", "adj_factor"]] if "ts_code" in adj.columns else adj[["trade_date", "adj_factor"]]

    if daily.empty:
        return daily  # nothing to merge against

    if adj.empty:
        # Defensive: missing adj_factor means qfq == raw. Carry a constant
        # column so downstream consumers don't need to special-case it.
        daily["adj_factor"] = 1.0
        return daily.sort_values("trade_date").reset_index(drop=True)

    merged = daily.merge(
        adj[["trade_date", "adj_factor"]] if "ts_code" not in adj.columns else adj.drop(columns=["ts_code"], errors="ignore"),
        on="trade_date",
        how="left",
    )
    # Forward-fill within the symbol's history is unsafe (would propagate a
    # stale factor across a split day). Tushare returns one adj_factor per
    # trade_date when daily exists, so a NaN here implies a real gap — set to
    # 1.0 and rely on downstream qfq view to degrade gracefully.
    merged["adj_factor"] = merged["adj_factor"].fillna(1.0)
    return merged.sort_values("trade_date").reset_index(drop=True)


def _fetch_or_cache_daily(
    tushare: Any,
    ts_code: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """Internal: load cache → maybe top up from Tushare → return full cached frame."""
    cache_path = _daily_cache_path(ts_code, cache_root=cache_root)
    cached = None if force_refresh else _read_parquet(cache_path)

    if cached is not None and _covers_range(cached, start, end):
        return cached

    fresh = _fetch_daily_merged(tushare, ts_code, start=start, end=end)
    if fresh.empty and cached is not None:
        return cached
    merged = _union_by_trade_date(cached, fresh)
    if not merged.empty:
        _write_parquet(cache_path, merged)
    return merged


def fetch_daily_raw(
    tushare: Any,
    ts_code: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """Return the raw (un-adjusted) daily frame for ``ts_code`` over [start, end].

    Columns: ``ts_code``, ``trade_date``, ``open`` / ``high`` / ``low`` /
    ``close`` / ``pre_close`` (whatever Tushare ``daily`` ships) plus
    ``adj_factor`` for the executor's reverse-mapping needs.
    """
    full = _fetch_or_cache_daily(
        tushare, ts_code, start, end,
        force_refresh=force_refresh, cache_root=cache_root,
    )
    return _slice_by_date(full, start, end)


def fetch_daily_qfq(
    tushare: Any,
    ts_code: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """Return the front-adjusted (qfq) daily frame for ``ts_code`` over [start, end].

    Each price column ``X`` becomes ``X_qfq = X * adj_factor / latest_adj_factor``
    where ``latest_adj_factor`` is the most recent adj_factor present in the
    full cached frame (NOT the requested window — this matters when the
    window ends before the latest split). Raw columns and ``adj_factor`` are
    preserved so callers can still cross-reference (e.g. to spot an ex-div
    day inside a position).
    """
    full = _fetch_or_cache_daily(
        tushare, ts_code, start, end,
        force_refresh=force_refresh, cache_root=cache_root,
    )
    if full.empty:
        return full
    latest_adj = float(full.iloc[-1]["adj_factor"])
    if latest_adj <= 0:
        latest_adj = 1.0
    qfq = full.copy()
    for col in _DAILY_PRICE_COLS:
        if col in qfq.columns:
            qfq[f"{col}_qfq"] = qfq[col].astype(float) * qfq["adj_factor"].astype(float) / latest_adj
    return _slice_by_date(qfq, start, end)


# ---------------------------------------------------------------------------
# Daily basic — turnover / amount / mcap
# ---------------------------------------------------------------------------


def _daily_basic_cache_path(ts_code: str, *, cache_root: Path | None = None) -> Path:
    root = cache_root or paths.daily_basic_cache_dir()
    return root / f"{ts_code}.parquet"


def fetch_daily_basic(
    tushare: Any,
    ts_code: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """Return ``daily_basic`` rows (turnover_rate, amount, total_mv, …) for [start, end]."""
    cache_path = _daily_basic_cache_path(ts_code, cache_root=cache_root)
    cached = None if force_refresh else _read_parquet(cache_path)
    if cached is not None and _covers_range(cached, start, end):
        return _slice_by_date(cached, start, end)

    fresh = tushare.call("daily_basic", ts_code=ts_code, start_date=start, end_date=end)
    if fresh is None:
        fresh = pd.DataFrame()
    if not fresh.empty:
        fresh = fresh.copy()
        fresh["trade_date"] = fresh["trade_date"].astype(str)
    merged = _union_by_trade_date(cached, fresh)
    if not merged.empty:
        _write_parquet(cache_path, merged)
    return _slice_by_date(merged, start, end)


# ---------------------------------------------------------------------------
# Stk_limit — per-day全市场涨跌停 (used by executor)
# ---------------------------------------------------------------------------


def _stk_limit_cache_path(trade_date: str, *, cache_root: Path | None = None) -> Path:
    root = cache_root or paths.stk_limit_cache_dir()
    return root / f"{trade_date}.parquet"


def fetch_stk_limit(
    tushare: Any,
    trade_date: str,
    *,
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """Return the entire market's up/down limit prices on ``trade_date``.

    Cached per-day because the data is immutable post-close and the entire
    market fits comfortably in one parquet (~5000 rows × 5 cols).
    """
    cache_path = _stk_limit_cache_path(trade_date, cache_root=cache_root)
    if not force_refresh:
        cached = _read_parquet(cache_path)
        if cached is not None and not cached.empty:
            return cached

    df = tushare.call("stk_limit", trade_date=trade_date)
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        df = df.copy()
        df["trade_date"] = df["trade_date"].astype(str)
        _write_parquet(cache_path, df)
    return df


# ---------------------------------------------------------------------------
# Index daily
# ---------------------------------------------------------------------------


def _index_daily_cache_path(index_code: str, *, cache_root: Path | None = None) -> Path:
    root = cache_root or paths.index_daily_cache_dir()
    return root / f"{index_code}.parquet"


def fetch_index_daily(
    tushare: Any,
    index_code: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
    cache_root: Path | None = None,
) -> pd.DataFrame:
    """Return ``index_daily`` close/open/high/low/amount for [start, end]."""
    cache_path = _index_daily_cache_path(index_code, cache_root=cache_root)
    cached = None if force_refresh else _read_parquet(cache_path)
    if cached is not None and _covers_range(cached, start, end):
        return _slice_by_date(cached, start, end)

    fresh = tushare.call("index_daily", ts_code=index_code, start_date=start, end_date=end)
    if fresh is None:
        fresh = pd.DataFrame()
    if not fresh.empty:
        fresh = fresh.copy()
        fresh["trade_date"] = fresh["trade_date"].astype(str)
    merged = _union_by_trade_date(cached, fresh)
    if not merged.empty:
        _write_parquet(cache_path, merged)
    return _slice_by_date(merged, start, end)


# ---------------------------------------------------------------------------
# Survivorship — namechange + stock_basic (L+D+P)
# ---------------------------------------------------------------------------


def _namechange_cache_path() -> Path:
    return paths.cache_dir() / "namechange.parquet"


def _stock_basic_cache_path() -> Path:
    return paths.cache_dir() / "stock_basic_all.parquet"


def fetch_namechange_history(
    tushare: Any,
    *,
    force_refresh: bool = False,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Pull the full Tushare ``namechange`` history.

    Tushare exposes name changes (and ST in/out) via the ``namechange`` API.
    Columns we rely on: ``ts_code`` / ``name`` / ``start_date``. ``end_date``
    / ``ann_date`` / ``change_reason`` are pass-through if Tushare ships them.
    """
    path = cache_path or _namechange_cache_path()
    if not force_refresh:
        cached = _read_parquet(path)
        if cached is not None and not cached.empty:
            return cached
    df = tushare.call("namechange")
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        df = df.copy()
        df["ts_code"] = df["ts_code"].astype(str)
        df["start_date"] = df["start_date"].astype(str)
        if "name" in df.columns:
            df["name"] = df["name"].astype(str)
        _write_parquet(path, df)
    return df


def fetch_stock_basic_all_statuses(
    tushare: Any,
    *,
    force_refresh: bool = False,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Pull ``stock_basic`` for ``list_status`` ∈ {L, D, P} and concatenate.

    Tushare returns only ``L`` (listed) by default. ``D`` (delisted) and
    ``P`` (paused / suspended for re-org) require explicit ``list_status``
    arguments; combining all three yields the survivorship-complete frame
    the universe builder needs (otherwise back-tests would silently exclude
    delisted names — the canonical look-ahead bias).
    """
    path = cache_path or _stock_basic_cache_path()
    if not force_refresh:
        cached = _read_parquet(path)
        if cached is not None and not cached.empty:
            return cached

    frames: list[pd.DataFrame] = []
    for status in ("L", "D", "P"):
        sub = tushare.call(
            "stock_basic",
            list_status=status,
            fields="ts_code,name,industry,market,exchange,list_status,list_date,delist_date",
        )
        if sub is not None and not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["ts_code", "list_status"], keep="last").reset_index(drop=True)
    # Normalise key text columns so downstream comparisons don't silently mix
    # numpy.str_ / NaN.
    for col in ("ts_code", "name", "industry", "market", "exchange", "list_status"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ("list_date", "delist_date"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: None if (pd.isna(v) or str(v) == "nan") else str(v))
    _write_parquet(path, df)
    return df


def _is_st_name(name: str | None) -> bool:
    """True iff ``name`` carries the A-share ST tag (warning or serious)."""
    if not name:
        return False
    n = str(name).strip()
    return n.startswith("*ST") or n.startswith("ST")


def build_status_history_rows(
    stock_basic: pd.DataFrame,
    namechange: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Derive one (ts_code, as_of_date) row per change event.

    Output shape (matches ``checkmate_stock_status_history`` columns):
        ts_code, as_of_date, list_status, is_st, name, industry,
        list_date, delist_date, raw_event_json

    Algorithm
    ---------
    For each ``ts_code`` present in ``stock_basic``:
      1. Emit a row at ``list_date`` using ``stock_basic.name`` as the initial
         name (so every stock has at least one row at IPO).
      2. For each ``namechange`` row keyed to this ts_code, sorted by
         ``start_date`` asc, emit one row using the new name. ``is_st`` is
         recomputed from the new name.
      3. If ``list_status`` ∈ {D, P} and ``delist_date`` is set, emit a final
         row at ``delist_date`` with ``list_status`` from ``stock_basic``.

    Rows are de-duplicated by (ts_code, as_of_date) keeping the LAST event —
    if a namechange happens on the IPO day, the namechange wins over the
    synthesized list-date row.
    """
    if stock_basic is None or stock_basic.empty:
        return []

    rows: list[dict[str, Any]] = []

    # Index namechange events by ts_code
    nc_by_code: dict[str, list[dict[str, Any]]] = {}
    if namechange is not None and not namechange.empty:
        for rec in namechange.to_dict("records"):
            ts_code = str(rec.get("ts_code", ""))
            if not ts_code:
                continue
            nc_by_code.setdefault(ts_code, []).append(rec)
        for ts_code in nc_by_code:
            nc_by_code[ts_code].sort(key=lambda r: str(r.get("start_date") or ""))

    for rec in stock_basic.to_dict("records"):
        ts_code = str(rec.get("ts_code", ""))
        if not ts_code:
            continue
        list_date = rec.get("list_date")
        if not list_date or str(list_date) in {"None", "nan"}:
            continue  # cannot anchor a synthetic row without an IPO date
        list_date = str(list_date)
        list_status = str(rec.get("list_status") or "L")
        delist_date = rec.get("delist_date")
        delist_date = str(delist_date) if delist_date and str(delist_date) not in {"None", "nan"} else None
        industry = rec.get("industry")
        initial_name = str(rec.get("name") or "")

        # 1) IPO-day synthetic row using stock_basic.name (latest known).
        rows.append({
            "ts_code": ts_code,
            "as_of_date": list_date,
            "list_status": "L",  # always L on the listing day
            "is_st": _is_st_name(initial_name),
            "name": initial_name,
            "industry": industry,
            "list_date": list_date,
            "delist_date": delist_date,
            "raw_event_json": json.dumps(
                {"source": "stock_basic", "event": "list"}, ensure_ascii=False,
            ),
        })

        # 2) Per-namechange-event rows
        for ev in nc_by_code.get(ts_code, []):
            start_date = str(ev.get("start_date") or "")
            if not start_date:
                continue
            new_name = str(ev.get("name") or "")
            rows.append({
                "ts_code": ts_code,
                "as_of_date": start_date,
                "list_status": "L",
                "is_st": _is_st_name(new_name),
                "name": new_name,
                "industry": industry,
                "list_date": list_date,
                "delist_date": delist_date,
                "raw_event_json": json.dumps(
                    {
                        "source": "namechange",
                        **{k: ev.get(k) for k in
                           ("name", "start_date", "end_date", "ann_date", "change_reason")
                           if k in ev},
                    },
                    ensure_ascii=False, default=str,
                ),
            })

        # 3) Delisting / pause synthetic row
        if list_status in {"D", "P"} and delist_date:
            last_name = initial_name
            evs = nc_by_code.get(ts_code, [])
            if evs:
                last_name = str(evs[-1].get("name") or initial_name)
            rows.append({
                "ts_code": ts_code,
                "as_of_date": delist_date,
                "list_status": list_status,
                "is_st": _is_st_name(last_name),
                "name": last_name,
                "industry": industry,
                "list_date": list_date,
                "delist_date": delist_date,
                "raw_event_json": json.dumps(
                    {"source": "stock_basic",
                     "event": "delist" if list_status == "D" else "paused"},
                    ensure_ascii=False,
                ),
            })

    # De-dupe (ts_code, as_of_date) keeping the last row (namechange beats synthetic).
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        seen[(r["ts_code"], r["as_of_date"])] = r
    return list(seen.values())


def upsert_status_history(db: Any, rows: Iterable[dict[str, Any]]) -> int:
    """INSERT OR REPLACE rows into ``checkmate_stock_status_history``.

    Returns the number of rows written. ``rows`` may be an iterator; this
    function consumes it.
    """
    now = datetime.now()
    n = 0
    for r in rows:
        db.execute(
            """
            INSERT OR REPLACE INTO checkmate_stock_status_history
                (ts_code, as_of_date, list_status, is_st, name, industry,
                 list_date, delist_date, raw_event_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                r["ts_code"], r["as_of_date"], r["list_status"], bool(r.get("is_st", False)),
                r.get("name"), r.get("industry"),
                r.get("list_date"), r.get("delist_date"),
                r.get("raw_event_json"), now,
            ],
        )
        n += 1
    return n


def query_status_as_of(db: Any, ts_code: str, as_of_date: str) -> dict[str, Any] | None:
    """Return the most recent status row at or before ``as_of_date``.

    Implements the survivorship contract: "as of trade_date, what was this
    stock's name / list_status / is_st"? Caller pre-normalises ``as_of_date``
    to ``YYYYMMDD``.
    """
    row = db.execute(
        """
        SELECT ts_code, as_of_date, list_status, is_st, name, industry,
               list_date, delist_date
        FROM checkmate_stock_status_history
        WHERE ts_code = ? AND as_of_date <= ?
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        [ts_code, as_of_date],
    ).fetchone()
    if row is None:
        return None
    cols = ("ts_code", "as_of_date", "list_status", "is_st", "name",
            "industry", "list_date", "delist_date")
    return dict(zip(cols, row))
