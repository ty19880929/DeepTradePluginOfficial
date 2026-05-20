"""Trade-calendar helper for Checkmate.

Wraps Tushare's ``trade_cal`` frame in a read-only :class:`TradeCalendar`
object with the three primitives the rest of the pipeline needs:

* :meth:`is_trade_day(date)`
* :meth:`prev_session(date)`
* :meth:`next_session(date)`

Plus a small set of range helpers (``sessions_in_range``, ``n_sessions_before``)
that are convenient for windowed feature calculation and backtest stepping.

A lightweight parquet cache at ``cache/trade_cal.parquet`` (see
:mod:`checkmate.paths`) lets ``sync`` / ``scan`` reuse a single calendar fetch
across CLI invocations. The "DB persistence" described in iteration_tasks.md
§2 PR-1.1 is realised here as a parquet on disk — the 10 frozen tables from
PR-0.2 intentionally do not include a trade-calendar table; parquet is the
equivalent persistent-storage backend without requiring a migration bump.

The class itself is I/O-free: tests can construct one from a synthetic frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import paths

# Tushare exchange code for the calendar probe. SSE = Shanghai; SZSE shares
# the same holiday schedule for A-shares so a single SSE pull is sufficient.
_DEFAULT_EXCHANGE = "SSE"


def _normalize(date: str) -> str:
    """Accept either ``YYYYMMDD`` or ``YYYY-MM-DD`` and return ``YYYYMMDD``."""
    d = str(date).strip()
    if len(d) == 10 and d[4] == "-" and d[7] == "-":
        return d.replace("-", "")
    if len(d) == 8 and d.isdigit():
        return d
    raise ValueError(f"unrecognised date format: {date!r}")


class TradeCalendar:
    """Read-only view over a trade_cal DataFrame.

    Required columns: ``cal_date`` (str YYYYMMDD), ``is_open`` (0/1). Other
    columns are preserved but unused. Construction normalises the frame so
    repeated lookups are O(1) on the date index.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not {"cal_date", "is_open"}.issubset(df.columns):
            raise ValueError("trade_cal frame missing required columns cal_date / is_open")
        norm = df.copy()
        norm["cal_date"] = norm["cal_date"].astype(str)
        norm["is_open"] = pd.to_numeric(norm["is_open"], errors="coerce").fillna(0).astype(int)
        norm = norm.sort_values("cal_date").drop_duplicates("cal_date", keep="last")
        self._df = norm.reset_index(drop=True)
        # cal_date → row index, for O(1) is_trade_day lookups.
        self._idx: dict[str, int] = {
            str(row.cal_date): i for i, row in enumerate(self._df.itertuples(index=False))
        }
        self._open_dates: list[str] = self._df.loc[self._df["is_open"] == 1, "cal_date"].astype(str).tolist()

    # ------------------------------------------------------------------ basic
    def is_trade_day(self, date: str) -> bool:
        d = _normalize(date)
        idx = self._idx.get(d)
        if idx is None:
            return False
        return int(self._df.at[idx, "is_open"]) == 1

    def next_session(self, date: str) -> str:
        """First open trading day strictly after ``date``. Raises if none."""
        d = _normalize(date)
        for od in self._open_dates:
            if od > d:
                return od
        raise ValueError(f"no future open trading day after {d}")

    def prev_session(self, date: str) -> str:
        """Most recent open trading day strictly before ``date``. Raises if none."""
        d = _normalize(date)
        last: str | None = None
        for od in self._open_dates:
            if od < d:
                last = od
            else:
                break
        if last is None:
            raise ValueError(f"no prior open trading day before {d}")
        return last

    # ------------------------------------------------------------------ range
    def sessions_in_range(self, start: str, end: str) -> list[str]:
        """Sorted YYYYMMDD trade dates in ``[start, end]`` inclusive."""
        s = _normalize(start)
        e = _normalize(end)
        if s > e:
            return []
        return [d for d in self._open_dates if s <= d <= e]

    def n_sessions_before(self, date: str, n: int) -> str:
        """Return the n-th open trading day strictly before ``date`` (n>=1)."""
        if n < 1:
            raise ValueError("n must be >= 1")
        d = _normalize(date)
        before = [od for od in self._open_dates if od < d]
        if len(before) < n:
            raise ValueError(f"only {len(before)} open days before {d}, need {n}")
        return before[-n]

    # --------------------------------------------------------------- internal
    @property
    def frame(self) -> pd.DataFrame:
        """Expose the normalised frame for callers that need it (read-only)."""
        return self._df


# ---------------------------------------------------------------------------
# Loader — wraps Tushare + parquet cache
# ---------------------------------------------------------------------------


def _read_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.is_file():
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:  # noqa: BLE001
        # Corrupt cache → fall back to refetch; never block the pipeline.
        return None


def _write_cache(cache_path: Path, df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def load_trade_calendar(
    tushare: Any,
    *,
    start: str = "20100101",
    end: str | None = None,
    refresh: bool = False,
    cache_path: Path | None = None,
    exchange: str = _DEFAULT_EXCHANGE,
) -> TradeCalendar:
    """Load (or fetch + cache) the trade calendar.

    Parameters
    ----------
    tushare : Any
        Object with ``.call(api_name, **kwargs) -> DataFrame``. The framework's
        :class:`TushareClient` satisfies this; tests pass a stub.
    start : str
        Earliest ``cal_date`` to request from Tushare on a refresh (YYYYMMDD).
    end : str | None
        Latest ``cal_date`` to request. ``None`` → today (UTC), widened by one
        day so a post-close run still pulls today's row.
    refresh : bool
        When ``True``, bypass the parquet cache and re-fetch from Tushare.
    cache_path : Path | None
        Override for the on-disk cache file. Defaults to
        :func:`checkmate.paths.trade_cal_cache_path`.
    exchange : str
        Tushare exchange code (default ``SSE``; SZSE shares the schedule).
    """
    path = cache_path or paths.trade_cal_cache_path()
    if not refresh:
        cached = _read_cache(path)
        if cached is not None and not cached.empty:
            return TradeCalendar(cached)

    if end is None:
        # Use a wide-open window: the framework's TushareClient supports
        # ``end_date`` in the future; passing 21001231 keeps the call
        # idempotent across calendar years without re-deriving "today".
        end = "21001231"
    df = tushare.call("trade_cal", exchange=exchange, start_date=start, end_date=end)
    if df is None or df.empty:
        raise RuntimeError(
            f"trade_cal({exchange}) returned no rows for {start}..{end}; "
            "check Tushare token / network."
        )
    _write_cache(path, df)
    return TradeCalendar(df)
