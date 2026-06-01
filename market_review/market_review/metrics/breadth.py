"""breadth — market-breadth aggregates (design §5.3.1 + Appendix A).

Public entry: :func:`compute_breadth(db, window, universes) -> BreadthReview`.
Pure read against the ``mr_*`` tables — no Tushare, no DB writes.

The single-day shape is :class:`BreadthSnapshot` (counts + ladder + index
returns). :class:`BreadthReview` wraps a per-day series plus the window
aggregates that ``risk_outlook`` / ``sentiment`` sections downstream consume
(median up-count, strongest / weakest day).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING

from ..data import WIDE_BASE_INDICES

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..universe import UniverseSnapshot
    from ..windows import Window


# Tushare's ``daily.amount`` field is reported in 千元 (thousand-yuan). One
# 亿 = 100,000 千元, so dividing by 1e5 converts to 亿.
_AMOUNT_THOUSANDS_PER_YI = 100_000.0

# Single-stock daily move thresholds (in percent points) used to bucket up /
# flat / down counts. The ``±0.01`` tolerance prevents a single tick at
# zero-change from being counted as a "down" day on rounding.
_FLAT_THRESHOLD_PCT = 0.01
# "5% 以上" cutoff for n_up5pct / n_down5pct (design §5.3.1).
_BIG_MOVE_PCT = 5.0


@dataclass(frozen=True)
class BreadthSnapshot:
    """Single-day snapshot — see design Appendix A."""

    trade_date: str
    n_total: int
    n_up: int
    n_down: int
    n_flat: int
    n_up5pct: int
    n_down5pct: int
    n_limit_up: int
    n_limit_down: int
    n_zhaban: int
    up_ladder: dict[int, int]
    n_lhb: int
    total_amount_yi: float
    index_returns: dict[str, float]


@dataclass(frozen=True)
class BreadthReview:
    """Window-level breadth result.

    ``series`` is ordered by ``trade_date`` ascending — sentiment / risk
    metrics walk it in calendar order.

    ``sentiment_extreme_day`` carries ``(strongest, weakest)`` based on the
    n_up - n_down spread. Ties resolve to the earlier date for determinism.
    """

    series: list[BreadthSnapshot] = field(default_factory=list)
    median_up_count: int = 0
    sentiment_extreme_day: tuple[str | None, str | None] = (None, None)


def compute_breadth(
    db: Database,
    window: Window,
    universes: dict[str, UniverseSnapshot],
    *,
    indices: tuple[str, ...] = WIDE_BASE_INDICES,
) -> BreadthReview:
    """Build the per-day snapshots + the window aggregate.

    ``universes`` is the per-day :class:`UniverseSnapshot` mapping returned
    by :func:`market_review.universe.build_window_universes`. A trade_date
    absent from ``universes`` is treated as empty universe (legitimate when
    upstream sync skipped a date — the snapshot for that day will have
    zero counts but still appear in ``series`` for time-axis continuity).
    """
    series: list[BreadthSnapshot] = []
    for trade_date in window.trade_dates:
        snap = universes.get(trade_date)
        ts_codes = snap.ts_codes if snap is not None else frozenset()
        series.append(_one_day(db, trade_date=trade_date, universe=ts_codes, indices=indices))

    up_counts = [s.n_up for s in series]
    median_up = int(median(up_counts)) if up_counts else 0

    strongest_day, weakest_day = _pick_extremes(series)
    return BreadthReview(
        series=series,
        median_up_count=median_up,
        sentiment_extreme_day=(strongest_day, weakest_day),
    )


# ---------------------------------------------------------------------------
# Per-day aggregation
# ---------------------------------------------------------------------------


def _one_day(
    db: Database,
    *,
    trade_date: str,
    universe: frozenset[str],
    indices: tuple[str, ...],
) -> BreadthSnapshot:
    in_clause, params = _bind_in(universe)
    if not universe:
        # Empty universe → all zeros; index returns still meaningful.
        return BreadthSnapshot(
            trade_date=trade_date,
            n_total=0, n_up=0, n_down=0, n_flat=0,
            n_up5pct=0, n_down5pct=0,
            n_limit_up=0, n_limit_down=0, n_zhaban=0,
            up_ladder={},
            n_lhb=0,
            total_amount_yi=0.0,
            index_returns=_index_returns(db, trade_date, indices),
        )

    counts_row = db.fetchone(
        f"""
        SELECT
            COUNT(*) AS n_total,
            SUM(CASE WHEN pct_chg > ? THEN 1 ELSE 0 END) AS n_up,
            SUM(CASE WHEN pct_chg < -? THEN 1 ELSE 0 END) AS n_down,
            SUM(CASE WHEN ABS(pct_chg) <= ? THEN 1 ELSE 0 END) AS n_flat,
            SUM(CASE WHEN pct_chg >= ? THEN 1 ELSE 0 END) AS n_up5,
            SUM(CASE WHEN pct_chg <= -? THEN 1 ELSE 0 END) AS n_down5,
            COALESCE(SUM(amount), 0) AS total_amt
        FROM mr_daily
        WHERE trade_date = ? AND ts_code IN {in_clause}
        """,
        [
            _FLAT_THRESHOLD_PCT,
            _FLAT_THRESHOLD_PCT,
            _FLAT_THRESHOLD_PCT,
            _BIG_MOVE_PCT,
            _BIG_MOVE_PCT,
            trade_date,
            *params,
        ],
    )
    if counts_row is None:
        n_total = n_up = n_down = n_flat = n_up5 = n_down5 = 0
        total_amt = 0.0
    else:
        n_total = int(counts_row[0] or 0)
        n_up = int(counts_row[1] or 0)
        n_down = int(counts_row[2] or 0)
        n_flat = int(counts_row[3] or 0)
        n_up5 = int(counts_row[4] or 0)
        n_down5 = int(counts_row[5] or 0)
        total_amt = float(counts_row[6] or 0.0)

    limit_counts = db.fetchall(
        f"""
        SELECT "limit", COUNT(DISTINCT ts_code)
        FROM mr_limit_list_d
        WHERE trade_date = ? AND ts_code IN {in_clause}
        GROUP BY "limit"
        """,
        [trade_date, *params],
    )
    limit_map = {str(row[0]): int(row[1] or 0) for row in limit_counts}

    ladder_rows = db.fetchall(
        f"""
        SELECT nums, COUNT(DISTINCT ts_code)
        FROM mr_limit_step
        WHERE trade_date = ? AND ts_code IN {in_clause} AND nums IS NOT NULL
        GROUP BY nums
        ORDER BY nums
        """,
        [trade_date, *params],
    )
    up_ladder: dict[int, int] = {
        int(row[0]): int(row[1] or 0) for row in ladder_rows if int(row[0]) >= 2
    }

    n_lhb_row = db.fetchone(
        f"""
        SELECT COUNT(DISTINCT ts_code) FROM mr_top_list
        WHERE trade_date = ? AND ts_code IN {in_clause}
        """,
        [trade_date, *params],
    )
    n_lhb = int(n_lhb_row[0] or 0) if n_lhb_row else 0

    return BreadthSnapshot(
        trade_date=trade_date,
        n_total=n_total,
        n_up=n_up,
        n_down=n_down,
        n_flat=n_flat,
        n_up5pct=n_up5,
        n_down5pct=n_down5,
        n_limit_up=limit_map.get("U", 0),
        n_limit_down=limit_map.get("D", 0),
        n_zhaban=limit_map.get("Z", 0),
        up_ladder=up_ladder,
        n_lhb=n_lhb,
        total_amount_yi=total_amt / _AMOUNT_THOUSANDS_PER_YI,
        index_returns=_index_returns(db, trade_date, indices),
    )


def _index_returns(
    db: Database, trade_date: str, indices: tuple[str, ...]
) -> dict[str, float]:
    if not indices:
        return {}
    in_clause = "(" + ",".join(["?"] * len(indices)) + ")"
    rows = db.fetchall(
        f"""
        SELECT ts_code, pct_chg FROM mr_index_daily
        WHERE trade_date = ? AND ts_code IN {in_clause}
        """,
        [trade_date, *indices],
    )
    return {str(row[0]): float(row[1] or 0.0) for row in rows}


def _pick_extremes(series: list[BreadthSnapshot]) -> tuple[str | None, str | None]:
    if not series:
        return (None, None)
    # n_up - n_down spread proxies a "赚钱效应" score; pick the strongest /
    # weakest day, breaking ties to the earlier date for stable output.
    sorted_series = sorted(
        series, key=lambda s: (s.n_up - s.n_down, -int(s.trade_date)), reverse=True
    )
    strongest = sorted_series[0].trade_date
    weakest = sorted_series[-1].trade_date
    return (strongest, weakest)


def _bind_in(ts_codes: frozenset[str]) -> tuple[str, list[str]]:
    """Render a deterministic ``IN (?,?,...)`` clause + matching params.

    The placeholder count tracks ``len(ts_codes)``; an empty universe
    short-circuits to ``IN (NULL)`` so the SQL stays syntactically valid
    while the row set is guaranteed empty.
    """
    if not ts_codes:
        return "(NULL)", []
    codes = sorted(ts_codes)
    return "(" + ",".join(["?"] * len(codes)) + ")", codes
