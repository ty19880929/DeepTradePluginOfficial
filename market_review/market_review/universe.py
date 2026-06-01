"""Universe construction — design §3.1.

Single entry point: :func:`build_universe(db, trade_date, ...)` reads the
already-synced ``mr_stock_basic`` / ``mr_stock_st`` / ``mr_suspend_d`` and
returns the per-day :class:`UniverseSnapshot`.

Rule summary (per design §3.1):

- ``market`` ∈ {主板, 创业板, 科创板, 北交所}; the北交所 inclusion can be
  toggled off via ``MrConfig.exclude_north_exchange`` (default keep).
- ST 剔除 — double-guard: name LIKE '*ST%' OR LIKE 'ST%', **AND**
  ``ts_code`` appearing in ``mr_stock_st`` for ``trade_date`` with non-empty
  ``st_status``.
- 退市整理 — name 含 '退', OR ``delist_date <= trade_date``.
- 停牌 — ``mr_suspend_d.suspend_type='S'`` on the given trade_date.

This module assumes :func:`market_review.data.sync_window` has already
populated the three exclusion tables for ``trade_date``. Callers that ask
for a date outside the synced range get whatever the DB has — an empty
result usually points at a missing :func:`sync_window` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


# Design §3.1 — the four exchanges market-review covers. Stored as a constant
# so tests and CLI ``settings show`` can introspect what's in scope.
INCLUDED_MARKETS: tuple[str, ...] = ("主板", "创业板", "科创板", "北交所")
NORTH_EXCHANGE_MARKET = "北交所"


@dataclass(frozen=True)
class UniverseSnapshot:
    """The investable universe at one trade_date + exclusion counters.

    The counters are informational — metrics modules consume ``ts_codes``;
    counters surface in events ("剔除 ST 132, 退市 6, 停牌 11") so the user
    can sanity-check that filters fired.
    """

    trade_date: str
    ts_codes: frozenset[str]
    n_total_before: int
    excluded_st: int
    excluded_delist: int
    excluded_suspend: int

    @property
    def n_total(self) -> int:
        return len(self.ts_codes)


def build_universe(
    db: Database,
    *,
    trade_date: str,
    exclude_north_exchange: bool = False,
) -> UniverseSnapshot:
    """Build the investable universe for ``trade_date`` from the synced DB.

    Returns an empty snapshot (``ts_codes = frozenset()``) when
    ``mr_stock_basic`` is empty — that's the caller's signal to run
    ``sync`` first.
    """
    markets = list(INCLUDED_MARKETS)
    if exclude_north_exchange:
        markets = [m for m in markets if m != NORTH_EXCHANGE_MARKET]

    market_ph = ",".join(["?"] * len(markets))

    n_total_before = _scalar(
        db,
        f"""
        SELECT COUNT(*) FROM mr_stock_basic
        WHERE market IN ({market_ph}) AND list_status = 'L'
        """,
        markets,
    )

    excluded_st = _scalar(
        db,
        f"""
        SELECT COUNT(DISTINCT sb.ts_code)
        FROM mr_stock_basic sb
        WHERE sb.market IN ({market_ph})
          AND sb.list_status = 'L'
          AND (
              sb.name LIKE '*ST%'
              OR sb.name LIKE 'ST%'
              OR sb.ts_code IN (
                  SELECT ts_code FROM mr_stock_st
                  WHERE trade_date = ? AND st_status IS NOT NULL AND st_status <> ''
              )
          )
        """,
        [*markets, trade_date],
    )

    excluded_delist = _scalar(
        db,
        f"""
        SELECT COUNT(*)
        FROM mr_stock_basic sb
        WHERE sb.market IN ({market_ph})
          AND sb.list_status = 'L'
          AND (
              sb.name LIKE '%退%'
              OR (sb.delist_date IS NOT NULL AND sb.delist_date <> ''
                  AND sb.delist_date <= ?)
          )
        """,
        [*markets, trade_date],
    )

    excluded_suspend = _scalar(
        db,
        """
        SELECT COUNT(DISTINCT ts_code)
        FROM mr_suspend_d
        WHERE trade_date = ? AND suspend_type = 'S'
        """,
        [trade_date],
    )

    rows = db.fetchall(
        f"""
        SELECT sb.ts_code
        FROM mr_stock_basic sb
        WHERE sb.market IN ({market_ph})
          AND sb.list_status = 'L'
          AND sb.name NOT LIKE '*ST%'
          AND sb.name NOT LIKE 'ST%'
          AND sb.name NOT LIKE '%退%'
          AND (
              sb.delist_date IS NULL OR sb.delist_date = '' OR sb.delist_date > ?
          )
          AND sb.ts_code NOT IN (
              SELECT ts_code FROM mr_stock_st
              WHERE trade_date = ? AND st_status IS NOT NULL AND st_status <> ''
          )
          AND sb.ts_code NOT IN (
              SELECT ts_code FROM mr_suspend_d
              WHERE trade_date = ? AND suspend_type = 'S'
          )
        ORDER BY sb.ts_code
        """,
        [*markets, trade_date, trade_date, trade_date],
    )
    survivors = frozenset(str(r[0]) for r in rows)

    return UniverseSnapshot(
        trade_date=trade_date,
        ts_codes=survivors,
        n_total_before=int(n_total_before),
        excluded_st=int(excluded_st),
        excluded_delist=int(excluded_delist),
        excluded_suspend=int(excluded_suspend),
    )


def _scalar(db: Database, sql: str, params: list[object]) -> int:
    row = db.fetchone(sql, params)
    if row is None:
        return 0
    return int(row[0] or 0)
