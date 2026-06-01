"""leaders — 龙头识别 (design §5.3.5 + Appendix A).

Four-dimensional scoring per candidate, each axis worth 0–25:

1. **梯队** (ladder): ``25 * log₂(n+1) / log₂(7)`` clamped, where ``n`` is the
   max ``mr_limit_step.nums`` observed for the stock in the window.
2. **涨幅** (range cumulative return): geometric chain of ``mr_daily.pct_chg``
   across the window → percentile × 25.
3. **资金** (capital): window-total ``mr_moneyflow.net_mf_amount`` →
   percentile × 25. v0.1 PR-3 omits the northbound add-on (the hsgt
   per-stock data isn't in our scope); PR-4 can layer it on.
4. **题材** (theme): if the stock's ``mr_stock_basic.industry`` matches any
   sector in :class:`SectorReview.today_top` → 25; in top 20 → 15; else 0.
   PR-4 will refine this via the framework's ``ConceptRepository``.

Total ∈ [0, 100]. Candidate pool = stocks at ladder ≥ 2 OR in the top-50
by range cumulative return; the union keeps the score search space
focused on the genuine高位 / 强势 names rather than the whole universe.

Top-K primary (default 5) + Top-K secondary (default 10) returned alongside
the cutoff ``min_score`` and a sector → ts_codes mapping for downstream
sector-narrative anchoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .breadth import _bind_in
from .sectors import SectorReview

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..universe import UniverseSnapshot
    from ..windows import Window


_LADDER_NORM_MAX = 7        # n=7 connecting board normalizes to ~25
_DEFAULT_PRIMARY_K = 5
_DEFAULT_SECONDARY_K = 10
_DEFAULT_MIN_SCORE = 50.0
_RANGE_TOP_POOL = 50        # pool size from the "by-return" pre-filter
_WAN_PER_YI = 10_000.0


@dataclass(frozen=True)
class LeaderCandidate:
    ts_code: str
    name: str
    score: float
    score_breakdown: dict[str, float]
    industries: list[str]
    concepts: list[str]
    sector_top_hit: list[str]
    ladder_height: int | None
    range_pct_chg: float | None
    cum_main_inflow_yi: float | None


@dataclass(frozen=True)
class LeaderReview:
    primary: list[LeaderCandidate] = field(default_factory=list)
    secondary: list[LeaderCandidate] = field(default_factory=list)
    min_score: float = _DEFAULT_MIN_SCORE
    sector_map: dict[str, list[str]] = field(default_factory=dict)


def compute_leaders(
    db: Database,
    window: Window,
    universes: dict[str, UniverseSnapshot],
    *,
    sector_review: SectorReview | None = None,
    primary_k: int = _DEFAULT_PRIMARY_K,
    secondary_k: int = _DEFAULT_SECONDARY_K,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> LeaderReview:
    """Score the window's leader candidates and pick primary + secondary."""
    trade_dates = list(window.trade_dates)
    if not trade_dates:
        return LeaderReview(min_score=min_score)

    union = frozenset().union(*[u.ts_codes for u in universes.values()]) if universes else frozenset()
    if not union:
        return LeaderReview(min_score=min_score)

    ladder = _ladder_per_stock(db, trade_dates=trade_dates, universe=union)
    range_ret = _range_return_per_stock(db, trade_dates=trade_dates, universe=union)
    cum_inflow = _cum_inflow_per_stock(db, trade_dates=trade_dates, universe=union)

    pool = _candidate_pool(ladder, range_ret)
    if not pool:
        return LeaderReview(min_score=min_score)

    names = _stock_names(db, pool)
    industries = _stock_industries(db, pool)
    theme_lookup = _theme_lookup(sector_review)

    # Compute percentile-based normalization once.
    pool_returns = [range_ret.get(c, 0.0) for c in pool]
    pool_inflows = [cum_inflow.get(c, 0.0) for c in pool]
    ret_rank = _rank_dict(pool, pool_returns)
    inflow_rank = _rank_dict(pool, pool_inflows)
    n_pool = max(1, len(pool))

    candidates: list[LeaderCandidate] = []
    for code in pool:
        ladder_n = ladder.get(code, 0)
        ladder_score = _score_ladder(ladder_n)
        ret_score = (ret_rank.get(code, 0) / n_pool) * 25.0
        capital_score = (inflow_rank.get(code, 0) / n_pool) * 25.0
        theme_score, sector_hit = _score_theme(industries.get(code, ""), theme_lookup)
        score = ladder_score + ret_score + capital_score + theme_score
        candidates.append(LeaderCandidate(
            ts_code=code,
            name=names.get(code, code),
            score=round(score, 2),
            score_breakdown={
                "ladder": round(ladder_score, 2),
                "return": round(ret_score, 2),
                "capital": round(capital_score, 2),
                "theme": round(theme_score, 2),
            },
            industries=[industries[code]] if code in industries and industries[code] else [],
            concepts=[],  # PR-4: ConceptRepository integration
            sector_top_hit=list(sector_hit),
            ladder_height=ladder_n if ladder_n > 0 else None,
            range_pct_chg=range_ret.get(code),
            cum_main_inflow_yi=cum_inflow.get(code) / _WAN_PER_YI if code in cum_inflow else None,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    above_cut = [c for c in candidates if c.score >= min_score]
    primary = above_cut[:primary_k]
    secondary = above_cut[primary_k : primary_k + secondary_k]

    # Sector → ts_codes mapping for downstream narrative.
    sector_map: dict[str, list[str]] = {}
    for cand in primary + secondary:
        for sector in cand.sector_top_hit:
            sector_map.setdefault(sector, []).append(cand.ts_code)

    return LeaderReview(
        primary=primary,
        secondary=secondary,
        min_score=min_score,
        sector_map=sector_map,
    )


# ---------------------------------------------------------------------------
# Data extractors
# ---------------------------------------------------------------------------


def _ladder_per_stock(
    db: Database, *, trade_dates: list[str], universe: frozenset[str]
) -> dict[str, int]:
    if not universe:
        return {}
    in_dates = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    in_codes, code_params = _bind_in(universe)
    rows = db.fetchall(
        f"""
        SELECT ts_code, MAX(COALESCE(nums, 0)) AS max_nums
        FROM mr_limit_step
        WHERE trade_date IN {in_dates} AND ts_code IN {in_codes}
        GROUP BY ts_code
        """,
        [*trade_dates, *code_params],
    )
    return {str(r[0]): int(r[1] or 0) for r in rows}


def _range_return_per_stock(
    db: Database, *, trade_dates: list[str], universe: frozenset[str]
) -> dict[str, float]:
    """Geometric chain of daily ``pct_chg``; missing days treated as 0."""
    if not universe or not trade_dates:
        return {}
    in_dates = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    in_codes, code_params = _bind_in(universe)
    rows = db.fetchall(
        f"""
        SELECT ts_code, trade_date, COALESCE(pct_chg, 0)
        FROM mr_daily
        WHERE trade_date IN {in_dates} AND ts_code IN {in_codes}
        ORDER BY ts_code, trade_date
        """,
        [*trade_dates, *code_params],
    )
    per: dict[str, float] = {}
    for code_raw, _td, pct in rows:
        code = str(code_raw)
        per.setdefault(code, 1.0)
        per[code] *= 1.0 + float(pct or 0.0) / 100.0
    return {c: (factor - 1.0) * 100.0 for c, factor in per.items()}


def _cum_inflow_per_stock(
    db: Database, *, trade_dates: list[str], universe: frozenset[str]
) -> dict[str, float]:
    if not universe or not trade_dates:
        return {}
    in_dates = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    in_codes, code_params = _bind_in(universe)
    rows = db.fetchall(
        f"""
        SELECT ts_code, SUM(COALESCE(net_mf_amount, 0)) AS net_total
        FROM mr_moneyflow
        WHERE trade_date IN {in_dates} AND ts_code IN {in_codes}
        GROUP BY ts_code
        """,
        [*trade_dates, *code_params],
    )
    return {str(r[0]): float(r[1] or 0.0) for r in rows}


def _stock_names(db: Database, ts_codes: list[str]) -> dict[str, str]:
    if not ts_codes:
        return {}
    in_clause = "(" + ",".join(["?"] * len(ts_codes)) + ")"
    rows = db.fetchall(
        f"SELECT ts_code, name FROM mr_stock_basic WHERE ts_code IN {in_clause}",
        list(ts_codes),
    )
    return {str(r[0]): str(r[1] or r[0]) for r in rows}


def _stock_industries(db: Database, ts_codes: list[str]) -> dict[str, str]:
    if not ts_codes:
        return {}
    in_clause = "(" + ",".join(["?"] * len(ts_codes)) + ")"
    rows = db.fetchall(
        f"SELECT ts_code, industry FROM mr_stock_basic WHERE ts_code IN {in_clause}",
        list(ts_codes),
    )
    return {str(r[0]): str(r[1] or "") for r in rows}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _candidate_pool(
    ladder: dict[str, int], range_ret: dict[str, float]
) -> list[str]:
    """Union of: 连板 ≥ 2 stocks + top-50 by range return."""
    pool: set[str] = {code for code, n in ladder.items() if n >= 2}
    by_return = sorted(range_ret.items(), key=lambda x: x[1], reverse=True)
    pool.update(code for code, _ in by_return[:_RANGE_TOP_POOL])
    return sorted(pool)


def _score_ladder(n: int) -> float:
    if n <= 1:
        return 0.0
    raw = 25.0 * math.log2(n + 1) / math.log2(_LADDER_NORM_MAX + 1)
    return min(25.0, max(0.0, raw))


def _theme_lookup(sector_review: SectorReview | None) -> tuple[set[str], set[str]]:
    """Return (top10_names, top20_names) for the theme score lookup."""
    if sector_review is None:
        return set(), set()
    top10 = {e.name for e in sector_review.today_top[:10]} | {
        e.name for e in sector_review.range_top[:10]
    }
    top20 = {e.name for e in sector_review.today_top[:20]} | {
        e.name for e in sector_review.range_top[:20]
    }
    return top10, top20


def _score_theme(industry: str, lookup: tuple[set[str], set[str]]) -> tuple[float, list[str]]:
    if not industry:
        return 0.0, []
    top10, top20 = lookup
    if industry in top10:
        return 25.0, [industry]
    if industry in top20:
        return 15.0, [industry]
    return 0.0, []


def _rank_dict(pool: list[str], values: list[float]) -> dict[str, int]:
    """Return a 1-based rank (highest value → rank ``len(pool)``) for each code."""
    paired = sorted(zip(pool, values), key=lambda x: x[1])
    return {code: rank + 1 for rank, (code, _v) in enumerate(paired)}
