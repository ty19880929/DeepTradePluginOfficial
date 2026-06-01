"""sentiment — 市场情绪与赚钱效应 (design §5.3.2 + Appendix A).

Two products:

- :class:`SentimentSnapshot` — per-day metrics (median pct_chg, ratios,
  the 0~100 sentiment thermometer).
- :class:`SentimentReview` — window aggregate (series + average score +
  strongest / weakest day labels).

The 0~100 thermometer follows design §5.3.2:

    sentiment = 0.30 * pos_ratio_norm
              + 0.20 * top_ratio_norm
              + 0.15 * limit_up_intensity_norm
              + 0.10 * connection_health_norm
              + 0.10 * (1 - crash_ratio_norm)
              + 0.10 * lhb_buy_intensity_norm
              + 0.05 * north_inflow_norm

The design proposes a 60-day robust z-score (median + MAD) per term. PR-3
ships a simpler in-window normalization (linear clamp to 0~100 with
plausible reference ranges) so the metric is self-contained — it doesn't
need a 60-day lookback synced. PR-4+ can swap in the historical
normalization once a "stats history" table exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING

from .breadth import BreadthReview, BreadthSnapshot, _bind_in

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..universe import UniverseSnapshot
    from ..windows import Window


# --- Default weight vector (design §5.3.2 — must sum to 1.0) ----------------
DEFAULT_SENTIMENT_WEIGHTS: dict[str, float] = {
    "pos_ratio": 0.30,
    "top_ratio": 0.20,
    "limit_up_intensity": 0.15,
    "connection_health": 0.10,
    "crash_ratio_inv": 0.10,
    "lhb_buy_intensity": 0.10,
    "north_inflow": 0.05,
}

# Reference ceiling for ``top_ratio`` (% of stocks > +5% on the day). A
# broad rally typically tops out near 20%; we clamp at 25% to give the
# normalizer some headroom but still hit 100 on truly extraordinary days.
_TOP_RATIO_CEILING = 0.25
_CRASH_RATIO_CEILING = 0.25

# Limit-up intensity normalizes ``n_limit_up / n_total``. Even on hottest
# days that ratio stays under ~3% (~150 / 5000); we use 3% as the 100-point
# ceiling so a hot day scores in the 80-100 band.
_LIMIT_UP_INTENSITY_CEILING = 0.03

# LHB buy intensity normalizes the count of dragon-tiger stocks per day
# against a 200-stock ceiling (very hot day's order of magnitude).
_LHB_INTENSITY_CEILING = 200.0

# North inflow normalization band: -50亿 floors to 0; +50亿 caps to 100.
# Within band we linearly interpolate.
_NORTH_INFLOW_FLOOR_YI = -50.0
_NORTH_INFLOW_CEIL_YI = 50.0


@dataclass(frozen=True)
class SentimentSnapshot:
    trade_date: str
    median_pct_chg: float
    mean_pct_chg: float
    pos_ratio: float            # n_up / max(1, n_total)
    top_ratio: float            # n_up5pct / max(1, n_total)
    crash_ratio: float          # n_down5pct / max(1, n_total)
    limit_up_intensity: float   # n_limit_up / max(1, n_total)
    connection_health: float    # 0~1; ladder contiguity heuristic
    n_lhb: int
    north_money_yi: float | None  # None when mr_moneyflow_hsgt has no row
    score_0_100: float


@dataclass(frozen=True)
class SentimentReview:
    series: list[SentimentSnapshot] = field(default_factory=list)
    avg_score: float = 0.0
    strongest_day: str | None = None
    weakest_day: str | None = None


def compute_sentiment(
    db: Database,
    window: Window,
    universes: dict[str, UniverseSnapshot],
    breadth: BreadthReview,
    *,
    weights: dict[str, float] = DEFAULT_SENTIMENT_WEIGHTS,
) -> SentimentReview:
    """Per-day sentiment thermometer over the window.

    ``breadth`` is the already-computed :class:`BreadthReview` — sentiment
    derives all its counts from there to avoid re-querying mr_daily /
    mr_limit_list_d. ``weights`` defaults to the §5.3.2 reference vector;
    pass ``MrConfig.sentiment_weights`` to honor user overrides.

    The function tolerates partial inputs (e.g., missing ``mr_moneyflow_hsgt``
    for a given day) — the corresponding term in the blend gets a zero
    contribution rather than raising.
    """
    if not breadth.series:
        return SentimentReview()

    _validate_weights(weights)

    # Pre-fetch per-day median / mean pct_chg + north flow in one pass each.
    daily_stats = _fetch_daily_pct_stats(db, window.trade_dates, universes)
    north_flow = _fetch_north_flow_yi(db, window.trade_dates)

    snapshots: list[SentimentSnapshot] = []
    for snap in breadth.series:
        td = snap.trade_date
        med_pct, mean_pct = daily_stats.get(td, (0.0, 0.0))
        pos_ratio = snap.n_up / max(1, snap.n_total)
        top_ratio = snap.n_up5pct / max(1, snap.n_total)
        crash_ratio = snap.n_down5pct / max(1, snap.n_total)
        limit_up_intensity = snap.n_limit_up / max(1, snap.n_total)
        connection_health = _connection_health(snap)
        north_yi = north_flow.get(td)

        score = _blend(
            weights=weights,
            pos_ratio=pos_ratio,
            top_ratio=top_ratio,
            limit_up_intensity=limit_up_intensity,
            connection_health=connection_health,
            crash_ratio=crash_ratio,
            n_lhb=snap.n_lhb,
            north_yi=north_yi,
        )
        snapshots.append(SentimentSnapshot(
            trade_date=td,
            median_pct_chg=med_pct,
            mean_pct_chg=mean_pct,
            pos_ratio=pos_ratio,
            top_ratio=top_ratio,
            crash_ratio=crash_ratio,
            limit_up_intensity=limit_up_intensity,
            connection_health=connection_health,
            n_lhb=snap.n_lhb,
            north_money_yi=north_yi,
            score_0_100=score,
        ))

    avg = sum(s.score_0_100 for s in snapshots) / len(snapshots) if snapshots else 0.0
    sorted_by_score = sorted(
        snapshots, key=lambda s: (s.score_0_100, -int(s.trade_date)), reverse=True
    )
    strongest = sorted_by_score[0].trade_date if sorted_by_score else None
    weakest = sorted_by_score[-1].trade_date if sorted_by_score else None
    return SentimentReview(
        series=snapshots,
        avg_score=avg,
        strongest_day=strongest,
        weakest_day=weakest,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_weights(weights: dict[str, float]) -> None:
    expected = set(DEFAULT_SENTIMENT_WEIGHTS)
    missing = expected - set(weights)
    if missing:
        raise ValueError(f"sentiment_weights missing keys: {sorted(missing)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"sentiment_weights must sum to 1.0, got {total:.6f}")


def _fetch_daily_pct_stats(
    db: Database,
    trade_dates: tuple[str, ...],
    universes: dict[str, UniverseSnapshot],
) -> dict[str, tuple[float, float]]:
    """Per-day (median, mean) of ``pct_chg`` restricted to that day's universe.

    Computed in Python rather than via SQL because (a) DuckDB's MEDIAN is
    a standard SQL agg, but (b) per-day MEDIAN over different universes
    needs to be sliced per-date anyway, so the loop cost is the same.
    """
    out: dict[str, tuple[float, float]] = {}
    for td in trade_dates:
        snap = universes.get(td)
        if snap is None or not snap.ts_codes:
            out[td] = (0.0, 0.0)
            continue
        in_clause, params = _bind_in(snap.ts_codes)
        rows = db.fetchall(
            f"""
            SELECT pct_chg FROM mr_daily
            WHERE trade_date = ? AND ts_code IN {in_clause} AND pct_chg IS NOT NULL
            """,
            [td, *params],
        )
        values = [float(row[0]) for row in rows]
        if not values:
            out[td] = (0.0, 0.0)
            continue
        out[td] = (median(values), sum(values) / len(values))
    return out


def _fetch_north_flow_yi(
    db: Database, trade_dates: tuple[str, ...]
) -> dict[str, float | None]:
    """Per-day northbound net inflow in 亿. ``mr_moneyflow_hsgt.north_money``
    is already reported in 万元 by Tushare — convert to 亿 via /1e4."""
    if not trade_dates:
        return {}
    in_clause = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    rows = db.fetchall(
        f"""
        SELECT trade_date, north_money FROM mr_moneyflow_hsgt
        WHERE trade_date IN {in_clause}
        """,
        list(trade_dates),
    )
    out: dict[str, float | None] = {td: None for td in trade_dates}
    for row in rows:
        td = str(row[0])
        north = row[1]
        out[td] = (float(north) / 10_000.0) if north is not None else None
    return out


def _connection_health(snap: BreadthSnapshot) -> float:
    """Ladder contiguity heuristic — design §5.3.2 connection_health.

    Returns 1.0 when the ladder counts (2 连板, 3 连板, ...) form a
    contiguous chain up to the max observed height; 0.0 when only the max
    is occupied with all middle heights empty; linearly interpolated in
    between. An empty ladder returns 0.0.
    """
    if not snap.up_ladder:
        return 0.0
    heights = sorted(snap.up_ladder)
    max_height = heights[-1]
    if max_height < 2:
        return 0.0
    expected = list(range(2, max_height + 1))
    occupied = sum(1 for h in expected if snap.up_ladder.get(h, 0) > 0)
    return occupied / len(expected)


def _blend(
    *,
    weights: dict[str, float],
    pos_ratio: float,
    top_ratio: float,
    limit_up_intensity: float,
    connection_health: float,
    crash_ratio: float,
    n_lhb: int,
    north_yi: float | None,
) -> float:
    """Apply §5.3.2's weighted sum after normalizing each input to 0~100."""
    pos_norm = _clamp01(pos_ratio) * 100.0
    top_norm = _clamp01(top_ratio / _TOP_RATIO_CEILING) * 100.0
    lu_norm = _clamp01(limit_up_intensity / _LIMIT_UP_INTENSITY_CEILING) * 100.0
    conn_norm = _clamp01(connection_health) * 100.0
    crash_inv_norm = (1.0 - _clamp01(crash_ratio / _CRASH_RATIO_CEILING)) * 100.0
    lhb_norm = _clamp01(n_lhb / _LHB_INTENSITY_CEILING) * 100.0
    if north_yi is None:
        north_norm = 50.0  # neutral when data is missing
    else:
        north_norm = _linear_map(
            north_yi, _NORTH_INFLOW_FLOOR_YI, _NORTH_INFLOW_CEIL_YI
        ) * 100.0

    score = (
        weights["pos_ratio"] * pos_norm
        + weights["top_ratio"] * top_norm
        + weights["limit_up_intensity"] * lu_norm
        + weights["connection_health"] * conn_norm
        + weights["crash_ratio_inv"] * crash_inv_norm
        + weights["lhb_buy_intensity"] * lhb_norm
        + weights["north_inflow"] * north_norm
    )
    return max(0.0, min(100.0, score))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _linear_map(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return _clamp01((value - lo) / (hi - lo))
