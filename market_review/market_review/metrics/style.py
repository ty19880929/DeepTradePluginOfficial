"""style — 风格切换 (design §5.3.6 + Appendix A).

Compares 4 风格轴 across the window:

- **大小盘** — 沪深 300 (000300.SH) vs 中证 1000 (000852.SH).
  The design names 中证 2000 as the small-cap leg; 2000 isn't part of our
  8000-credit basket so we substitute 1000 (the next-tier small-cap proxy
  the Tushare standard tier serves reliably).
- **价值 / 成长** — 沪深 300 vs 创业板指 (399006.SZ) proxy. Design §15.4
  notes this is a coarse stand-in; a true value/growth split needs
  ``index_weight`` of CSI 800 价值 / 成长 sub-indices, which PR-3 doesn't
  fetch (see PR-2 deferred list).
- **大盘量价** — sum of large-cap index amounts vs small-cap amounts.
  PR-3 reads ``mr_index_daily.amount`` (Tushare reports in 千元 → /1e5 for 亿).

Outputs:

- :class:`StyleSeriesPoint` — per trade_date (large/small/growth/value rets +
  cumulative ratio).
- :class:`StyleReview` — dominant_style + flip_signal + range_summary.

``flip_signal`` is ``True`` when the first-half vs second-half
``big_to_small_ratio`` flips sign. Single-day windows always return
``False`` (no halves to compare).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..windows import Window


LARGE_CAP_INDEX = "000300.SH"
SMALL_CAP_INDEX = "000852.SH"
GROWTH_INDEX = "399006.SZ"


DominantStyle = Literal["large_cap", "small_cap", "growth", "value", "balanced", "rotating"]


@dataclass(frozen=True)
class StyleSeriesPoint:
    trade_date: str
    large_cap_ret: float       # 当日 pct_chg
    small_cap_ret: float
    growth_ret: float | None
    value_ret: float | None
    big_to_small_ratio: float  # 累计大/小盘相对强度


@dataclass(frozen=True)
class StyleReview:
    dominant_style: DominantStyle = "balanced"
    flip_signal: bool = False
    series: list[StyleSeriesPoint] = field(default_factory=list)
    range_summary: dict[str, float] = field(default_factory=dict)


def compute_style(db: Database, window: Window) -> StyleReview:
    """Build the风格 series + dominant-style label."""
    trade_dates = list(window.trade_dates)
    if not trade_dates:
        return StyleReview()

    in_clause = "(" + ",".join(["?"] * len(trade_dates)) + ")"
    rows = db.fetchall(
        f"""
        SELECT ts_code, trade_date, COALESCE(pct_chg, 0)
        FROM mr_index_daily
        WHERE trade_date IN {in_clause}
          AND ts_code IN (?, ?, ?)
        ORDER BY trade_date
        """,
        [*trade_dates, LARGE_CAP_INDEX, SMALL_CAP_INDEX, GROWTH_INDEX],
    )
    by_code_date: dict[tuple[str, str], float] = {}
    for code, td, pct in rows:
        by_code_date[(str(code), str(td))] = float(pct or 0.0)

    series: list[StyleSeriesPoint] = []
    big_cum = 1.0
    small_cum = 1.0
    for td in trade_dates:
        large = by_code_date.get((LARGE_CAP_INDEX, td), 0.0)
        small = by_code_date.get((SMALL_CAP_INDEX, td), 0.0)
        growth = by_code_date.get((GROWTH_INDEX, td))
        big_cum *= 1.0 + large / 100.0
        small_cum *= 1.0 + small / 100.0
        ratio = (big_cum - small_cum) * 100.0  # cumulative spread in %
        series.append(StyleSeriesPoint(
            trade_date=td,
            large_cap_ret=large,
            small_cap_ret=small,
            growth_ret=growth if growth is not None else None,
            value_ret=large if growth is not None else None,  # 价值 ≈ large in proxy
            big_to_small_ratio=ratio,
        ))

    big_cum_pct = (big_cum - 1.0) * 100.0
    small_cum_pct = (small_cum - 1.0) * 100.0
    spread = big_cum_pct - small_cum_pct

    dominant = _classify_dominant(big_cum_pct, small_cum_pct)
    flip_signal = _detect_flip(series)
    avg_ratio = mean(s.big_to_small_ratio for s in series) if series else 0.0

    return StyleReview(
        dominant_style=dominant,
        flip_signal=flip_signal,
        series=series,
        range_summary={
            "large_cap_cum_pct": round(big_cum_pct, 4),
            "small_cap_cum_pct": round(small_cum_pct, 4),
            "spread_pct": round(spread, 4),
            "avg_big_to_small_ratio": round(avg_ratio, 4),
        },
    )


def _classify_dominant(big_cum_pct: float, small_cum_pct: float) -> DominantStyle:
    """Coarse classifier — the spread threshold (≥2pp) is a v0.1 heuristic.

    A tighter, multi-axis classifier (incl. growth/value) needs index_weight
    data; PR-6 can tighten this with real run data and validation samples.
    """
    spread = big_cum_pct - small_cum_pct
    if spread >= 2.0:
        return "large_cap"
    if spread <= -2.0:
        return "small_cap"
    if abs(spread) < 1.0:
        return "balanced"
    return "rotating"


def _detect_flip(series: list[StyleSeriesPoint]) -> bool:
    if len(series) < 2:
        return False
    half = len(series) // 2
    if half == 0:
        return False
    earlier_avg = mean(s.big_to_small_ratio for s in series[:half])
    later_avg = mean(s.big_to_small_ratio for s in series[half:])
    # A flip requires opposite signs (one cumulative spread positive, the
    # other negative) by at least 1pp magnitude so a noise crossing near 0
    # doesn't trigger.
    if (earlier_avg > 1.0 and later_avg < -1.0) or (earlier_avg < -1.0 and later_avg > 1.0):
        return True
    return False
