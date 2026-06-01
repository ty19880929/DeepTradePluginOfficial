"""style — dominant style classifier + flip detection."""

from __future__ import annotations

from deeptrade.core.db import Database

from market_review.metrics.style import (
    GROWTH_INDEX,
    LARGE_CAP_INDEX,
    SMALL_CAP_INDEX,
    StyleReview,
    compute_style,
)
from market_review.windows import Window


def _idx(db, ts_code, trade_date, pct_chg):
    db.execute(
        "INSERT INTO mr_index_daily (ts_code, trade_date, pct_chg) VALUES (?, ?, ?)",
        [ts_code, trade_date, pct_chg],
    )


def _w(trade_dates):
    return Window(
        mode="range" if len(trade_dates) > 1 else "day",
        start=trade_dates[0], end=trade_dates[-1],
        trade_dates=trade_dates, anchor=trade_dates[-1],
    )


def test_empty_window_returns_default(mr_db: Database) -> None:
    # Build via a trivial trade_date but no index data — series stays empty? No, series is built per trade_date even with 0% returns.
    review = compute_style(mr_db, _w(("20260530",)))
    assert isinstance(review, StyleReview)
    assert len(review.series) == 1
    assert review.dominant_style == "balanced"


def test_large_cap_dominant_when_300_outperforms(mr_db: Database) -> None:
    _idx(mr_db, LARGE_CAP_INDEX, "20260530", 3.0)
    _idx(mr_db, SMALL_CAP_INDEX, "20260530", 0.5)
    review = compute_style(mr_db, _w(("20260530",)))
    assert review.dominant_style == "large_cap"
    # range_summary has the right keys
    assert "spread_pct" in review.range_summary
    assert review.range_summary["spread_pct"] > 0


def test_small_cap_dominant_when_1000_outperforms(mr_db: Database) -> None:
    _idx(mr_db, LARGE_CAP_INDEX, "20260530", 0.5)
    _idx(mr_db, SMALL_CAP_INDEX, "20260530", 4.0)
    review = compute_style(mr_db, _w(("20260530",)))
    assert review.dominant_style == "small_cap"


def test_balanced_when_spreads_tight(mr_db: Database) -> None:
    _idx(mr_db, LARGE_CAP_INDEX, "20260530", 0.5)
    _idx(mr_db, SMALL_CAP_INDEX, "20260530", 0.6)
    review = compute_style(mr_db, _w(("20260530",)))
    assert review.dominant_style == "balanced"


def test_flip_signal_when_sign_changes_across_halves(mr_db: Database) -> None:
    """Earlier half large-cap dominant, later half small-cap dominant → flip."""
    # Days 1-2: 大盘强（300 ↑, 1000 ↓）→ big-to-small ratio positive
    # Days 3-4: 大盘弱（300 ↓, 1000 ↑）→ ratio turns negative
    _idx(mr_db, LARGE_CAP_INDEX, "20260527", 3.0); _idx(mr_db, SMALL_CAP_INDEX, "20260527", -1.0)
    _idx(mr_db, LARGE_CAP_INDEX, "20260528", 2.0); _idx(mr_db, SMALL_CAP_INDEX, "20260528", -1.0)
    _idx(mr_db, LARGE_CAP_INDEX, "20260529", -2.0); _idx(mr_db, SMALL_CAP_INDEX, "20260529", 3.0)
    _idx(mr_db, LARGE_CAP_INDEX, "20260530", -3.0); _idx(mr_db, SMALL_CAP_INDEX, "20260530", 4.0)
    review = compute_style(
        mr_db, _w(("20260527", "20260528", "20260529", "20260530"))
    )
    assert review.flip_signal is True


def test_growth_index_carried_in_series(mr_db: Database) -> None:
    _idx(mr_db, LARGE_CAP_INDEX, "20260530", 1.0)
    _idx(mr_db, SMALL_CAP_INDEX, "20260530", 1.0)
    _idx(mr_db, GROWTH_INDEX, "20260530", 2.5)
    review = compute_style(mr_db, _w(("20260530",)))
    assert review.series[0].growth_ret == 2.5
