"""sentiment — thermometer score 0-100 + weight validation."""

from __future__ import annotations

import pytest
from deeptrade.core.db import Database

from market_review.metrics.breadth import compute_breadth
from market_review.metrics.sentiment import (
    DEFAULT_SENTIMENT_WEIGHTS,
    compute_sentiment,
)
from market_review.universe import UniverseSnapshot
from market_review.windows import Window


def _daily(db, ts_code, trade_date, pct_chg, amount=1000.0):
    db.execute(
        "INSERT INTO mr_daily (ts_code, trade_date, pct_chg, amount, close) VALUES (?, ?, ?, ?, ?)",
        [ts_code, trade_date, pct_chg, amount, 10.0],
    )


def _hsgt(db, trade_date, north_money_wan):
    db.execute(
        "INSERT INTO mr_moneyflow_hsgt (trade_date, north_money) VALUES (?, ?)",
        [trade_date, north_money_wan],
    )


def _w(trade_dates):
    return Window(
        mode="range" if len(trade_dates) > 1 else "day",
        start=trade_dates[0], end=trade_dates[-1],
        trade_dates=trade_dates, anchor=trade_dates[-1],
    )


def _universes(trade_dates, codes):
    return {
        td: UniverseSnapshot(
            trade_date=td, ts_codes=frozenset(codes),
            n_total_before=len(codes), excluded_st=0, excluded_delist=0, excluded_suspend=0,
        ) for td in trade_dates
    }


def test_score_in_zero_to_hundred_band(mr_db: Database) -> None:
    for code in ("A", "B", "C"):
        _daily(mr_db, code, "20260530", 5.0)
    universes = _universes(("20260530",), ["A", "B", "C"])
    breadth = compute_breadth(mr_db, _w(("20260530",)), universes)
    review = compute_sentiment(mr_db, _w(("20260530",)), universes, breadth)
    assert 0.0 <= review.series[0].score_0_100 <= 100.0


def test_weights_must_sum_to_one(mr_db: Database) -> None:
    _daily(mr_db, "A", "20260530", 1.0)
    universes = _universes(("20260530",), ["A"])
    breadth = compute_breadth(mr_db, _w(("20260530",)), universes)
    bad_weights = dict(DEFAULT_SENTIMENT_WEIGHTS)
    bad_weights["pos_ratio"] = 0.5  # was 0.30 → sum becomes 1.2
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_sentiment(mr_db, _w(("20260530",)), universes, breadth, weights=bad_weights)


def test_missing_weight_key_raises(mr_db: Database) -> None:
    _daily(mr_db, "A", "20260530", 1.0)
    universes = _universes(("20260530",), ["A"])
    breadth = compute_breadth(mr_db, _w(("20260530",)), universes)
    incomplete = {k: v for k, v in DEFAULT_SENTIMENT_WEIGHTS.items() if k != "north_inflow"}
    with pytest.raises(ValueError, match="missing keys"):
        compute_sentiment(mr_db, _w(("20260530",)), universes, breadth, weights=incomplete)


def test_north_money_converted_to_yi(mr_db: Database) -> None:
    _daily(mr_db, "A", "20260530", 1.0)
    # 100 亿 = 1,000,000 万元
    _hsgt(mr_db, "20260530", 1_000_000.0)
    universes = _universes(("20260530",), ["A"])
    breadth = compute_breadth(mr_db, _w(("20260530",)), universes)
    review = compute_sentiment(mr_db, _w(("20260530",)), universes, breadth)
    assert review.series[0].north_money_yi == 100.0


def test_extreme_day_picks_highest_score(mr_db: Database) -> None:
    """Strongest day should have the highest score; weakest the lowest."""
    # Day 1: weak (mostly down)
    for code in ("A", "B", "C"):
        _daily(mr_db, code, "20260528", -5.0)
    # Day 2: strong (mostly up)
    for code in ("A", "B", "C"):
        _daily(mr_db, code, "20260529", 6.0)
    universes = _universes(("20260528", "20260529"), ["A", "B", "C"])
    breadth = compute_breadth(mr_db, _w(("20260528", "20260529")), universes)
    review = compute_sentiment(mr_db, _w(("20260528", "20260529")), universes, breadth)
    assert review.strongest_day == "20260529"
    assert review.weakest_day == "20260528"
    assert 0.0 <= review.avg_score <= 100.0


def test_empty_universe_still_returns_zero_snapshot(mr_db: Database) -> None:
    """Empty universe → breadth produces 1 zero-snapshot per trade_date for
    time-axis continuity; sentiment mirrors that, score blends to the
    neutral-prior value (~12.5 with default weights)."""
    breadth = compute_breadth(mr_db, _w(("20260530",)), {})
    review = compute_sentiment(mr_db, _w(("20260530",)), {}, breadth)
    assert len(review.series) == 1
    snap = review.series[0]
    assert snap.trade_date == "20260530"
    assert snap.pos_ratio == 0.0
    # crash_ratio_inv contributes (1-0)*100*0.10 = 10; north_inflow neutral
    # 50 * 0.05 = 2.5. Other terms zero. Total = 12.5.
    assert 0.0 <= snap.score_0_100 <= 100.0
