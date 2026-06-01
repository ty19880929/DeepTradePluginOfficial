"""breadth — per-day counts + window aggregate."""

from __future__ import annotations

from deeptrade.core.db import Database

from market_review.metrics.breadth import BreadthReview, compute_breadth
from market_review.universe import UniverseSnapshot
from market_review.windows import Window


def _daily(db: Database, ts_code: str, trade_date: str, pct_chg: float, amount: float = 1000.0) -> None:
    db.execute(
        "INSERT INTO mr_daily (ts_code, trade_date, pct_chg, amount, close) VALUES (?, ?, ?, ?, ?)",
        [ts_code, trade_date, pct_chg, amount, 10.0],
    )


def _limit(db: Database, ts_code: str, trade_date: str, limit_kind: str) -> None:
    db.execute(
        """INSERT INTO mr_limit_list_d (trade_date, ts_code, "limit") VALUES (?, ?, ?)""",
        [trade_date, ts_code, limit_kind],
    )


def _step(db: Database, ts_code: str, trade_date: str, nums: int) -> None:
    db.execute(
        "INSERT INTO mr_limit_step (trade_date, ts_code, nums) VALUES (?, ?, ?)",
        [trade_date, ts_code, nums],
    )


def _idx_daily(db: Database, ts_code: str, trade_date: str, pct_chg: float) -> None:
    db.execute(
        "INSERT INTO mr_index_daily (ts_code, trade_date, pct_chg) VALUES (?, ?, ?)",
        [ts_code, trade_date, pct_chg],
    )


def _w(trade_dates: tuple[str, ...]) -> Window:
    return Window(
        mode="range" if len(trade_dates) > 1 else "day",
        start=trade_dates[0], end=trade_dates[-1],
        trade_dates=trade_dates, anchor=trade_dates[-1],
    )


def _universe(trade_date: str, codes: list[str]) -> dict[str, UniverseSnapshot]:
    return {
        trade_date: UniverseSnapshot(
            trade_date=trade_date,
            ts_codes=frozenset(codes),
            n_total_before=len(codes),
            excluded_st=0, excluded_delist=0, excluded_suspend=0,
        )
    }


def test_breadth_counts_up_down_flat(mr_db: Database) -> None:
    _daily(mr_db, "A", "20260530", 2.0)
    _daily(mr_db, "B", "20260530", -3.0)
    _daily(mr_db, "C", "20260530", 0.0)
    _daily(mr_db, "D", "20260530", 6.0)
    _daily(mr_db, "E", "20260530", -7.0)
    universes = _universe("20260530", ["A", "B", "C", "D", "E"])
    review = compute_breadth(mr_db, _w(("20260530",)), universes)
    snap = review.series[0]
    assert snap.n_total == 5
    assert snap.n_up == 2
    assert snap.n_down == 2
    assert snap.n_flat == 1
    assert snap.n_up5pct == 1
    assert snap.n_down5pct == 1


def test_breadth_limit_and_ladder(mr_db: Database) -> None:
    for code, pct in [("A", 9.9), ("B", -9.9), ("C", 5.0)]:
        _daily(mr_db, code, "20260530", pct)
    _limit(mr_db, "A", "20260530", "U")
    _limit(mr_db, "B", "20260530", "D")
    _limit(mr_db, "C", "20260530", "Z")
    _step(mr_db, "A", "20260530", 3)
    _step(mr_db, "X", "20260530", 5)  # not in universe → should not count
    universes = _universe("20260530", ["A", "B", "C"])
    review = compute_breadth(mr_db, _w(("20260530",)), universes)
    snap = review.series[0]
    assert snap.n_limit_up == 1
    assert snap.n_limit_down == 1
    assert snap.n_zhaban == 1
    assert snap.up_ladder == {3: 1}  # X filtered out by universe


def test_breadth_index_returns_and_extremes(mr_db: Database) -> None:
    _daily(mr_db, "A", "20260528", 2.0)
    _daily(mr_db, "A", "20260529", -3.0)
    _daily(mr_db, "B", "20260528", 1.0)
    _daily(mr_db, "B", "20260529", -1.0)
    _idx_daily(mr_db, "000001.SH", "20260528", 0.5)
    _idx_daily(mr_db, "000001.SH", "20260529", -0.8)

    universes = {
        td: UniverseSnapshot(
            trade_date=td, ts_codes=frozenset(["A", "B"]),
            n_total_before=2, excluded_st=0, excluded_delist=0, excluded_suspend=0,
        ) for td in ("20260528", "20260529")
    }
    review = compute_breadth(mr_db, _w(("20260528", "20260529")), universes)
    assert review.series[0].index_returns["000001.SH"] == 0.5
    assert review.series[1].index_returns["000001.SH"] == -0.8
    assert review.sentiment_extreme_day[0] == "20260528"  # n_up=2 day
    assert review.sentiment_extreme_day[1] == "20260529"
    # n_up for 20260528 = 2 (both A and B up), 20260529 = 0
    assert review.series[0].n_up == 2
    assert review.series[1].n_up == 0
    # median of [2, 0] is 1 (statistics.median returns float; int() truncates)
    assert review.median_up_count == 1


def test_breadth_empty_universe_returns_zeros(mr_db: Database) -> None:
    review = compute_breadth(mr_db, _w(("20260530",)), {})
    assert len(review.series) == 1
    assert review.series[0].n_total == 0
    assert review.median_up_count == 0
    assert isinstance(review, BreadthReview)


def test_breadth_amount_yi_conversion(mr_db: Database) -> None:
    """amount in 千元 → /1e5 = 亿. 1e9 千元 = 1e4 亿。"""
    _daily(mr_db, "A", "20260530", 1.0, amount=1_000_000_000.0)
    universes = _universe("20260530", ["A"])
    review = compute_breadth(mr_db, _w(("20260530",)), universes)
    assert review.series[0].total_amount_yi == 10_000.0
