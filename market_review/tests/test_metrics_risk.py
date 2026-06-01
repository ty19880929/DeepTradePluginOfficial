"""risk — 8 signals; smoke + key trigger / no-trigger paths."""

from __future__ import annotations

from deeptrade.core.db import Database

from market_review.metrics.breadth import BreadthReview, BreadthSnapshot, compute_breadth
from market_review.metrics.capital import (
    CapitalReview,
    NorthFlowRow,
    compute_capital,
)
from market_review.metrics.risk import RiskReview, compute_risk
from market_review.universe import UniverseSnapshot
from market_review.windows import Window


def _daily(db, ts_code, trade_date, pct_chg, open_=10.0, close=10.0, pre_close=10.0):
    db.execute(
        """INSERT INTO mr_daily
        (ts_code, trade_date, pct_chg, open, close, pre_close, amount)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [ts_code, trade_date, pct_chg, open_, close, pre_close, 1000.0],
    )


def _step(db, ts_code, trade_date, nums):
    db.execute(
        "INSERT INTO mr_limit_step (trade_date, ts_code, nums) VALUES (?, ?, ?)",
        [trade_date, ts_code, nums],
    )


def _limit(db, ts_code, trade_date, kind):
    db.execute(
        """INSERT INTO mr_limit_list_d (trade_date, ts_code, "limit") VALUES (?, ?, ?)""",
        [trade_date, ts_code, kind],
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


def test_review_returns_all_8_signals(mr_db: Database) -> None:
    universes = _universes(("20260530",), [])
    breadth = compute_breadth(mr_db, _w(("20260530",)), universes)
    capital = compute_capital(mr_db, _w(("20260530",)), universes)
    review = compute_risk(mr_db, _w(("20260530",)), universes, breadth=breadth, capital=capital)
    assert isinstance(review, RiskReview)
    assert len(review.signals) == 8
    names = {s.name for s in review.signals}
    assert names == {
        "high_position_drop",
        "stagnant_on_high_volume",
        "index_volume_divergence",
        "north_capital_outflow",
        "limit_down_spread",
        "block_trade_discount",
        "margin_balance_swing",
        "yaog_topping",
    }


def test_yaog_topping_triggered_on_5board_zhaban(mr_db: Database) -> None:
    _daily(mr_db, "A", "20260530", 0.0)
    _step(mr_db, "A", "20260530", 5)
    _limit(mr_db, "A", "20260530", "Z")  # 炸板
    universes = _universes(("20260530",), ["A"])
    breadth = compute_breadth(mr_db, _w(("20260530",)), universes)
    capital = compute_capital(mr_db, _w(("20260530",)), universes)
    review = compute_risk(mr_db, _w(("20260530",)), universes, breadth=breadth, capital=capital)
    yaog = next(s for s in review.signals if s.name == "yaog_topping")
    assert yaog.triggered is True
    assert "A" in yaog.affected_samples


def test_limit_down_spread_warning_at_threshold(mr_db: Database) -> None:
    breadth = BreadthReview(series=[BreadthSnapshot(
        trade_date="20260530",
        n_total=5000, n_up=2000, n_down=2900, n_flat=100,
        n_up5pct=100, n_down5pct=500,
        n_limit_up=10, n_limit_down=35, n_zhaban=5,
        up_ladder={2: 5, 3: 2}, n_lhb=20,
        total_amount_yi=8000.0, index_returns={"000001.SH": -2.0},
    )])
    capital = CapitalReview()
    universes = _universes(("20260530",), [])
    review = compute_risk(mr_db, _w(("20260530",)), universes, breadth=breadth, capital=capital)
    ldown = next(s for s in review.signals if s.name == "limit_down_spread")
    assert ldown.triggered is True
    assert ldown.severity == "critical"  # ≥ 30 triggers critical


def test_north_outflow_warning_when_both_today_and_window(mr_db: Database) -> None:
    breadth = BreadthReview()
    capital = CapitalReview(
        north_series=[NorthFlowRow(trade_date="20260530", north_money_yi=-15.0)],
        north_total_yi=-40.0,
    )
    universes = _universes(("20260530",), [])
    review = compute_risk(mr_db, _w(("20260530",)), universes, breadth=breadth, capital=capital)
    north = next(s for s in review.signals if s.name == "north_capital_outflow")
    assert north.triggered is True
    assert north.severity == "warning"


def test_index_volume_divergence_when_index_up_volume_shrinks(mr_db: Database) -> None:
    """上证涨 + 总成交额环比缩 > 15% → triggered."""
    breadth = BreadthReview(series=[
        BreadthSnapshot(trade_date="20260529", n_total=1, n_up=0, n_down=0, n_flat=0,
                        n_up5pct=0, n_down5pct=0, n_limit_up=0, n_limit_down=0, n_zhaban=0,
                        up_ladder={}, n_lhb=0, total_amount_yi=10000.0,
                        index_returns={"000001.SH": 0.0}),
        BreadthSnapshot(trade_date="20260530", n_total=1, n_up=0, n_down=0, n_flat=0,
                        n_up5pct=0, n_down5pct=0, n_limit_up=0, n_limit_down=0, n_zhaban=0,
                        up_ladder={}, n_lhb=0, total_amount_yi=8000.0,
                        index_returns={"000001.SH": 0.5}),
    ])
    capital = CapitalReview()
    universes = _universes(("20260529", "20260530"), [])
    review = compute_risk(mr_db, _w(("20260529", "20260530")),
                          universes, breadth=breadth, capital=capital)
    div = next(s for s in review.signals if s.name == "index_volume_divergence")
    # 10000 → 8000 is -20% (more than 15%) AND 上证 +0.5%
    assert div.triggered is True
