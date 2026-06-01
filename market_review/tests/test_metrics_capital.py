"""capital — 6 calibers + unit conversion + universe-respecting stock flows."""

from __future__ import annotations

from deeptrade.core.db import Database

from market_review.metrics.capital import CapitalReview, compute_capital
from market_review.universe import UniverseSnapshot
from market_review.windows import Window


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


def test_empty_universe_still_keeps_time_axis(mr_db: Database) -> None:
    """Empty universe doesn't strip the per-trade-date series — north_series
    keeps one entry per trade_date with ``north_money_yi=None`` for
    time-axis continuity (downstream LLM prompt needs the consistent shape)."""
    review = compute_capital(mr_db, _w(("20260530",)), {})
    assert isinstance(review, CapitalReview)
    assert len(review.north_series) == 1
    assert review.north_series[0].north_money_yi is None
    assert review.industry_top == []  # no industry rows → empty


def test_north_series_converts_wan_to_yi(mr_db: Database) -> None:
    # 北向 50 亿 = 500000 万元
    mr_db.execute(
        "INSERT INTO mr_moneyflow_hsgt (trade_date, north_money) VALUES (?, ?)",
        ["20260530", 500_000.0],
    )
    universes = _universes(("20260530",), ["A"])
    review = compute_capital(mr_db, _w(("20260530",)), universes)
    assert len(review.north_series) == 1
    assert review.north_series[0].north_money_yi == 50.0
    assert review.north_total_yi == 50.0


def test_north_top10_anchor_day_only(mr_db: Database) -> None:
    mr_db.execute(
        """INSERT INTO mr_hsgt_top10
        (trade_date, ts_code, name, market_type, net_amount) VALUES (?, ?, ?, ?, ?)""",
        ["20260529", "X", "X名", "1", 10_000.0],
    )
    mr_db.execute(
        """INSERT INTO mr_hsgt_top10
        (trade_date, ts_code, name, market_type, net_amount) VALUES (?, ?, ?, ?, ?)""",
        ["20260530", "Y", "Y名", "1", 20_000.0],
    )
    universes = _universes(("20260529", "20260530"), ["A"])
    review = compute_capital(mr_db, _w(("20260529", "20260530")), universes)
    # anchor=20260530 → only Y appears
    assert len(review.north_top10_anchor) == 1
    assert review.north_top10_anchor[0].ts_code == "Y"


def test_industry_top_and_bottom(mr_db: Database) -> None:
    for trade_date, name, net in [
        ("20260530", "光模块", 50_000.0),
        ("20260530", "机器人", 20_000.0),
        ("20260530", "银行", -30_000.0),
    ]:
        mr_db.execute(
            """INSERT INTO mr_moneyflow_ind_ths
            (trade_date, name, net_amount, pct_change) VALUES (?, ?, ?, ?)""",
            [trade_date, name, net, 1.0],
        )
    universes = _universes(("20260530",), ["A"])
    review = compute_capital(mr_db, _w(("20260530",)), universes)
    assert review.industry_top[0].name == "光模块"
    assert review.industry_top[0].net_inflow_yi == 5.0
    assert review.industry_bottom[0].name == "银行"
    assert review.industry_bottom[0].net_inflow_yi == -3.0


def test_stock_flow_respects_universe(mr_db: Database) -> None:
    for code, net in [("A", 30_000.0), ("B", -20_000.0), ("OUT", 99_999.0)]:
        mr_db.execute(
            "INSERT INTO mr_moneyflow (ts_code, trade_date, net_mf_amount) VALUES (?, ?, ?)",
            [code, "20260530", net],
        )
    # OUT is not in the universe — should NOT appear in top_inflow.
    universes = _universes(("20260530",), ["A", "B"])
    review = compute_capital(mr_db, _w(("20260530",)), universes)
    inflow_codes = {r.ts_code for r in review.stock_top_inflow}
    assert "OUT" not in inflow_codes
    assert "A" in inflow_codes
    assert review.stock_top_outflow[0].ts_code == "B"


def test_lhb_per_day_count(mr_db: Database) -> None:
    mr_db.execute(
        """INSERT INTO mr_top_list (trade_date, ts_code, net_amount) VALUES (?, ?, ?)""",
        ["20260530", "A", 50_000.0],
    )
    mr_db.execute(
        """INSERT INTO mr_top_list (trade_date, ts_code, net_amount) VALUES (?, ?, ?)""",
        ["20260530", "B", 20_000.0],
    )
    universes = _universes(("20260530",), ["A", "B"])
    review = compute_capital(mr_db, _w(("20260530",)), universes)
    assert len(review.lhb_series) == 1
    assert review.lhb_series[0].n_stocks == 2
    assert review.lhb_series[0].net_buy_yi == 7.0


def test_mkt_series_main_vs_retail(mr_db: Database) -> None:
    # 主力(lg+elg)净 = (5e4+3e4) - (2e4+1e4) = 5e4 → 5 亿
    # 散户(sm+md)净 = (4e4+3e4) - (3e4+2e4) = 2e4 → 2 亿
    mr_db.execute(
        """INSERT INTO mr_moneyflow_mkt
        (trade_date, buy_lg_amount, buy_elg_amount, sell_lg_amount, sell_elg_amount,
         buy_sm_amount, buy_md_amount, sell_sm_amount, sell_md_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ["20260530", 50_000.0, 30_000.0, 20_000.0, 10_000.0,
         40_000.0, 30_000.0, 30_000.0, 20_000.0],
    )
    universes = _universes(("20260530",), ["A"])
    review = compute_capital(mr_db, _w(("20260530",)), universes)
    assert len(review.mkt_series) == 1
    assert review.mkt_series[0].main_net_yi == 5.0
    assert review.mkt_series[0].retail_net_yi == 2.0
