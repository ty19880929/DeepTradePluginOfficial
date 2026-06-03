"""leaders — 4-axis scoring + candidate pool + sector_top_hit."""

from __future__ import annotations

import math

from deeptrade.core.db import Database

from market_review.metrics.leaders import (
    LeaderCandidate,
    LeaderReview,
    _score_ladder,
    compute_leaders,
)
from market_review.metrics.sectors import SectorEntry, SectorReview
from market_review.universe import UniverseSnapshot
from market_review.windows import Window


def _stock(db, ts_code, name, industry=""):
    db.execute(
        """INSERT INTO mr_stock_basic
        (ts_code, symbol, name, industry, market, list_status)
        VALUES (?, ?, ?, ?, ?, ?)""",
        [ts_code, ts_code.split(".")[0], name, industry, "主板", "L"],
    )


def _daily(db, ts_code, trade_date, pct_chg):
    db.execute(
        "INSERT INTO mr_daily (ts_code, trade_date, pct_chg, close) VALUES (?, ?, ?, ?)",
        [ts_code, trade_date, pct_chg, 10.0],
    )


def _step(db, ts_code, trade_date, nums):
    db.execute(
        "INSERT INTO mr_limit_step (trade_date, ts_code, nums) VALUES (?, ?, ?)",
        [trade_date, ts_code, nums],
    )


def _moneyflow(db, ts_code, trade_date, net_amount):
    db.execute(
        "INSERT INTO mr_moneyflow (ts_code, trade_date, net_mf_amount) VALUES (?, ?, ?)",
        [ts_code, trade_date, net_amount],
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


def test_ladder_score_grows_log_with_nums() -> None:
    assert _score_ladder(0) == 0.0
    assert _score_ladder(1) == 0.0
    s2 = _score_ladder(2)
    s4 = _score_ladder(4)
    s7 = _score_ladder(7)
    assert 0 < s2 < s4 < s7 <= 25.0
    # Math: 25 * log2(8) / log2(8) = 25
    assert math.isclose(s7, 25.0, rel_tol=1e-6)


def test_empty_universe_returns_empty_review(mr_db: Database) -> None:
    review = compute_leaders(mr_db, _w(("20260530",)), {})
    assert isinstance(review, LeaderReview)
    assert review.primary == []


def test_pool_includes_ladder_and_top_returns(mr_db: Database) -> None:
    """Candidate pool = 连板≥2 ∪ top-50 by range return."""
    _stock(mr_db, "A", "A名"); _stock(mr_db, "B", "B名"); _stock(mr_db, "C", "C名")
    _daily(mr_db, "A", "20260530", 1.0)
    _daily(mr_db, "B", "20260530", 1.0)
    _daily(mr_db, "C", "20260530", 1.0)
    _step(mr_db, "A", "20260530", 3)  # ladder candidate
    # B will be top by return below; C is just along for the ride.
    universes = _universes(("20260530",), ["A", "B", "C"])
    review = compute_leaders(
        mr_db, _w(("20260530",)), universes,
        primary_k=5, min_score=0.0,
    )
    codes = {c.ts_code for c in review.primary}
    assert "A" in codes  # via ladder
    assert review.primary[0].score >= 0.0
    # Score bounds: 0..100
    for c in review.primary:
        assert 0.0 <= c.score <= 100.0


def test_score_breakdown_keys(mr_db: Database) -> None:
    _stock(mr_db, "A", "A名", industry="光模块")
    _daily(mr_db, "A", "20260530", 9.9)
    _step(mr_db, "A", "20260530", 4)
    _moneyflow(mr_db, "A", "20260530", 50_000.0)
    universes = _universes(("20260530",), ["A"])
    sector_review = SectorReview(today_top=[
        SectorEntry(ts_code="X", name="光模块", pct_chg=10.0, persistence_days=1),
    ])
    review = compute_leaders(
        mr_db, _w(("20260530",)), universes,
        sector_review=sector_review, min_score=0.0,
    )
    cand = review.primary[0]
    assert cand.ts_code == "A"
    assert set(cand.score_breakdown) == {"ladder", "return", "capital", "theme"}
    # 光模块 matches stock industry → theme = 25
    assert cand.score_breakdown["theme"] == 25.0
    assert "光模块" in cand.sector_top_hit
    # 4 连板 → ladder ~22 (≈ 25 * log2(5)/log2(8) ≈ 19.34)
    assert 10.0 < cand.score_breakdown["ladder"] <= 25.0


def test_min_score_filter(mr_db: Database) -> None:
    _stock(mr_db, "A", "A名")
    _daily(mr_db, "A", "20260530", 1.0)
    _step(mr_db, "A", "20260530", 2)
    universes = _universes(("20260530",), ["A"])
    review = compute_leaders(
        mr_db, _w(("20260530",)), universes,
        min_score=99.0,
    )
    assert review.primary == []


def test_default_min_score_is_30(mr_db: Database) -> None:
    """v0.1.14 — _DEFAULT_MIN_SCORE lowered from 50 → 30.

    Reason: the 50 cutoff combined with the v0.1 theme axis being
    structurally near-zero (Tushare ``stock_basic.industry`` 申万体系 ↔
    ``mr_ths_daily`` THS 板块 name taxonomy mismatch, see leaders.py top
    docstring) meant that on quiet days with no 连板, no candidate could
    clear 50, so primary+secondary came back empty and the LLM emitted
    a refusal in ``LeadersSection.error``. 30 lets median-percentile
    candidates pass on a single-axis edge so the LLM has data to score.
    """
    review = compute_leaders(mr_db, _w(("20260530",)), {})
    assert review.min_score == 30.0


def test_sector_map_filled_for_picked_candidates(mr_db: Database) -> None:
    _stock(mr_db, "A", "A名", industry="光模块")
    _stock(mr_db, "B", "B名", industry="光模块")
    _daily(mr_db, "A", "20260530", 5.0)
    _daily(mr_db, "B", "20260530", 5.0)
    _step(mr_db, "A", "20260530", 4)
    _step(mr_db, "B", "20260530", 3)
    universes = _universes(("20260530",), ["A", "B"])
    sector_review = SectorReview(today_top=[
        SectorEntry(ts_code="X", name="光模块", pct_chg=10.0, persistence_days=1),
    ])
    review = compute_leaders(
        mr_db, _w(("20260530",)), universes,
        sector_review=sector_review, min_score=0.0,
    )
    assert "光模块" in review.sector_map
    assert set(review.sector_map["光模块"]) == {"A", "B"}
