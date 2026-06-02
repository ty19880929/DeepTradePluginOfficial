"""sectors — top / range / persistence / 三栏分类 / matrix."""

from __future__ import annotations

from deeptrade.core.db import Database

from market_review.metrics.sectors import SectorReview, compute_sectors
from market_review.windows import Window


def _ths(db, ts_code, trade_date, pct_change):
    db.execute(
        "INSERT INTO mr_ths_daily (ts_code, trade_date, pct_change, close) VALUES (?, ?, ?, ?)",
        [ts_code, trade_date, pct_change, 100.0],
    )


def _name(db, ts_code, name):
    db.execute(
        """INSERT INTO mr_ths_index (ts_code, name, type) VALUES (?, ?, ?)""",
        [ts_code, name, "N"],
    )


def _cnt_name(db, ts_code, name):
    db.execute(
        """INSERT INTO mr_moneyflow_cnt_ths
        (trade_date, ts_code, name, net_amount) VALUES (?, ?, ?, ?)""",
        ["20260530", ts_code, name, 0.0],
    )


def _w(trade_dates):
    return Window(
        mode="range" if len(trade_dates) > 1 else "day",
        start=trade_dates[0], end=trade_dates[-1],
        trade_dates=trade_dates, anchor=trade_dates[-1],
    )


def test_empty_returns_default(mr_db: Database) -> None:
    review = compute_sectors(mr_db, _w(("20260530",)))
    assert isinstance(review, SectorReview)
    assert review.today_top == []
    assert review.matrix is None


def test_today_top_ordered_by_anchor_day_pct(mr_db: Database) -> None:
    for code, pct in [("S1", 5.0), ("S2", 3.0), ("S3", 1.0), ("S4", -1.0)]:
        _ths(mr_db, code, "20260530", pct)
        _name(mr_db, code, f"{code}名")
    review = compute_sectors(mr_db, _w(("20260530",)), top_k=3)
    assert [e.ts_code for e in review.today_top] == ["S1", "S2", "S3"]
    assert review.today_top[0].name == "S1名"


def test_range_top_uses_cumulative_chain(mr_db: Database) -> None:
    # S1: +5% then +5% = ~10.25%
    # S2: +20% then -10% = ~8% (1.20 * 0.90 = 1.08)
    _ths(mr_db, "S1", "20260529", 5.0); _ths(mr_db, "S1", "20260530", 5.0)
    _ths(mr_db, "S2", "20260529", 20.0); _ths(mr_db, "S2", "20260530", -10.0)
    _name(mr_db, "S1", "S1"); _name(mr_db, "S2", "S2")
    review = compute_sectors(mr_db, _w(("20260529", "20260530")), top_k=2)
    range_codes = [e.ts_code for e in review.range_top]
    assert range_codes[0] == "S1"  # 10.25% > 8%
    assert range_codes[1] == "S2"


def test_persistence_counts_top_appearances(mr_db: Database) -> None:
    """Persistence = number of days the sector enters that day's top_k.

    Layout (top_k=2 of 3 sectors):
      - day1 pcts: S1=5, S2=6, S3=0.5 → top_2 = {S2, S1}
      - day2 pcts: S1=5, S2=-2, S3=0.1 → top_2 = {S1, S3}  ← S3 sneaks in
    So persistence(S1)=2, persistence(S2)=1, persistence(S3)=1.
    """
    _ths(mr_db, "S1", "20260529", 5.0); _ths(mr_db, "S1", "20260530", 5.0)
    _ths(mr_db, "S2", "20260529", 6.0); _ths(mr_db, "S2", "20260530", -2.0)
    _ths(mr_db, "S3", "20260529", 0.5); _ths(mr_db, "S3", "20260530", 0.1)
    _name(mr_db, "S1", "S1"); _name(mr_db, "S2", "S2"); _name(mr_db, "S3", "S3")
    review = compute_sectors(mr_db, _w(("20260529", "20260530")), top_k=2)
    per = {e.ts_code: e.persistence_days for e in (review.today_top + review.range_top)}
    assert per.get("S1") == 2
    assert per.get("S2") == 1
    assert per.get("S3") == 1


def test_classification_new_relay_fading(mr_db: Database) -> None:
    # day1 top = {S_old}; day2 top = {S_new, S_old}; day3 top = {S_new}.
    # earlier_half is first 1 day (3//2=1); today (day3) is S_new only.
    # → S_new is new_mainline (not in earlier_top); S_old is fading.
    _ths(mr_db, "S_old", "20260528", 6.0); _ths(mr_db, "S_old", "20260529", 4.0); _ths(mr_db, "S_old", "20260530", -3.0)
    _ths(mr_db, "S_new", "20260528", -1.0); _ths(mr_db, "S_new", "20260529", 3.0); _ths(mr_db, "S_new", "20260530", 7.0)
    _name(mr_db, "S_old", "S_old"); _name(mr_db, "S_new", "S_new")

    review = compute_sectors(mr_db, _w(("20260528", "20260529", "20260530")), top_k=1)
    today_codes = [e.ts_code for e in review.today_top]
    assert today_codes == ["S_new"]
    assert any(e.ts_code == "S_new" for e in review.new_mainline)
    assert any(e.ts_code == "S_old" for e in review.fading)


def test_industry_ti_name_resolved_from_ths_index_catalog(mr_db: Database) -> None:
    """v0.1.6 回归：``.TI`` 行业指数必须能从 ``mr_ths_index`` 拿到中文名，
    不再 fallback 到代码字面（v0.1.5 的渲染 bug：「板块名称显示 883422.TI」）。"""
    _ths(mr_db, "883422.TI", "20260530", 5.0)
    _name(mr_db, "883422.TI", "光模块")
    review = compute_sectors(mr_db, _w(("20260530",)))
    assert review.today_top[0].name == "光模块"
    assert review.today_top[0].ts_code == "883422.TI"


def test_sector_name_falls_back_to_moneyflow_cnt_ths(mr_db: Database) -> None:
    """``mr_ths_index`` 是主源；缺失时退到 ``mr_moneyflow_cnt_ths``。"""
    _ths(mr_db, "885000.TI", "20260530", 4.0)
    _cnt_name(mr_db, "885000.TI", "AI 概念")
    review = compute_sectors(mr_db, _w(("20260530",)))
    assert review.today_top[0].name == "AI 概念"


def test_sector_name_falls_back_to_code_when_no_mapping(mr_db: Database) -> None:
    """主源 + 退路都查不到时回到代码字面（不抛、不丢行）。"""
    _ths(mr_db, "883999.TI", "20260530", 3.0)
    review = compute_sectors(mr_db, _w(("20260530",)))
    assert review.today_top[0].name == "883999.TI"


def test_matrix_rows_ordered_by_cum(mr_db: Database) -> None:
    _ths(mr_db, "S_hot", "20260530", 10.0)
    _ths(mr_db, "S_cold", "20260530", -5.0)
    _name(mr_db, "S_hot", "S_hot"); _name(mr_db, "S_cold", "S_cold")
    review = compute_sectors(mr_db, _w(("20260530",)))
    matrix = review.matrix
    assert matrix is not None
    assert matrix.sectors[0] == "S_hot"
    assert matrix.cum_pct_chg[0] > matrix.cum_pct_chg[1]
