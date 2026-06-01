"""build_universe — ST / 退市 / 停牌 / 板块过滤组合 (design §3.1)."""

from __future__ import annotations

from deeptrade.core.db import Database

from market_review.universe import (
    NORTH_EXCHANGE_MARKET,
    UniverseSnapshot,
    build_universe,
)


def _insert_stock(
    db: Database,
    *,
    ts_code: str,
    name: str,
    market: str = "主板",
    list_status: str = "L",
    delist_date: str | None = None,
) -> None:
    db.execute(
        """
        INSERT INTO mr_stock_basic
            (ts_code, symbol, name, area, industry, market, exchange,
             list_status, list_date, delist_date)
        VALUES (?, ?, ?, NULL, NULL, ?, NULL, ?, NULL, ?)
        """,
        [ts_code, ts_code.split(".")[0], name, market, list_status, delist_date],
    )


def _insert_st(db: Database, ts_code: str, trade_date: str, status: str = "ST") -> None:
    db.execute(
        "INSERT INTO mr_stock_st (ts_code, trade_date, st_status) VALUES (?, ?, ?)",
        [ts_code, trade_date, status],
    )


def _insert_suspend(
    db: Database, ts_code: str, trade_date: str, suspend_type: str = "S"
) -> None:
    db.execute(
        "INSERT INTO mr_suspend_d (ts_code, trade_date, suspend_type) VALUES (?, ?, ?)",
        [ts_code, trade_date, suspend_type],
    )


def test_empty_db_returns_empty_snapshot(mr_db: Database) -> None:
    snap = build_universe(mr_db, trade_date="20260530")
    assert isinstance(snap, UniverseSnapshot)
    assert snap.ts_codes == frozenset()
    assert snap.n_total == 0
    assert snap.n_total_before == 0


def test_includes_all_four_markets_by_default(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="主板A", market="主板")
    _insert_stock(mr_db, ts_code="300001.SZ", name="创业板A", market="创业板")
    _insert_stock(mr_db, ts_code="688001.SH", name="科创A", market="科创板")
    _insert_stock(mr_db, ts_code="830001.BJ", name="北证A", market="北交所")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset(
        {"600001.SH", "300001.SZ", "688001.SH", "830001.BJ"}
    )


def test_exclude_north_exchange_removes_only_bj(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="主板A", market="主板")
    _insert_stock(mr_db, ts_code="830001.BJ", name="北证A", market="北交所")
    snap = build_universe(mr_db, trade_date="20260530", exclude_north_exchange=True)
    assert snap.ts_codes == frozenset({"600001.SH"})
    assert NORTH_EXCHANGE_MARKET == "北交所"


def test_excludes_st_by_name_prefix(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="某公司", market="主板")
    _insert_stock(mr_db, ts_code="600002.SH", name="ST示例", market="主板")
    _insert_stock(mr_db, ts_code="600003.SH", name="*ST警示", market="主板")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset({"600001.SH"})
    assert snap.excluded_st == 2


def test_excludes_st_by_stock_st_table(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="正常公司", market="主板")
    _insert_stock(mr_db, ts_code="600002.SH", name="未来ST", market="主板")
    _insert_st(mr_db, "600002.SH", "20260530", status="ST")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset({"600001.SH"})
    assert snap.excluded_st == 1


def test_excludes_delisted_by_name_substring(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="正常", market="主板")
    _insert_stock(mr_db, ts_code="600002.SH", name="某退市股", market="主板")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset({"600001.SH"})
    assert snap.excluded_delist == 1


def test_excludes_delisted_by_delist_date(mr_db: Database) -> None:
    _insert_stock(
        mr_db,
        ts_code="600001.SH",
        name="正常",
        market="主板",
        delist_date=None,
    )
    _insert_stock(
        mr_db,
        ts_code="600002.SH",
        name="历史退市",
        market="主板",
        delist_date="20260101",
    )
    _insert_stock(
        mr_db,
        ts_code="600003.SH",
        # Avoid the '退' char in the name — the two delist filters compose
        # (name OR date); this case isolates the date-only path.
        name="中国XYZ",
        market="主板",
        delist_date="20270101",
    )
    snap = build_universe(mr_db, trade_date="20260530")
    # 600002 退市日 ≤ 当前日 → 剔除；600003 退市日仍在未来 → 保留。
    assert snap.ts_codes == frozenset({"600001.SH", "600003.SH"})


def test_excludes_suspended_only_on_that_day(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="A", market="主板")
    _insert_stock(mr_db, ts_code="600002.SH", name="B", market="主板")
    _insert_suspend(mr_db, "600002.SH", "20260530", suspend_type="S")
    # Suspended on a different day should not affect 20260530's universe.
    _insert_suspend(mr_db, "600001.SH", "20260529", suspend_type="S")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset({"600001.SH"})
    assert snap.excluded_suspend == 1


def test_excludes_non_listed_status(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="正常", market="主板", list_status="L")
    _insert_stock(mr_db, ts_code="600002.SH", name="待上市", market="主板", list_status="P")
    _insert_stock(mr_db, ts_code="600003.SH", name="退市", market="主板", list_status="D")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset({"600001.SH"})


def test_combined_filters_count_separately(mr_db: Database) -> None:
    _insert_stock(mr_db, ts_code="600001.SH", name="正常A", market="主板")
    _insert_stock(mr_db, ts_code="600002.SH", name="ST已名前缀", market="主板")
    _insert_stock(
        mr_db, ts_code="600003.SH", name="退A", market="主板"
    )
    _insert_stock(mr_db, ts_code="600004.SH", name="停牌A", market="主板")
    _insert_suspend(mr_db, "600004.SH", "20260530")
    snap = build_universe(mr_db, trade_date="20260530")
    assert snap.ts_codes == frozenset({"600001.SH"})
    assert snap.excluded_st >= 1
    assert snap.excluded_delist >= 1
    assert snap.excluded_suspend >= 1
    assert snap.n_total_before == 4
