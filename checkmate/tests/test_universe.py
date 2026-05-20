"""Universe-builder tests — one happy + edge case per reason_code (PR-1.3).

Strategy
--------
Each test builds a tiny in-memory DuckDB with one or a few rows in
``checkmate_stock_status_history``, plants per-symbol daily / daily_basic
parquet fixtures under tmp_path, and runs :func:`build_universe`. The trade
calendar is loaded from a stub Tushare so we don't touch the network.

Why parquet fixtures instead of stubbing the fetcher: the cache hit path is
what the rest of the pipeline will use (sync populates caches once;
scan/signals/backtest only read), so exercising it here doubles as a
regression test for ``fetch_daily_raw`` / ``fetch_daily_basic`` behaviour
under exact-window slicing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from checkmate import data, paths
from checkmate.config import UniverseConfig
from checkmate.runtime import CheckmateRuntime
from checkmate.universe import build_universe, upsert_universe_daily


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# ---------------------------------------------------------------------------
# Stub Tushare — only trade_cal is consulted at universe-build time (everything
# else hits the parquet cache we plant in tmp_path).
# ---------------------------------------------------------------------------


class _StubTushare:
    def __init__(self, trade_cal: pd.DataFrame) -> None:
        self._trade_cal = trade_cal

    def call(self, api_name: str, **kwargs):
        if api_name == "trade_cal":
            return self._trade_cal.copy()
        return pd.DataFrame()


def _make_trade_cal(start: str = "20240101", end: str = "20240331") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B").strftime("%Y%m%d")
    return pd.DataFrame({"cal_date": list(dates), "is_open": [1] * len(dates)})


# ---------------------------------------------------------------------------
# Fixture: isolated runtime with migrated DB + tmp paths
# ---------------------------------------------------------------------------


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()

    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())

    rt = CheckmateRuntime(
        db=db, config=None,  # type: ignore[arg-type]
        tushare=_StubTushare(_make_trade_cal()),
    )
    yield rt
    db.close()


# ---------------------------------------------------------------------------
# Helpers to plant the DB + parquet fixtures
# ---------------------------------------------------------------------------


def _seed_status(
    db,
    ts_code: str,
    *,
    name: str = "测试股",
    industry: str = "test",
    list_date: str = "20100101",
    list_status: str = "L",
    is_st: bool = False,
    as_of_date: str | None = None,
) -> None:
    db.execute(
        """
        INSERT OR REPLACE INTO checkmate_stock_status_history
            (ts_code, as_of_date, list_status, is_st, name, industry,
             list_date, delist_date, raw_event_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, CURRENT_TIMESTAMP)
        """,
        [ts_code, as_of_date or list_date, list_status, is_st, name,
         industry, list_date, json.dumps({"source": "test"})],
    )


def _plant_daily_basic(
    ts_code: str,
    *,
    n_days: int = 20,
    end_date: str = "20240329",
    amount_qianyuan: float = 100_000.0,  # 千元 → 1 亿/天 by default
    turnover_rate: float = 1.5,
    blank_days: int = 0,
) -> None:
    """Write a daily_basic parquet at the canonical cache path."""
    dates = pd.bdate_range(end=end_date, periods=n_days).strftime("%Y%m%d")
    df = pd.DataFrame({
        "ts_code": [ts_code] * n_days,
        "trade_date": list(dates),
        "amount": [amount_qianyuan] * n_days,
        "turnover_rate": [turnover_rate] * n_days,
        "total_mv": [1e10] * n_days,
    })
    if blank_days > 0:
        df = df.iloc[blank_days:].reset_index(drop=True)
    cache_path = paths.daily_basic_cache_dir() / f"{ts_code}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def _plant_daily(
    ts_code: str,
    *,
    n_days: int = 20,
    end_date: str = "20240329",
    close: float = 10.0,
    one_way_days: int = 0,
) -> None:
    """Write a daily parquet (with adj_factor=1) at the canonical cache path."""
    dates = list(pd.bdate_range(end=end_date, periods=n_days).strftime("%Y%m%d"))
    high = [close * 1.02] * n_days
    low = [close * 0.98] * n_days
    # Make the first ``one_way_days`` rows have high==low (一字 detection).
    for i in range(min(one_way_days, n_days)):
        high[i] = close
        low[i] = close
    df = pd.DataFrame({
        "ts_code": [ts_code] * n_days,
        "trade_date": dates,
        "open": [close] * n_days,
        "high": high,
        "low": low,
        "close": [close] * n_days,
        "pre_close": [close] * n_days,
        "adj_factor": [1.0] * n_days,
    })
    cache_path = paths.daily_cache_dir() / f"{ts_code}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_eligible(rt) -> None:
    _seed_status(rt.db, "600519.SH", name="贵州茅台", list_date="20010827")
    _plant_daily_basic("600519.SH", amount_qianyuan=2_000_000.0)  # 20 亿/天
    _plant_daily("600519.SH", close=1700.0)

    snap = build_universe(rt, "20240329", UniverseConfig(price_band_high=None))
    assert len(snap.rows) == 1
    row = snap.rows[0]
    assert row.eligible
    assert row.reason_codes == []
    # 2_000_000 千元 × 1000 = 2e9 yuan; liquidity_score = 20.0 (亿元)
    assert row.liquidity_score == pytest.approx(20.0, abs=1e-6)
    assert row.is_st is False


# ---------------------------------------------------------------------------
# st
# ---------------------------------------------------------------------------


def test_reason_st_flagged(rt) -> None:
    _seed_status(rt.db, "000005.SZ", name="ST星源", is_st=True,
                 list_date="19901210")
    _plant_daily_basic("000005.SZ", amount_qianyuan=200_000.0)  # 2 亿/天
    _plant_daily("000005.SZ", close=5.0)
    snap = build_universe(rt, "20240329", UniverseConfig())
    row = snap.rows[0]
    assert "st" in row.reason_codes
    assert row.eligible is False


# ---------------------------------------------------------------------------
# new_listing
# ---------------------------------------------------------------------------


def test_reason_new_listing(rt) -> None:
    # Listed only 30 calendar days before trade_date.
    list_date = (datetime.strptime("20240329", "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    _seed_status(rt.db, "300999.SZ", name="新上市", list_date=list_date)
    _plant_daily_basic("300999.SZ")
    _plant_daily("300999.SZ", close=20.0)
    snap = build_universe(rt, "20240329", UniverseConfig(listed_days_min=250))
    row = snap.rows[0]
    assert "new_listing" in row.reason_codes


def test_listed_long_enough_passes_new_listing_gate(rt) -> None:
    list_date = (datetime.strptime("20240329", "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
    _seed_status(rt.db, "300888.SZ", name="老上市", list_date=list_date)
    _plant_daily_basic("300888.SZ")
    _plant_daily("300888.SZ", close=20.0)
    snap = build_universe(rt, "20240329", UniverseConfig(listed_days_min=250))
    row = snap.rows[0]
    assert "new_listing" not in row.reason_codes


# ---------------------------------------------------------------------------
# thin_trading
# ---------------------------------------------------------------------------


def test_reason_thin_trading(rt) -> None:
    _seed_status(rt.db, "600100.SH", name="经常停牌", list_date="20000101")
    _plant_daily_basic("600100.SH", blank_days=15)  # only 5 rows after slicing
    _plant_daily("600100.SH", n_days=5, close=10.0)
    snap = build_universe(rt, "20240329", UniverseConfig(thin_trading_min_days=18))
    row = snap.rows[0]
    assert "thin_trading" in row.reason_codes


# ---------------------------------------------------------------------------
# low_amount
# ---------------------------------------------------------------------------


def test_reason_low_amount(rt) -> None:
    _seed_status(rt.db, "002000.SZ", name="小流动性", list_date="20100101")
    # 1 千元/天 → 1000 元/天, well below 5000 万 floor.
    _plant_daily_basic("002000.SZ", amount_qianyuan=1.0)
    _plant_daily("002000.SZ", close=10.0)
    snap = build_universe(rt, "20240329", UniverseConfig())
    row = snap.rows[0]
    assert "low_amount" in row.reason_codes


# ---------------------------------------------------------------------------
# one_way_limit
# ---------------------------------------------------------------------------


def test_reason_one_way_limit(rt) -> None:
    _seed_status(rt.db, "002001.SZ", name="一字股", list_date="20000101")
    _plant_daily_basic("002001.SZ")
    # 10 of 20 sessions one-way → 50% > 25% threshold.
    _plant_daily("002001.SZ", close=10.0, one_way_days=10)
    snap = build_universe(rt, "20240329",
                          UniverseConfig(one_way_limit_max_ratio=0.25))
    row = snap.rows[0]
    assert "one_way_limit" in row.reason_codes


def test_one_way_below_threshold_does_not_flag(rt) -> None:
    _seed_status(rt.db, "002002.SZ", name="偶尔一字", list_date="20000101")
    _plant_daily_basic("002002.SZ")
    _plant_daily("002002.SZ", close=10.0, one_way_days=3)  # 3/20 = 15% < 25%
    snap = build_universe(rt, "20240329",
                          UniverseConfig(one_way_limit_max_ratio=0.25))
    row = snap.rows[0]
    assert "one_way_limit" not in row.reason_codes


# ---------------------------------------------------------------------------
# price_band
# ---------------------------------------------------------------------------


def test_reason_price_band_low(rt) -> None:
    _seed_status(rt.db, "300111.SZ", name="仙股", list_date="20100101")
    _plant_daily_basic("300111.SZ")
    _plant_daily("300111.SZ", close=1.5)  # below 2.0 floor
    snap = build_universe(rt, "20240329", UniverseConfig(price_band_low=2.0))
    row = snap.rows[0]
    assert "price_band" in row.reason_codes


def test_reason_price_band_high(rt) -> None:
    _seed_status(rt.db, "600999.SH", name="高价股", list_date="20100101")
    _plant_daily_basic("600999.SH")
    _plant_daily("600999.SH", close=5000.0)
    snap = build_universe(rt, "20240329",
                          UniverseConfig(price_band_low=2.0, price_band_high=1000.0))
    row = snap.rows[0]
    assert "price_band" in row.reason_codes


# ---------------------------------------------------------------------------
# Multiple reason_codes stack; delisted is dropped entirely
# ---------------------------------------------------------------------------


def test_multiple_reasons_stack(rt) -> None:
    _seed_status(rt.db, "002003.SZ", name="ST多病", is_st=True, list_date="20240101")
    _plant_daily_basic("002003.SZ", amount_qianyuan=1.0)
    _plant_daily("002003.SZ", close=1.0)
    snap = build_universe(rt, "20240329",
                          UniverseConfig(listed_days_min=250, price_band_low=2.0))
    row = snap.rows[0]
    assert set(row.reason_codes) >= {"st", "new_listing", "low_amount", "price_band"}
    assert row.eligible is False


def test_delisted_stock_dropped_from_universe(rt) -> None:
    _seed_status(rt.db, "600485.SH", name="*ST信威", list_date="20111219",
                 list_status="D", is_st=True)
    snap = build_universe(rt, "20240329", UniverseConfig())
    assert all(r.ts_code != "600485.SH" for r in snap.rows)


# ---------------------------------------------------------------------------
# Sort stability + persistence
# ---------------------------------------------------------------------------


def test_rows_sorted_descending_by_liquidity_then_ts_code(rt) -> None:
    # Three stocks with different liquidity scores
    _seed_status(rt.db, "600001.SH", name="A", list_date="20000101")
    _seed_status(rt.db, "600002.SH", name="B", list_date="20000101")
    _seed_status(rt.db, "600003.SH", name="C", list_date="20000101")
    _plant_daily_basic("600001.SH", amount_qianyuan=100_000.0)   # 1 亿
    _plant_daily_basic("600002.SH", amount_qianyuan=300_000.0)   # 3 亿
    _plant_daily_basic("600003.SH", amount_qianyuan=200_000.0)   # 2 亿
    for code in ("600001.SH", "600002.SH", "600003.SH"):
        _plant_daily(code, close=10.0)
    snap = build_universe(rt, "20240329", UniverseConfig())
    ts_codes = [r.ts_code for r in snap.rows]
    assert ts_codes == ["600002.SH", "600003.SH", "600001.SH"]


def test_upsert_universe_daily_writes_all_rows(rt) -> None:
    _seed_status(rt.db, "600519.SH", name="贵州茅台", list_date="20010827")
    _plant_daily_basic("600519.SH", amount_qianyuan=1_000_000.0)
    _plant_daily("600519.SH", close=1700.0)
    snap = build_universe(rt, "20240329", UniverseConfig(price_band_high=None))
    n = upsert_universe_daily(rt.db, snap)
    assert n == len(snap.rows)
    rows = rt.db.execute(
        "SELECT ts_code, eligible, reason_codes, liquidity_score "
        "FROM checkmate_universe_daily WHERE trade_date = ?",
        ["20240329"],
    ).fetchall()
    assert len(rows) == n
    # Persisted reason_codes is JSON
    assert json.loads(rows[0][2]) == []
