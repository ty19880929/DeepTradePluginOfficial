from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from deeptrade.core.db import Database

from vwap_reversion.data import (
    build_daily_features,
    list_enabled_universe,
    sync_etf_daily,
    sync_etf_universe,
    sync_margin_eligibility,
)

MIGRATIONS = [
    Path(__file__).resolve().parent.parent / "migrations" / "20260603_001_init.sql",
    Path(__file__).resolve().parent.parent / "migrations" / "20260701_001_etf_cache.sql",
]


class FakeTushare:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, api: str, params: dict | None = None, **_kw):
        self.calls.append((api, params or {}))
        if api == "fund_basic":
            return pd.DataFrame([
                {
                    "ts_code": "159518.SZ",
                    "name": "标的ETF",
                    "fund_type": "股票型",
                    "invest_type": "指数型",
                    "market": "E",
                    "status": "L",
                    "list_date": "20230101",
                    "management": "基金公司",
                    "benchmark": "基准指数",
                }
            ])
        if api == "margin_secs":
            return pd.DataFrame([{"ts_code": "159518.SZ", "name": "标的ETF"}])
        if api == "fund_daily":
            return pd.DataFrame([
                {
                    "ts_code": "159518.SZ",
                    "trade_date": "20260630",
                    "open": 1.0,
                    "high": 1.02,
                    "low": 0.99,
                    "close": 1.01,
                    "pre_close": 1.0,
                    "pct_chg": 1.0,
                    "vol": 1000000,
                    "amount": 1010000,
                }
            ])
        if api == "fund_adj":
            return pd.DataFrame([{"ts_code": "159518.SZ", "trade_date": "20260630", "adj_factor": 1.2}])
        if api == "fund_share":
            return pd.DataFrame([{"ts_code": "159518.SZ", "trade_date": "20260630", "fd_share": 500000000}])
        if api == "fund_nav":
            return pd.DataFrame([
                {
                    "ts_code": "159518.SZ",
                    "nav_date": "20260630",
                    "unit_nav": 1.005,
                    "adj_nav": 1.206,
                }
            ])
        if api == "stk_limit":
            return pd.DataFrame([
                {
                    "ts_code": "159518.SZ",
                    "trade_date": "20260630",
                    "up_limit": 1.111,
                    "down_limit": 0.909,
                }
            ])
        return pd.DataFrame()


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "data.duckdb")
    for migration in MIGRATIONS:
        sql = migration.read_text(encoding="utf-8")
        for stmt in sql.split(";"):
            if stmt.strip():
                database.execute(stmt)
    yield database
    database.close()


def test_sync_etf_universe_marks_t0_and_margin(db: Database) -> None:
    ts = FakeTushare()
    result = sync_etf_universe(db, ts, t0_whitelist=["159518.SZ"])
    assert result.rows == 1

    margin = sync_margin_eligibility(db, ts, trade_date="20260630")
    assert margin.rows == 2  # SSE + SZSE fake responses

    rows = list_enabled_universe(db)
    assert rows[0]["ts_code"] == "159518.SZ"
    assert rows[0]["t0_eligible"] == 1
    assert rows[0]["margin_eligible"] == 1


def test_sync_etf_daily_combines_auxiliary_sources(db: Database) -> None:
    ts = FakeTushare()
    result = sync_etf_daily(
        db,
        ts,
        code="159518.SZ",
        start="20260601",
        end="20260630",
    )
    assert result.rows == 1
    row = db.fetchone(
        "SELECT close, adj_factor, fd_share, unit_nav, up_limit, down_limit "
        "FROM vwr_etf_daily WHERE ts_code = '159518.SZ' AND trade_date = '20260630'"
    )
    assert row == pytest.approx((1.01, 1.2, 500000000, 1.005, 1.111, 0.909))


def test_build_daily_features_from_cached_daily(db: Database) -> None:
    for i in range(25):
        day = f"202606{i + 1:02d}"
        pre_close = 1.0 + i * 0.001
        close = pre_close * 1.002
        db.execute(
            "INSERT INTO vwr_etf_daily(ts_code, trade_date, open, high, low, close, "
            "pre_close, amount) VALUES ('159518.SZ', ?, ?, ?, ?, ?, ?, ?)",
            (
                day,
                pre_close * 1.001,
                close * 1.01,
                close * 0.99,
                close,
                pre_close,
                300_000_000 + i * 1_000_000,
            ),
        )
    result = build_daily_features(
        db,
        code="159518.SZ",
        start="20260620",
        end="20260625",
        min_amount_ma20=200_000_000,
    )
    assert result.rows == 6
    row = db.fetchone(
        "SELECT ret_1d, ret_5d, amount_ma20, liquidity_ok, volatility_regime, "
        "trend_regime FROM vwr_daily_features WHERE ts_code='159518.SZ' "
        "AND trade_date='20260625'"
    )
    assert row[0] == pytest.approx(0.002)
    assert row[1] is not None
    assert row[2] >= 300_000_000
    assert row[3] == 1
    assert row[4] in {"low", "normal", "high"}
    assert row[5] in {"range", "up", "down"}
