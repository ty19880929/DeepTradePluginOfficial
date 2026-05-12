"""Pytest fixtures shared across LGB-related tests.

Lives at the top of ``tests/`` so individual test modules can pull the same
stub TushareClient / on-disk fixtures / DB seed without cross-module imports
(``tests/`` is not a package, so relative imports don't work).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from deeptrade.core.db import Database

from volume_anomaly.lgb import paths as lgb_paths


# ---------------------------------------------------------------------------
# Stub TushareClient
# ---------------------------------------------------------------------------


class FakeTushareClient:
    """Returns pre-baked DataFrames per (api_name, cache_key).

    cache_key resolution:
        * ``trade_date`` (kwarg) → that string
        * ``params['trade_date']`` → that string
        * ``params['start_date'] + ':' + params['end_date']`` → range key
        * else → "*"
    Missing fixture returns an empty DataFrame (matches the framework's
    optional-API fallthrough).
    """

    def __init__(self, fixtures: dict[tuple[str, str], pd.DataFrame]) -> None:
        self._fixtures = fixtures

    def call(
        self,
        api_name: str,
        *,
        trade_date: str | None = None,
        params: dict[str, Any] | None = None,
        fields: str | None = None,  # noqa: ARG002
        force_sync: bool = False,  # noqa: ARG002
    ) -> pd.DataFrame:
        params = dict(params or {})
        if trade_date is not None:
            key = str(trade_date)
        elif "trade_date" in params:
            key = str(params["trade_date"])
        elif "start_date" in params and "end_date" in params:
            key = f"{params['start_date']}:{params['end_date']}"
        else:
            key = "*"
        return self._fixtures.get((api_name, key), pd.DataFrame()).copy()


# ---------------------------------------------------------------------------
# Date constants (kept package-private; tests import via fixtures below)
# ---------------------------------------------------------------------------


OPEN_DATES = [
    "20260601", "20260602", "20260603", "20260604", "20260605",
    "20260608", "20260609", "20260610", "20260611", "20260612",
]
ANOMALY_DATES = [
    "20260608", "20260609", "20260610", "20260611", "20260612",
]
CODES = ["000001.SZ", "000002.SZ", "600000.SH"]


def trade_cal_df() -> pd.DataFrame:
    rows = []
    cur = pd.Timestamp("2026-05-25")
    end = pd.Timestamp("2026-06-30")
    while cur <= end:
        cal_date = cur.strftime("%Y%m%d")
        is_open = 1 if cal_date in OPEN_DATES else 0
        rows.append({"cal_date": cal_date, "is_open": is_open})
        cur += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _daily_for_trade_date(td: str) -> pd.DataFrame:
    rows = []
    for i, code in enumerate(CODES):
        seed = (sum(ord(c) for c in code) + sum(int(d) for d in td)) % 17
        close = 10.0 + seed * 0.15 + 0.05 * int(td[-2:])
        rows.append(
            {
                "ts_code": code,
                "trade_date": td,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.3,
                "close": close,
                "pct_chg": 1.0 + 0.1 * seed,
                "vol": 100000 + seed * 100,
                "amount": close * (100000 + seed * 100) / 10.0,
            }
        )
    return pd.DataFrame(rows)


def _daily_basic_df(td: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": td,
                "turnover_rate": 5.0 + i,
                "volume_ratio": 1.2 + 0.1 * i,
                "pe": 18.0 + i,
                "pb": 1.6 + 0.1 * i,
                "circ_mv": (80.0 + 10 * i) * 1e4,
                "total_mv": (120.0 + 10 * i) * 1e4,
            }
            for i, code in enumerate(CODES)
        ]
    )


def _build_fixtures() -> dict[tuple[str, str], pd.DataFrame]:
    fixtures: dict[tuple[str, str], pd.DataFrame] = {}
    fixtures[("trade_cal", "*")] = trade_cal_df()
    for td in OPEN_DATES:
        fixtures[("daily", td)] = _daily_for_trade_date(td)
        fixtures[("daily_basic", td)] = _daily_basic_df(td)
    return fixtures


# ---------------------------------------------------------------------------
# Fixtures shared across test modules
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_tushare() -> FakeTushareClient:
    return FakeTushareClient(_build_fixtures())


@pytest.fixture
def fixture_calendar() -> pd.DataFrame:
    return trade_cal_df()


@pytest.fixture
def anomaly_dates() -> list[str]:
    return list(ANOMALY_DATES)


@pytest.fixture
def isolated_plugin_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    base = tmp_path / "deeptrade_home"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lgb_paths, "plugin_data_dir", lambda: base)
    # Re-derive subpath helpers from the patched plugin_data_dir.
    monkeypatch.setattr(lgb_paths, "models_dir", lambda: base / "models")
    monkeypatch.setattr(lgb_paths, "datasets_dir", lambda: base / "datasets")
    monkeypatch.setattr(
        lgb_paths, "checkpoints_dir", lambda: base / "checkpoints"
    )
    monkeypatch.setattr(
        lgb_paths, "latest_pointer", lambda: base / "models" / "latest.txt"
    )
    return base


@pytest.fixture
def db_with_anomalies(tmp_path: Path) -> Database:
    """Seed va_anomaly_history + va_realized_returns with a deterministic
    plan covering 5 anomaly_dates and a mix of labeled / pending samples."""
    db = Database(tmp_path / "lgb_dataset_test.duckdb")
    db.execute(
        """
        CREATE TABLE va_anomaly_history (
            trade_date          VARCHAR NOT NULL,
            ts_code             VARCHAR NOT NULL,
            name                VARCHAR,
            industry            VARCHAR,
            pct_chg             DOUBLE,
            close               DOUBLE,
            open                DOUBLE,
            high                DOUBLE,
            low                 DOUBLE,
            vol                 DOUBLE,
            amount              DOUBLE,
            body_ratio          DOUBLE,
            turnover_rate       DOUBLE,
            vol_ratio_5d        DOUBLE,
            max_vol_60d         DOUBLE,
            raw_metrics_json    VARCHAR,
            PRIMARY KEY (trade_date, ts_code)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE va_realized_returns (
            anomaly_date  VARCHAR NOT NULL,
            ts_code       VARCHAR NOT NULL,
            t_close       DOUBLE,
            t1_close      DOUBLE,
            t3_close      DOUBLE,
            t5_close      DOUBLE,
            t10_close     DOUBLE,
            ret_t1        DOUBLE,
            ret_t3        DOUBLE,
            ret_t5        DOUBLE,
            ret_t10       DOUBLE,
            max_close_5d  DOUBLE,
            max_close_10d DOUBLE,
            max_ret_5d    DOUBLE,
            max_ret_10d   DOUBLE,
            max_dd_5d     DOUBLE,
            computed_at   TIMESTAMP,
            data_status   VARCHAR NOT NULL,
            PRIMARY KEY (anomaly_date, ts_code)
        )
        """
    )
    plan = {
        "20260608": [("000001.SZ", 6.5, "complete"), ("000002.SZ", 2.0, "complete")],
        "20260609": [("000001.SZ", 5.0, "complete"), ("600000.SH", 4.9, "partial")],
        "20260610": [("000002.SZ", 8.0, "complete")],
        "20260611": [("000001.SZ", None, "pending"), ("600000.SH", 9.5, "complete")],
        "20260612": [("000002.SZ", 1.0, "complete"), ("600000.SH", 6.0, "complete")],
    }
    for anomaly_date, hits in plan.items():
        for code, mr5, status in hits:
            db.execute(
                "INSERT INTO va_anomaly_history "
                "(trade_date, ts_code, name, industry, pct_chg, close, open, high, "
                "low, vol, amount, body_ratio, turnover_rate, vol_ratio_5d, max_vol_60d) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    anomaly_date, code, f"Co_{code}", "测试行业",
                    7.0, 12.0, 11.5, 12.5, 11.0, 200000.0, 250000.0,
                    0.72, 8.5, 2.1, 250000.0,
                ),
            )
            db.execute(
                "INSERT INTO va_realized_returns "
                "(anomaly_date, ts_code, t_close, max_ret_5d, ret_t3, max_ret_10d, "
                "data_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (anomaly_date, code, 12.0, mr5, mr5, mr5, status),
            )
    yield db
    db.close()
