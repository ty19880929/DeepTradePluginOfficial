"""End-to-end screen test with mock Tushare — T2.12.

Spins up an in-memory DuckDB via the framework's Database, applies the APW
migration, hands a fake TushareClient to the runner, and asserts on persisted
rows.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.runner import ApwRunner, ScreenParams
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import NullRenderer
from tests.conftest import make_quotes


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def fresh_db(tmp_path):
    from deeptrade.core.db import Database

    dbpath = tmp_path / "apw_test.duckdb"
    db = Database(dbpath)
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for stmt in path.read_text(encoding="utf-8").split(";"):
            stmt = stmt.strip()
            if stmt:
                db.execute(stmt)
    yield db
    db.close()


class _FakeConfig:
    """Mimics enough of ConfigService for the runner."""

    def __init__(self, token: str = "test-token") -> None:
        self._values: dict[str, Any] = {"tushare.token": token}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_app_config(self):  # pragma: no cover - not used in fake path
        raise NotImplementedError


class _FakeLLMs:
    pass


class _FakeTushare:
    """Honour the call(api, **kwargs) contract used by data.fetch_*."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        # 5 fictional stocks with varied chart shapes
        codes = ["600000.SH", "600001.SH", "600002.SH", "600003.SH", "600004.SH"]
        self._codes = codes
        # one healthy 链路 (probe + washout), one ST, one no-probe, two random.
        self._daily: dict[str, pd.DataFrame] = {}
        for i, c in enumerate(codes):
            if i == 0:
                df = make_quotes(ts_code=c, pattern="flat", n=130,
                                 probe_index=110, probe_multiplier=5.0)
                probe_close = df.at[110, "close"]
                for j in range(111, 130):
                    df.at[j, "close"] = probe_close * 0.985
                    df.at[j, "high"] = df.at[j, "close"] * 1.005
                    df.at[j, "low"] = df.at[j, "close"] * 0.995
                    df.at[j, "vol"] = df.at[110, "vol"] * 0.4
                # Last day breakout
                df.at[129, "close"] = probe_close * 1.02
                df.at[129, "high"] = df.at[129, "close"] * 1.02
                df.at[129, "low"] = df.at[128, "close"] * 1.00
                df.at[129, "vol"] = df.at[110, "vol"] * 0.8
            else:
                df = make_quotes(ts_code=c, pattern=("flat" if i % 2 else "uptrend"), n=130)
            self._daily[c] = df

    def call(
        self,
        api: str,
        *,
        trade_date: str | None = None,
        params: dict[str, Any] | None = None,
        fields: str | None = None,
        force_sync: bool = False,
    ):
        kwargs: dict[str, Any] = {
            "trade_date": trade_date,
            "params": params,
            "fields": fields,
            "force_sync": force_sync,
        }
        self.calls.append((api, kwargs))
        params = params or {}
        if api == "trade_cal":
            base = pd.date_range("2024-01-01", "2026-12-31", freq="D")
            return pd.DataFrame(
                {
                    "cal_date": base.strftime("%Y%m%d"),
                    "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                    "pretrade_date": [None] * len(base),
                }
            )
        if api == "stock_basic":
            return pd.DataFrame(
                [
                    {"ts_code": c, "name": f"试盘股{i}", "market": "主板",
                     "exchange": "SSE", "list_status": "L", "list_date": "20100101",
                     "industry": "电子"}
                    for i, c in enumerate(self._codes)
                ]
            )
        if api == "stock_st":
            return pd.DataFrame([{"ts_code": "600001.SH"}])  # one ST
        if api == "suspend_d":
            return pd.DataFrame()
        if api == "daily":
            codes = params.get("ts_code", "").split(",")
            frames = [self._daily[c] for c in codes if c in self._daily]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        if api == "daily_basic":
            # New per-day shape: runner calls daily_basic(trade_date=YYYYMMDD).
            if trade_date is not None:
                frames = []
                for c, df in self._daily.items():
                    sub = df[df["trade_date"].astype(str) == str(trade_date)][
                        ["ts_code", "trade_date", "turnover_rate", "circ_mv"]
                    ]
                    if not sub.empty:
                        frames.append(sub)
                return (
                    pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                )
            # Legacy ts_code-list shape (unused by current runner; kept for
            # back-compat in tests that pin it). Real Tushare returns 0 rows
            # here — see APW空结果修复方案.
            codes = params.get("ts_code", "").split(",")
            frames = []
            for c in codes:
                if c in self._daily:
                    sub = self._daily[c][["ts_code", "trade_date", "turnover_rate", "circ_mv"]]
                    frames.append(sub)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if api == "moneyflow":
            # Provide positive net inflow → strengthens accumulation score
            codes = params.get("ts_code", "").split(",")
            rows: list[dict] = []
            for c in codes:
                if c not in self._daily:
                    continue
                for d in self._daily[c]["trade_date"].tail(30):
                    rows.append({"ts_code": c, "trade_date": d, "net_mf_amount": 4000.0})
            return pd.DataFrame(rows)
        return pd.DataFrame()


def test_screen_writes_signal_history_and_watchlist(fresh_db, monkeypatch):
    """Full screen path — 5 stocks, ST is dropped, hits land in DB."""
    tushare = _FakeTushare()
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=tushare,
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))

    assert outcome.status.value in {"success", "partial_failed"}, outcome.error
    # signal_history must have ≥ 0 rows (depending on score thresholds), and
    # watchlist must be a subset of signal_history rows.
    sh = fresh_db.fetchall("SELECT ts_code, phase FROM apw_signal_history")
    wl = fresh_db.fetchall("SELECT ts_code, phase FROM apw_watchlist")
    sh_codes = {row[0] for row in sh}
    wl_codes = {row[0] for row in wl}
    assert wl_codes.issubset(sh_codes)
    # ST stock must not appear anywhere
    assert "600001.SH" not in sh_codes
    assert "600001.SH" not in wl_codes


def test_screen_run_row_written(fresh_db):
    tushare = _FakeTushare()
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=tushare,
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))

    runs = fresh_db.fetchall("SELECT run_id, mode, status FROM apw_runs WHERE run_id = ?",
                              [outcome.run_id])
    assert len(runs) == 1
    assert runs[0][1] == "screen"
    # Events row should be > 0
    n_events = fresh_db.fetchone(
        "SELECT COUNT(*) FROM apw_events WHERE run_id = ?", [outcome.run_id]
    )[0]
    assert n_events > 0
