"""End-to-end smoke test for the ``sync`` orchestrator (PR-1.2).

5 synthetic ts_codes × 60-day window against a stub Tushare. Verifies:
  * ``checkmate_stock_status_history`` rows are written for every ts_code
  * Per-symbol parquet caches land under the daily / daily_basic cache roots
  * Re-running with the same cache does not re-call the upstream APIs
  * ``--force-refresh`` bypasses the parquet cache and re-pulls
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from checkmate import paths
from checkmate.runtime import CheckmateRuntime
from checkmate.sync import SyncParams, run_sync


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# ---------------------------------------------------------------------------
# Stub Tushare client (response-driven)
# ---------------------------------------------------------------------------


SYMBOLS = ["600000.SH", "600519.SH", "000001.SZ", "002415.SZ", "300750.SZ"]


def _trade_cal_frame(start: str = "20240101", end: str = "20240331") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B").strftime("%Y%m%d")
    return pd.DataFrame({"cal_date": list(dates), "is_open": [1] * len(dates)})


def _stock_basic_frame() -> pd.DataFrame:
    base = [
        ("600000.SH", "浦发银行", "银行",      "主板", "SSE",  "L", "19991110", None),
        ("600519.SH", "贵州茅台", "白酒",      "主板", "SSE",  "L", "20010827", None),
        ("000001.SZ", "平安银行", "银行",      "主板", "SZSE", "L", "19910403", None),
        ("002415.SZ", "海康威视", "电子",      "主板", "SZSE", "L", "20100528", None),
        ("300750.SZ", "宁德时代", "电池",      "创业板", "SZSE", "L", "20180611", None),
    ]
    cols = ("ts_code", "name", "industry", "market", "exchange",
            "list_status", "list_date", "delist_date")
    return pd.DataFrame(base, columns=cols)


def _namechange_frame() -> pd.DataFrame:
    return pd.DataFrame([
        # Pretend 000001.SZ had an ST stint in 2009 (synthetic)
        {"ts_code": "000001.SZ", "name": "ST深发展", "start_date": "20080101"},
        {"ts_code": "000001.SZ", "name": "平安银行",  "start_date": "20120101"},
    ])


def _daily_frame(ts_code: str) -> pd.DataFrame:
    dates = pd.date_range("20240102", "20240329", freq="B").strftime("%Y%m%d")
    n = len(dates)
    return pd.DataFrame({
        "ts_code":   [ts_code] * n,
        "trade_date": list(dates),
        "open":      [10.0 + i * 0.01 for i in range(n)],
        "high":      [10.5 + i * 0.01 for i in range(n)],
        "low":       [ 9.5 + i * 0.01 for i in range(n)],
        "close":     [10.2 + i * 0.01 for i in range(n)],
        "pre_close": [10.1 + i * 0.01 for i in range(n)],
        "vol":       [1_000_000.0] * n,
        "amount":    [10_000.0] * n,
    })


def _adj_factor_frame(ts_code: str) -> pd.DataFrame:
    dates = pd.date_range("20240102", "20240329", freq="B").strftime("%Y%m%d")
    return pd.DataFrame({
        "ts_code":   [ts_code] * len(dates),
        "trade_date": list(dates),
        "adj_factor": [1.0] * len(dates),
    })


def _daily_basic_frame(ts_code: str) -> pd.DataFrame:
    dates = pd.date_range("20240102", "20240329", freq="B").strftime("%Y%m%d")
    n = len(dates)
    return pd.DataFrame({
        "ts_code":       [ts_code] * n,
        "trade_date":    list(dates),
        "turnover_rate": [0.5] * n,
        "amount":        [10_000.0] * n,
        "total_mv":      [1e10] * n,
    })


class _StubTushare:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, api_name: str, **kwargs):
        self.calls.append((api_name, dict(kwargs)))
        if api_name == "trade_cal":
            return _trade_cal_frame()
        if api_name == "stock_basic":
            status = kwargs.get("list_status", "L")
            sb = _stock_basic_frame()
            return sb[sb["list_status"] == status]
        if api_name == "namechange":
            return _namechange_frame()
        if api_name == "daily":
            return _daily_frame(kwargs["ts_code"])
        if api_name == "adj_factor":
            return _adj_factor_frame(kwargs["ts_code"])
        if api_name == "daily_basic":
            return _daily_basic_frame(kwargs["ts_code"])
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every checkmate.paths.* helper into tmp_path."""
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()
    return tmp_path


@pytest.fixture
def rt(isolated_paths, tmp_path):
    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(isolated_paths / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())
    # Build a minimal CheckmateRuntime — ConfigService is None because the
    # sync orchestrator never reads it (TushareClient injection happens in
    # cli.cmd_sync via build_tushare_client; tests inject the stub directly).
    rt = CheckmateRuntime(db=db, config=None, tushare=_StubTushare())  # type: ignore[arg-type]
    yield rt
    db.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sync_writes_status_history_and_caches(rt, isolated_paths) -> None:
    log: list[str] = []
    out = run_sync(
        rt,
        SyncParams(start="20240102", end="20240329", symbols=SYMBOLS),
        echo=log.append,
    )
    assert out.n_symbols == 5
    assert out.n_status_rows >= 5  # at least one per ts_code
    assert out.errors == []

    # DB rows
    n_rows = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_stock_status_history"
    ).fetchone()[0]
    assert n_rows == out.n_status_rows

    # Parquet caches
    daily_dir = isolated_paths / "checkmate" / "cache" / "daily"
    basic_dir = isolated_paths / "checkmate" / "cache" / "daily_basic"
    for code in SYMBOLS:
        assert (daily_dir / f"{code}.parquet").is_file(), f"missing daily cache for {code}"
        assert (basic_dir / f"{code}.parquet").is_file(), f"missing daily_basic cache for {code}"

    # trade_cal + stock_basic_all parquets
    assert (isolated_paths / "checkmate" / "cache" / "trade_cal.parquet").is_file()
    assert (isolated_paths / "checkmate" / "cache" / "namechange.parquet").is_file()
    assert (isolated_paths / "checkmate" / "cache" / "stock_basic_all.parquet").is_file()

    # Progress lines emitted
    assert any("[sync] window=" in line for line in log)
    assert any("[sync] done" in line for line in log)


def test_second_sync_hits_cache(rt) -> None:
    run_sync(rt, SyncParams(start="20240102", end="20240329", symbols=SYMBOLS))
    n_calls_after_first = len(rt.tushare.calls)
    run_sync(rt, SyncParams(start="20240102", end="20240329", symbols=SYMBOLS))
    # No new tushare calls — every fetcher hits parquet/duckdb cache.
    assert len(rt.tushare.calls) == n_calls_after_first


def test_force_refresh_bypasses_caches(rt) -> None:
    run_sync(rt, SyncParams(start="20240102", end="20240329", symbols=SYMBOLS))
    n_calls_after_first = len(rt.tushare.calls)
    run_sync(
        rt,
        SyncParams(start="20240102", end="20240329", symbols=SYMBOLS,
                   force_refresh=True),
    )
    # 12 calls per refresh: trade_cal + 3 stock_basic + namechange + 5×(daily+adj_factor) + 5×daily_basic
    delta = len(rt.tushare.calls) - n_calls_after_first
    assert delta > 0, "force_refresh should re-issue API calls"
    # Sanity: status history rows should not have multiplied (INSERT OR REPLACE).
    n_rows = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_stock_status_history"
    ).fetchone()[0]
    assert n_rows <= 20  # 5 stocks × at most a few rows each


def test_sync_runs_without_explicit_symbol_subset(rt) -> None:
    """When --symbols is omitted, fall back to stock_basic L list."""
    out = run_sync(rt, SyncParams(start="20240102", end="20240329"))
    assert out.n_symbols == 5  # All 5 are listed=L in the stub
