"""End-to-end smoke test for ``scan`` orchestrator (PR-2.3).

Fixture: 10 synthetic ts_codes × 250 trading days, mix of trend strengths so
the cross-sectional ``rs_pctile`` has signal. Assertions:
  * All three daily tables (universe / features / regime) carry rows for
    ``trade_date``.
  * ``score`` is in [0, 100] for every eligible features row.
  * ``checkmate_runs`` has exactly one new row with mode='scan' and
    status='success'.
  * ``checkmate_events`` has the expected step events.
  * Repeat run for the same trade_date does NOT raise (INSERT OR REPLACE PK
    behaviour) and writes a fresh run_id without touching the daily rows
    beyond replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkmate import paths
from checkmate.config import (
    FeaturesConfig,
    RegimeConfig,
    UniverseConfig,
)
from checkmate.runtime import CheckmateRuntime
from checkmate.scan import ScanParams, run_scan


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# 10 synthetic ts_codes with assorted starting prices and trend slopes.
SYMBOLS = [
    ("600000.SH", 8.5,  0.005),
    ("600519.SH", 1600.0, 0.50),   # high-priced
    ("000001.SZ", 12.0, 0.010),
    ("002415.SZ", 30.0, 0.020),
    ("300750.SZ", 200.0, 0.10),
    ("600036.SH", 35.0, -0.005),   # mild downtrend
    ("000333.SZ", 60.0, 0.015),
    ("000651.SZ", 40.0, 0.000),    # flat
    ("600276.SH", 50.0, 0.030),
    ("600887.SH", 28.0, -0.010),
]
END_DATE = "20240329"
N_DAYS = 250


# ---------------------------------------------------------------------------
# Stub Tushare — never called when caches are planted, but used for trade_cal.
# ---------------------------------------------------------------------------


def _trade_cal_frame() -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    return pd.DataFrame({"cal_date": dates, "is_open": [1] * N_DAYS})


class _StubTushare:
    def call(self, api_name: str, **kwargs):
        if api_name == "trade_cal":
            return _trade_cal_frame()
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Plant helpers
# ---------------------------------------------------------------------------


def _make_qfq(ts_code: str, base: float, trend: float) -> pd.DataFrame:
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    closes = np.array([base + trend * i for i in range(N_DAYS)], dtype=float)
    # Inject mild noise so MA/ATR aren't degenerate.
    noise = np.linspace(-0.005, 0.005, N_DAYS) * base
    closes = closes + noise
    highs = closes * 1.015
    lows = closes * 0.985
    pre = np.empty_like(closes)
    pre[0] = closes[0] * 0.99
    pre[1:] = closes[:-1]
    df = pd.DataFrame({
        "ts_code": [ts_code] * N_DAYS,
        "trade_date": dates,
        "open": closes, "high": highs, "low": lows,
        "close": closes, "pre_close": pre,
        "adj_factor": [1.0] * N_DAYS,
    })
    for col in ("open", "high", "low", "close", "pre_close"):
        df[f"{col}_qfq"] = df[col]
    return df


def _make_daily_basic(ts_code: str, amount_qianyuan: float = 500_000.0) -> pd.DataFrame:
    """5 亿/天 default → liquidity check passes for every symbol."""
    dates = pd.bdate_range(end=END_DATE, periods=N_DAYS).strftime("%Y%m%d").tolist()
    return pd.DataFrame({
        "ts_code": [ts_code] * N_DAYS,
        "trade_date": dates,
        "amount": [amount_qianyuan] * N_DAYS,
        "turnover_rate": [2.0] * N_DAYS,
        "total_mv": [1e10] * N_DAYS,
    })


def _plant_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _plant_index_rising(index_code: str) -> None:
    """Rising index → close > MA(120). Reused by both CSI and HS300."""
    n = 200
    dates = pd.bdate_range(end=END_DATE, periods=n).strftime("%Y%m%d").tolist()
    closes = [1000.0 * (1.0 + 0.002 * i) for i in range(n)]
    df = pd.DataFrame({
        "ts_code": [index_code] * n,
        "trade_date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
    })
    _plant_parquet(paths.index_daily_cache_dir() / f"{index_code}.parquet", df)


def _seed_status(db, ts_code: str, list_date: str = "20100101") -> None:
    db.execute(
        """
        INSERT INTO checkmate_stock_status_history
            (ts_code, as_of_date, list_status, is_st, name, industry,
             list_date, delist_date, raw_event_json, updated_at)
        VALUES (?, ?, 'L', FALSE, ?, '行业', ?, NULL,
                ?, CURRENT_TIMESTAMP)
        """,
        [ts_code, list_date, ts_code, list_date, json.dumps({"src": "test"})],
    )


# ---------------------------------------------------------------------------
# Fixture
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

    # Plant trade_cal + index caches + per-symbol caches + status rows
    for code, base, trend in SYMBOLS:
        _plant_parquet(paths.daily_cache_dir() / f"{code}.parquet",
                       _make_qfq(code, base, trend))
        _plant_parquet(paths.daily_basic_cache_dir() / f"{code}.parquet",
                       _make_daily_basic(code))
        _seed_status(db, code)
    _plant_index_rising(RegimeConfig().index_csi_code)
    _plant_index_rising(RegimeConfig().index_hs300_code)

    rt = CheckmateRuntime(db=db, config=None, tushare=_StubTushare())  # type: ignore[arg-type]
    yield rt
    db.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scan_e2e_writes_all_three_tables(rt) -> None:
    log: list[str] = []
    outcome = run_scan(rt, ScanParams(trade_date=END_DATE), echo=log.append)

    # Outcome sanity
    assert outcome.status == "success"
    assert outcome.trade_date == END_DATE
    assert outcome.n_universe == len(SYMBOLS)
    assert outcome.n_eligible == len(SYMBOLS)  # all default-eligible
    assert outcome.n_features == len(SYMBOLS)
    assert outcome.regime in {"strong", "neutral", "weak", "risk"}

    # universe table
    universe_n = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_universe_daily WHERE trade_date = ?",
        [END_DATE],
    ).fetchone()[0]
    assert universe_n == len(SYMBOLS)

    # features table — every row has a score in [0, 100]
    feat_rows = rt.db.execute(
        "SELECT ts_code, score FROM checkmate_features_daily WHERE trade_date = ?",
        [END_DATE],
    ).fetchall()
    assert len(feat_rows) == len(SYMBOLS)
    for ts_code, score in feat_rows:
        assert score is not None, f"{ts_code} missing score"
        assert 0.0 <= score <= 100.0, f"{ts_code} score out of range: {score}"

    # regime table
    regime = rt.db.execute(
        "SELECT regime, exposure_cap FROM checkmate_regime_daily WHERE trade_date = ?",
        [END_DATE],
    ).fetchone()
    assert regime is not None
    assert regime[0] == outcome.regime
    assert regime[1] == outcome.exposure_cap

    # runs / events
    run_row = rt.db.execute(
        "SELECT mode, status, exit_code FROM checkmate_runs WHERE run_id = ?",
        [outcome.run_id],
    ).fetchone()
    assert run_row == ("scan", "success", 0)
    n_events = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_events WHERE run_id = ?",
        [outcome.run_id],
    ).fetchone()[0]
    # RUN_STARTED + 3×(STEP_STARTED + STEP_FINISHED) + RUN_FINISHED = 8
    assert n_events == 8


def test_scan_legacy_stream_emits_step_lines(rt) -> None:
    log: list[str] = []
    run_scan(rt, ScanParams(trade_date=END_DATE), echo=log.append)
    joined = "\n".join(log)
    assert "[STEP_STARTED]" in joined
    assert "[STEP_FINISHED]" in joined
    assert "Step 0" in joined  # universe
    assert "Step 1" in joined  # features
    assert "Step 2" in joined  # regime


def test_scan_repeat_same_date_does_not_conflict(rt) -> None:
    """PR-2.3 spec: 重复跑同日 PK 不冲突（INSERT OR REPLACE）."""
    first = run_scan(rt, ScanParams(trade_date=END_DATE))
    second = run_scan(rt, ScanParams(trade_date=END_DATE))
    assert first.run_id != second.run_id

    # universe rows still equal to symbol count (replaced, not duplicated)
    n_universe = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_universe_daily WHERE trade_date = ?",
        [END_DATE],
    ).fetchone()[0]
    assert n_universe == len(SYMBOLS)

    # Two run rows present
    n_runs = rt.db.execute(
        "SELECT COUNT(*) FROM checkmate_runs WHERE mode = 'scan'"
    ).fetchone()[0]
    assert n_runs == 2


def test_top_scored_payload_sorted_desc(rt) -> None:
    outcome = run_scan(rt, ScanParams(trade_date=END_DATE))
    scores = [r["score"] for r in outcome.top_scored if r["score"] is not None]
    assert scores == sorted(scores, reverse=True)
    assert len(scores) <= 30
