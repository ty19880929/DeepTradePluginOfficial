"""Feature-computation tests — happy + NaN edge per group, plus schema-drift
and determinism guards (PR-2.1).

Test strategy: feed :func:`compute_features_for_symbol` synthetic qfq +
daily_basic frames whose shapes are tuned to exercise one feature group at
a time. Cross-sectional ``rs_pctile`` is exercised separately by writing
parquet fixtures and invoking :func:`compute_features_frame` against a
multi-symbol cohort.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checkmate import features, paths
from checkmate.config import FeaturesConfig
from checkmate.features import (
    FEATURE_COLUMNS,
    FeaturesRow,
    compute_features_for_symbol,
    compute_features_frame,
    compute_score,
    upsert_features_daily,
)
from checkmate.runtime import CheckmateRuntime


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _synthetic_qfq(
    *,
    n: int = 130,
    end_date: str = "20240329",
    base: float = 10.0,
    trend: float = 0.0,
    pre_close_factor: float = 0.99,  # default makes the daily pct_chg tiny
    one_way_indices: tuple[int, ...] = (),
    limit_indices: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Build a qfq daily frame with knobs for shape control.

    ``trend`` is per-session drift on the close. ``one_way_indices`` injects
    ``high==low`` days; ``limit_indices`` injects ``|pct_chg| > 10%`` days
    (used for limit_freq_60d).
    """
    dates = pd.bdate_range(end=end_date, periods=n).strftime("%Y%m%d").tolist()
    closes = np.array([base + trend * i for i in range(n)], dtype=float)
    highs = closes * 1.02
    lows = closes * 0.98
    pre = np.empty_like(closes)
    pre[0] = closes[0] / pre_close_factor
    pre[1:] = closes[:-1]
    for i in one_way_indices:
        if 0 <= i < n:
            highs[i] = closes[i]
            lows[i] = closes[i]
    for i in limit_indices:
        if 0 <= i < n:
            pre[i] = closes[i] / 1.10  # +10% pct_chg
    df = pd.DataFrame({
        "ts_code": ["TEST.SH"] * n,
        "trade_date": dates,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "pre_close": pre,
        "adj_factor": [1.0] * n,
    })
    # Mirror the "_qfq" columns that fetch_daily_qfq would attach so the
    # production path is exercised. (Raw == qfq when adj_factor is constant.)
    for col in ("open", "high", "low", "close", "pre_close"):
        df[f"{col}_qfq"] = df[col]
    return df


def _synthetic_daily_basic(
    *,
    n: int = 130,
    end_date: str = "20240329",
    amount_qianyuan: float = 100_000.0,  # 1 亿/天
    turnover_rate: float = 1.5,
) -> pd.DataFrame:
    dates = pd.bdate_range(end=end_date, periods=n).strftime("%Y%m%d").tolist()
    return pd.DataFrame({
        "ts_code": ["TEST.SH"] * n,
        "trade_date": dates,
        "amount": [amount_qianyuan] * n,
        "turnover_rate": [turnover_rate] * n,
        "total_mv": [1e10] * n,
    })


# ===========================================================================
# Schema-drift guard
# ===========================================================================


def test_feature_columns_match_migration_order() -> None:
    """If a new feature lands, FEATURE_COLUMNS + the migration MUST be edited
    together. This locks the order so accidental list mutation is caught."""
    expected = (
        "trade_date", "ts_code",
        "close_qfq",
        "ma20", "ma60", "ma120", "ma_slope60",
        "atr20", "atr_pct",
        "ret_60", "ret_120",
        "rs60_pctile", "rs120_pctile",
        "amount_20d_avg", "turnover_20d_avg", "limit_freq_60d",
        "drawdown_60d_high", "quiet_score", "above_ma20_days",
        "score",
    )
    assert FEATURE_COLUMNS == expected


def test_features_row_dataclass_default_is_none_for_every_metric() -> None:
    row = FeaturesRow(trade_date="20240329", ts_code="TEST.SH")
    for col in FEATURE_COLUMNS:
        if col in {"trade_date", "ts_code", "score"}:
            continue
        assert getattr(row, col) is None, f"{col} should default to None"


# ===========================================================================
# Trend group (ma20 / ma60 / ma120 / ma_slope60)
# ===========================================================================


def test_trend_happy_rising_series() -> None:
    qfq = _synthetic_qfq(n=130, base=10.0, trend=0.05)  # +5 fen/天
    db = _synthetic_daily_basic(n=130)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.ma20 is not None and row.ma60 is not None and row.ma120 is not None
    # On a strictly rising series, ma20 > ma60 > ma120.
    assert row.ma20 > row.ma60 > row.ma120
    assert row.ma_slope60 is not None
    assert row.ma_slope60 > 0  # positive slope


def test_trend_nan_when_history_too_short_for_ma60() -> None:
    qfq = _synthetic_qfq(n=30, base=10.0)
    db = _synthetic_daily_basic(n=30)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.ma20 is not None  # 30 ≥ 20
    assert row.ma60 is None  # 30 < 60
    assert row.ma120 is None
    assert row.ma_slope60 is None  # needs 60


# ===========================================================================
# Volatility group (atr20 / atr_pct)
# ===========================================================================


def test_volatility_atr_happy() -> None:
    qfq = _synthetic_qfq(n=60, base=10.0)
    db = _synthetic_daily_basic(n=60)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.atr20 is not None
    assert row.atr_pct is not None
    # With ±2% bands around a ~10 yuan price, ATR is ≈ 0.4 yuan → ATR% ≈ 4%.
    assert 0.02 < row.atr_pct < 0.08


def test_volatility_nan_with_insufficient_history() -> None:
    # atr_window=20 → needs 21 rows to compute the first ATR
    qfq = _synthetic_qfq(n=15, base=10.0)
    db = _synthetic_daily_basic(n=15)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.atr20 is None
    assert row.atr_pct is None


# ===========================================================================
# Strength group (ret_60 / ret_120 — per-symbol only)
# ===========================================================================


def test_strength_returns_happy() -> None:
    qfq = _synthetic_qfq(n=130, base=10.0, trend=0.05)
    db = _synthetic_daily_basic(n=130)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.ret_60 is not None and row.ret_60 > 0
    assert row.ret_120 is not None and row.ret_120 > row.ret_60  # longer = more cumulative drift


def test_strength_nan_with_short_history() -> None:
    qfq = _synthetic_qfq(n=50, base=10.0)
    db = _synthetic_daily_basic(n=50)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.ret_60 is None  # needs 61 rows
    assert row.ret_120 is None


# ===========================================================================
# Liquidity group
# ===========================================================================


def test_liquidity_happy() -> None:
    qfq = _synthetic_qfq(n=130, base=10.0)
    db = _synthetic_daily_basic(n=130, amount_qianyuan=500_000.0)  # 5亿/天
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    assert row.amount_20d_avg == pytest.approx(5e8, rel=1e-6)
    assert row.turnover_20d_avg == pytest.approx(1.5)


def test_liquidity_missing_daily_basic() -> None:
    qfq = _synthetic_qfq(n=130, base=10.0)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, pd.DataFrame())
    assert row.amount_20d_avg is None
    assert row.turnover_20d_avg is None


def test_limit_freq_counts_pct_chg_above_threshold() -> None:
    # Plant 6 limit-up days in the last 60 sessions.
    limit_idx = tuple(range(70, 76))
    qfq = _synthetic_qfq(n=130, base=10.0, limit_indices=limit_idx)
    db = _synthetic_daily_basic(n=130)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, db)
    # 6/60 of trailing window = 0.1
    assert row.limit_freq_60d is not None
    assert row.limit_freq_60d == pytest.approx(0.1, abs=1e-4)


# ===========================================================================
# Pullback group (drawdown_60d_high / quiet_score / above_ma20_days)
# ===========================================================================


def test_drawdown_when_below_recent_high() -> None:
    """Synthesise: 120 rising days then 10 days pulling back ~5%."""
    n = 130
    closes = np.array([10.0 + 0.05 * i for i in range(120)] + [16.0 - 0.05 * i for i in range(10)])
    df = pd.DataFrame({
        "ts_code": ["TEST.SH"] * n,
        "trade_date": pd.bdate_range(end="20240329", periods=n).strftime("%Y%m%d"),
        "close": closes, "high": closes * 1.01, "low": closes * 0.99,
        "open": closes, "pre_close": np.concatenate([[closes[0] * 0.99], closes[:-1]]),
        "close_qfq": closes, "high_qfq": closes * 1.01, "low_qfq": closes * 0.99,
        "open_qfq": closes, "pre_close_qfq": np.concatenate([[closes[0] * 0.99], closes[:-1]]),
        "adj_factor": [1.0] * n,
    })
    row = compute_features_for_symbol("TEST.SH", "20240329", df, _synthetic_daily_basic(n=n))
    assert row.drawdown_60d_high is not None
    assert row.drawdown_60d_high < 0  # below the peak
    # Roughly -3% to -5% pullback
    assert -0.10 < row.drawdown_60d_high < -0.01


def test_quiet_score_high_when_amplitude_small() -> None:
    qfq = _synthetic_qfq(n=60, base=10.0)
    # Tighten high/low to ±0.5% — quiet_score should be high.
    qfq["high"] = qfq["close"] * 1.005
    qfq["low"] = qfq["close"] * 0.995
    qfq["high_qfq"] = qfq["high"]
    qfq["low_qfq"] = qfq["low"]
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, _synthetic_daily_basic(n=60))
    assert row.quiet_score is not None and row.quiet_score > 80


def test_above_ma20_days_in_rising_market() -> None:
    qfq = _synthetic_qfq(n=130, base=10.0, trend=0.05)
    row = compute_features_for_symbol("TEST.SH", "20240329", qfq, _synthetic_daily_basic(n=130))
    assert row.above_ma20_days is not None
    # Strictly rising series: close is above ma20 nearly every day.
    assert row.above_ma20_days >= 50  # most of 60-day window


# ===========================================================================
# Cross-sectional rs_pctile + scoring (frame-level)
# ===========================================================================


@pytest.fixture
def rt_with_planted_caches(tmp_path, monkeypatch):
    """Frame-level helper: monkey-patch paths so the fetcher cache layer reads
    the parquet files we plant in tmp_path."""
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()

    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())
    rt = CheckmateRuntime(db=db, config=None, tushare=None)  # type: ignore[arg-type]
    yield rt, tmp_path
    db.close()


def _plant_qfq_cache(ts_code: str, df: pd.DataFrame) -> None:
    p = paths.daily_cache_dir() / f"{ts_code}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _plant_basic_cache(ts_code: str, df: pd.DataFrame) -> None:
    p = paths.daily_basic_cache_dir() / f"{ts_code}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def test_cross_sectional_pctile_ranks(rt_with_planted_caches) -> None:
    """Three stocks with distinct ret_60 → strict pctile ordering."""
    rt, _ = rt_with_planted_caches
    # n=200 ensures the planted cache window covers the ~221 cal-day request
    # window that compute_features_frame derives from FeaturesConfig defaults.
    for code, trend in (("A.SH", 0.10), ("B.SH", 0.02), ("C.SH", -0.05)):
        qf = _synthetic_qfq(n=200, base=10.0, trend=trend)
        qf["ts_code"] = code
        _plant_qfq_cache(code, qf)
        _plant_basic_cache(code, _synthetic_daily_basic(n=200))

    df, rows = compute_features_frame(rt, "20240329", ["A.SH", "B.SH", "C.SH"])
    by_code = {r.ts_code: r for r in rows}
    assert by_code["A.SH"].rs60_pctile == pytest.approx(1.0)
    assert by_code["B.SH"].rs60_pctile == pytest.approx(2.0 / 3.0)
    assert by_code["C.SH"].rs60_pctile == pytest.approx(1.0 / 3.0)
    # Scores propagate the pctile rank → A.SH > B.SH > C.SH on strength alone.
    assert by_code["A.SH"].score > by_code["B.SH"].score > by_code["C.SH"].score


def test_frame_columns_stable(rt_with_planted_caches) -> None:
    rt, _ = rt_with_planted_caches
    qf = _synthetic_qfq(n=200, base=10.0)
    qf["ts_code"] = "X.SH"
    _plant_qfq_cache("X.SH", qf)
    _plant_basic_cache("X.SH", _synthetic_daily_basic(n=200))
    df, _ = compute_features_frame(rt, "20240329", ["X.SH"])
    assert list(df.columns) == list(FEATURE_COLUMNS)


def test_features_frame_deterministic(rt_with_planted_caches) -> None:
    rt, _ = rt_with_planted_caches
    for code, trend in (("D.SH", 0.05), ("E.SH", 0.01)):
        qf = _synthetic_qfq(n=200, base=10.0, trend=trend)
        qf["ts_code"] = code
        _plant_qfq_cache(code, qf)
        _plant_basic_cache(code, _synthetic_daily_basic(n=200))

    df1, rows1 = compute_features_frame(rt, "20240329", ["D.SH", "E.SH"])
    df2, rows2 = compute_features_frame(rt, "20240329", ["D.SH", "E.SH"])
    pd.testing.assert_frame_equal(df1, df2)
    # Score breakdown JSON-serialisable & equal
    assert [r.score_breakdown for r in rows1] == [r.score_breakdown for r in rows2]


# ===========================================================================
# compute_score sanity
# ===========================================================================


def test_score_weights_sum_to_one() -> None:
    cfg = FeaturesConfig()
    total = (
        cfg.score_weight_trend + cfg.score_weight_volatility +
        cfg.score_weight_strength + cfg.score_weight_liquidity +
        cfg.score_weight_pullback
    )
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_score_breakdown_includes_all_components() -> None:
    row = FeaturesRow(trade_date="20240329", ts_code="TEST.SH",
                      close_qfq=10.0, ma20=9.5, ma60=9.0, ma120=8.5,
                      atr20=0.25, atr_pct=0.025,
                      ret_60=10.0, ret_120=18.0,
                      rs60_pctile=0.8, rs120_pctile=0.75,
                      amount_20d_avg=2e9, turnover_20d_avg=2.5,
                      drawdown_60d_high=-0.03, quiet_score=70.0,
                      above_ma20_days=55, limit_freq_60d=0.05)
    score, bd = compute_score(row)
    assert 0.0 <= score <= 100.0
    assert set(bd["components"]) == {"trend", "volatility", "strength", "liquidity", "pullback"}
    assert set(bd["weights"]) == set(bd["components"])


# ===========================================================================
# Persistence
# ===========================================================================


def test_upsert_features_daily_writes_score_breakdown_as_json(rt_with_planted_caches) -> None:
    import json

    rt, _ = rt_with_planted_caches
    qf = _synthetic_qfq(n=200, base=10.0, trend=0.05)
    qf["ts_code"] = "Z.SH"
    _plant_qfq_cache("Z.SH", qf)
    _plant_basic_cache("Z.SH", _synthetic_daily_basic(n=200))
    _, rows = compute_features_frame(rt, "20240329", ["Z.SH"])
    n = upsert_features_daily(rt.db, rows)
    assert n == 1
    persisted = rt.db.execute(
        "SELECT score, score_breakdown FROM checkmate_features_daily WHERE ts_code = ?",
        ["Z.SH"],
    ).fetchone()
    assert persisted is not None
    score_db, bd_json = persisted
    assert score_db == pytest.approx(rows[0].score)
    assert json.loads(bd_json)["weights"]["trend"] == pytest.approx(0.25)
