"""v0.4.0 — derived-feature helpers in ``data.py``.

Tests for compute_vcp_features / compute_long_range_features /
compute_alpha_features / compute_volume_event_score / compute_ma_distances.
Each helper is self-contained and pure-pandas, so the tests stay hermetic
(no Tushare mock needed).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from accumulation_probe_washout.data import (
    compute_alpha_features,
    compute_long_range_features,
    compute_ma_distances,
    compute_vcp_features,
    compute_volume_event_score,
)
from tests.conftest import make_quotes


# ---------------------------------------------------------------------------
# VCP — ATR / BBW
# ---------------------------------------------------------------------------


def test_vcp_returns_nones_when_history_too_short():
    df = make_quotes(n=8)
    out = compute_vcp_features(df)
    assert out["atr_10d"] is None
    assert out["bbw_20d"] is None


def test_vcp_returns_finite_values_on_full_window():
    df = make_quotes(n=130, pattern="flat")
    out = compute_vcp_features(df)
    assert out["atr_10d"] is not None and out["atr_10d"] > 0
    assert out["atr_10d_pct"] is not None and out["atr_10d_pct"] > 0
    assert out["bbw_20d"] is not None and out["bbw_20d"] > 0
    # Compression ratio close to 1.0 for a flat-volatility series.
    assert out["bbw_compression_ratio"] is not None
    assert 0.3 <= out["bbw_compression_ratio"] <= 3.0


def test_vcp_atr_quantile_bounded_0_1():
    df = make_quotes(n=130, pattern="uptrend")
    out = compute_vcp_features(df)
    q = out["atr_10d_quantile_in_60d"]
    assert q is not None
    assert 0.0 <= q <= 1.0


# ---------------------------------------------------------------------------
# Long-range resistance — 120/250d distances
# ---------------------------------------------------------------------------


def test_long_range_returns_none_when_history_below_120():
    df = make_quotes(n=80)
    out = compute_long_range_features(df)
    assert out["dist_to_120d_high_pct"] is None
    assert out["dist_to_250d_high_pct"] is None


def test_long_range_120d_only_when_under_250d_history():
    df = make_quotes(n=200, pattern="flat")
    out = compute_long_range_features(df)
    assert out["dist_to_120d_high_pct"] is not None
    assert out["dist_to_250d_high_pct"] is None
    # Flat series ends near the high; pos_in_120d_range ∈ [0, 1]
    assert 0.0 <= out["pos_in_120d_range"] <= 1.0


def test_long_range_breakout_above_120d_high_flag():
    df = make_quotes(n=200, pattern="flat")
    # Patch the last row to exceed the 120d high so the flag must flip True.
    df.at[len(df) - 1, "close"] = df["high"].iloc[-120:].max() * 1.2
    df.at[len(df) - 1, "high"] = df.at[len(df) - 1, "close"] * 1.02
    out = compute_long_range_features(df)
    assert out["is_above_120d_high"] is True


# ---------------------------------------------------------------------------
# Alpha — multi-horizon excess return vs baseline
# ---------------------------------------------------------------------------


def _make_index(n: int, *, daily_pct: float = 0.001) -> pd.DataFrame:
    base = 3000.0
    rows = []
    for i in range(n):
        close = base * (1.0 + daily_pct * i)
        rows.append(
            {
                "trade_date": (
                    pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                ).strftime("%Y%m%d"),
                "close": close,
            }
        )
    return pd.DataFrame(rows)


def test_alpha_none_when_index_missing():
    df = make_quotes(n=80)
    out = compute_alpha_features(df, None)
    assert all(v is None for v in out.values())


def test_alpha_5d_positive_when_stock_outperforms():
    stock = make_quotes(n=80, pattern="uptrend")  # +0.5%/day
    idx = _make_index(80, daily_pct=0.0)  # flat
    out = compute_alpha_features(stock, idx)
    assert out["alpha_5d_pct"] is not None
    assert out["alpha_5d_pct"] > 0
    assert out["alpha_20d_pct"] is not None
    assert out["alpha_20d_pct"] > out["alpha_5d_pct"]  # longer horizon → larger alpha


def test_alpha_leading_label_threshold_5_pct():
    stock = make_quotes(n=80, pattern="uptrend")
    idx = _make_index(80, daily_pct=0.0)
    out = compute_alpha_features(stock, idx)
    assert out["alpha_leading"] in ("LEADING", "NEUTRAL", "LAGGING")
    if out["alpha_20d_pct"] is not None and out["alpha_20d_pct"] >= 5.0:
        assert out["alpha_leading"] == "LEADING"


# ---------------------------------------------------------------------------
# MA distances
# ---------------------------------------------------------------------------


def test_ma_distances_emits_ma60_value_for_prune_rule():
    """v0.3.0 prune rule needs `ma60` on the candidate JSON."""
    df = make_quotes(n=80, pattern="flat")
    out = compute_ma_distances(df)
    assert out["ma60"] is not None
    assert out["close_to_ma60_pct"] is not None
    # MA60 of a flat series ≈ start price.
    assert abs(out["ma60"] - df["close"].iloc[-1]) / out["ma60"] < 0.1


def test_ma_distances_omits_long_window_when_short_history():
    df = make_quotes(n=40)
    out = compute_ma_distances(df)
    assert out["ma60"] is None
    assert out["close_to_ma60_pct"] is None
    assert out["ma20"] is not None


# ---------------------------------------------------------------------------
# Volume event score — VA T-day rating downgraded to auxiliary feature
# ---------------------------------------------------------------------------


def test_volume_event_score_none_on_thin_history():
    df = make_quotes(n=3)
    assert compute_volume_event_score(df) is None


def test_volume_event_score_increases_with_volume_spike():
    """Higher T-day volume / amplitude → higher rating."""
    quiet = make_quotes(n=20, pattern="flat")
    spiky = make_quotes(n=20, pattern="flat", probe_index=19, probe_multiplier=4.0)
    score_quiet = compute_volume_event_score(quiet)
    score_spike = compute_volume_event_score(spiky)
    assert score_quiet is not None
    assert score_spike is not None
    assert score_spike > score_quiet


def test_volume_event_score_bounded_0_100():
    df = make_quotes(n=20, probe_index=19, probe_multiplier=10.0)
    s = compute_volume_event_score(df)
    assert s is not None
    assert 0.0 <= s <= 100.0


# ---------------------------------------------------------------------------
# Integration — pack_candidate threads everything together
# ---------------------------------------------------------------------------


def test_pack_candidate_emits_all_v04_keys():
    """Smoke check: pack_candidate with the new bundles populates every
    v0.4.0 key, even when individual helpers returned None."""
    from accumulation_probe_washout.data import pack_candidate
    from accumulation_probe_washout.schemas import APWPhase

    cand = pack_candidate(
        trade_date="20260520",
        ts_code="600000.SH",
        name="测试",
        phase=APWPhase.LAUNCH_READY,
        basic={"close": 10.0, "amount_yi": 5.0},
        accumulation={"accumulation_score": 70.0},
        probe={"probe_quality_score": 80.0, "probe_low": 9.0, "probe_high": 11.0},
        washout={"washout_score": 60.0},
        launch={"launch_setup_score": 75.0},
        vcp={"atr_10d_pct": 2.5, "bbw_20d": 0.05},
        long_range={"dist_to_120d_high_pct": -3.0, "is_above_120d_high": False},
        alpha={"alpha_20d_pct": 6.0, "alpha_leading": "LEADING"},
        ma_distances={"ma60": 9.5, "close_to_ma60_pct": 5.3},
        volume_event_score=78.0,
    )
    # Every v0.4.0 key surfaces on the candidate dict.
    for k in (
        "atr_10d_pct", "bbw_20d", "dist_to_120d_high_pct", "is_above_120d_high",
        "alpha_20d_pct", "alpha_leading", "ma60", "close_to_ma60_pct",
        "volume_event_score", "probe_low", "probe_high",
    ):
        assert k in cand, k
    assert cand["alpha_leading"] == "LEADING"
    assert cand["probe_low"] == 9.0
