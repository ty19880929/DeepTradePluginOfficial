"""volume-anomaly v0.3.0 — VCP / 阻力位特征 单元测试。

只测纯函数；不接 tushare / DB / LLM。覆盖：
    * _compute_atr_series — 收敛 / 发散 / 短历史
    * _compute_bbw_series — 收敛 / 发散 / 短历史
    * _build_candidate_row 内的 ATR/BBW/阻力位字段在不同历史长度下的降级
"""

from __future__ import annotations

from typing import Any

import pytest

from volume_anomaly.data import (
    _build_candidate_row,
    _compute_atr_series,
    _compute_bbw_series,
)


# ---------------------------------------------------------------------------
# Test fixtures — synthetic OHLC generators
# ---------------------------------------------------------------------------


def _row(
    *,
    trade_date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    pct_chg: float = 0.0,
    vol: float = 1_000_000.0,
) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "pct_chg": pct_chg,
        "vol": vol,
    }


def _make_dates(n: int, *, end: str = "20261231") -> list[str]:
    """Return n consecutive YYYYMMDD strings ending at ``end`` (calendar days,
    not trade days — fine for unit tests since the date order is what matters).
    """
    from datetime import datetime, timedelta

    end_dt = datetime.strptime(end, "%Y%m%d")
    return [(end_dt - timedelta(days=n - 1 - i)).strftime("%Y%m%d") for i in range(n)]


def _make_history(
    n: int,
    *,
    start_close: float = 10.0,
    daily_range_pct: float = 2.0,
    drift: float = 0.0,
    range_decay: float = 1.0,
    end_date: str = "20261231",
) -> list[dict[str, Any]]:
    """Build synthetic history with controllable volatility profile.

    Closes oscillate ±0.5 × ``daily_range_pct`` around the trend (start_close +
    i*drift), so BBW / ATR actually respond to ``daily_range_pct``.
    ``range_decay`` shrinks the oscillation amplitude exponentially over time.
    Deterministic (no randomness).
    """
    dates = _make_dates(n, end=end_date)
    out: list[dict[str, Any]] = []
    prev_close = start_close
    for i in range(n):
        scale = max(range_decay ** i, 0.05)
        amplitude = start_close * (daily_range_pct / 100.0) * 0.5 * scale
        # Alternating oscillation around the trend so closes actually move
        sign = 1.0 if (i % 2 == 0) else -1.0
        trend_close = start_close + drift * i
        new_close = trend_close + sign * amplitude
        rng = max(abs(new_close - prev_close) * 1.5, start_close * 0.001)
        high = max(prev_close, new_close) + rng / 2
        low = min(prev_close, new_close) - rng / 2
        open_ = prev_close
        out.append(
            _row(
                trade_date=dates[i],
                open_=open_,
                high=high,
                low=low,
                close=new_close,
                pct_chg=(new_close - prev_close) / max(prev_close, 1e-9) * 100,
            )
        )
        prev_close = new_close
    return out


# ---------------------------------------------------------------------------
# _compute_atr_series
# ---------------------------------------------------------------------------


class TestComputeATRSeries:
    def test_short_history_returns_nones(self) -> None:
        # < 11 rows (need ATR_WINDOW=10 + first row TR=None) → all None at start
        h = _make_history(8)
        atr = _compute_atr_series(h)
        assert len(atr) == 8
        assert all(v is None for v in atr)

    def test_full_history_late_atr_decreases_when_range_decays(self) -> None:
        # Range decays exponentially → late ATR < early ATR
        h = _make_history(80, range_decay=0.95)
        atr = _compute_atr_series(h)
        # First non-None index = 9 (need 10 TRs starting from i=1)
        first_valid = next(i for i, v in enumerate(atr) if v is not None)
        assert first_valid >= 9
        # Compare early vs late values
        early = atr[first_valid + 5]  # somewhere near the start of valid region
        late = atr[-1]
        assert early is not None and late is not None
        assert late < early, f"late ATR should be smaller, got early={early} late={late}"

    def test_full_history_late_atr_increases_when_range_grows(self) -> None:
        # Range grows → late ATR > early ATR
        h = _make_history(80, range_decay=1.05)
        atr = _compute_atr_series(h)
        first_valid = next(i for i, v in enumerate(atr) if v is not None)
        early = atr[first_valid + 5]
        late = atr[-1]
        assert early is not None and late is not None
        assert late > early


# ---------------------------------------------------------------------------
# _compute_bbw_series
# ---------------------------------------------------------------------------


class TestComputeBBWSeries:
    def test_short_history_returns_nones(self) -> None:
        h = _make_history(15)
        bbw = _compute_bbw_series(h)
        assert all(v is None for v in bbw)

    def test_constant_close_bbw_is_zero(self) -> None:
        # close fixed at 10 across 30 rows → std=0 → BBW=0
        h = [
            _row(
                trade_date=f"2026{i:04d}",
                open_=10.0, high=10.0, low=10.0, close=10.0,
            )
            for i in range(30)
        ]
        bbw = _compute_bbw_series(h)
        # The first 19 are None; from index 19 onward BBW should be 0
        assert bbw[18] is None
        assert bbw[19] == pytest.approx(0.0)
        assert bbw[-1] == pytest.approx(0.0)

    def test_volatile_then_calm_bbw_decreases(self) -> None:
        # 15 volatile + 30 calm → late BBW < early BBW
        h_vol = _make_history(15, daily_range_pct=4.0, drift=0.1)
        h_calm = _make_history(30, daily_range_pct=0.2, drift=0.0)
        # Renumber dates so they are contiguous and unique
        dates = _make_dates(45)
        for i, r in enumerate(h_vol + h_calm):
            r["trade_date"] = dates[i]
        h = h_vol + h_calm
        bbw = _compute_bbw_series(h)
        valid = [(i, v) for i, v in enumerate(bbw) if v is not None]
        assert valid, "Should have valid BBW values"
        # Last index BBW should be smaller than the earliest valid one
        early_bbw = valid[0][1]
        late_bbw = valid[-1][1]
        assert late_bbw < early_bbw


# ---------------------------------------------------------------------------
# _build_candidate_row — full integration of new fields
# ---------------------------------------------------------------------------


def _watchlist_row() -> dict[str, Any]:
    return {
        "ts_code": "000001.SZ",
        "name": "测试",
        "industry": "测试",
        "tracked_since": "20260601",
        "last_screened": "20260601",
        "last_pct_chg": 6.5,
        "last_close": 10.0,
        "last_vol": 1_000_000,
        "last_amount": 10_000_000,
        "last_body_ratio": 0.7,
        "last_turnover_rate": 5.0,
        "last_vol_ratio_5d": 2.5,
        "last_max_vol_60d": 1_500_000,
    }


class TestBuildCandidateRowVcpFields:
    def test_full_250d_history_emits_all_new_fields(self) -> None:
        h = _make_history(250, daily_range_pct=2.0, drift=0.05)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
        )
        # P0-3
        assert rec["atr_10d_pct"] is not None
        assert rec["atr_10d_quantile_in_60d"] is not None
        assert 0 <= rec["atr_10d_quantile_in_60d"] <= 1
        assert rec["bbw_20d"] is not None
        assert rec["bbw_compression_ratio"] is not None
        # P0-4
        assert rec["high_250d"] is not None
        assert rec["high_120d"] is not None
        assert rec["low_120d"] is not None
        assert rec["dist_to_120d_high_pct"] is not None
        assert rec["dist_to_250d_high_pct"] is not None
        assert rec["pos_in_120d_range"] is not None
        assert 0 <= rec["pos_in_120d_range"] <= 1
        assert isinstance(rec["is_above_120d_high"], bool)
        assert isinstance(rec["is_above_250d_high"], bool)

    def test_150d_history_high_250d_is_none(self) -> None:
        # drift > 0 so closes vary → high_120d != low_120d → pos_in_120d_range valid
        h = _make_history(150, daily_range_pct=2.0, drift=0.02)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
        )
        assert rec["high_250d"] is None
        assert rec["dist_to_250d_high_pct"] is None
        assert rec["is_above_250d_high"] is False
        # 120d still computable
        assert rec["high_120d"] is not None
        assert rec["dist_to_120d_high_pct"] is not None
        assert rec["pos_in_120d_range"] is not None

    def test_50d_history_short_window_degrades_correctly(self) -> None:
        h = _make_history(50, daily_range_pct=2.0)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
        )
        # < 60 → quantile None; < 60 → bbw_compression_ratio None
        assert rec["atr_10d_quantile_in_60d"] is None
        assert rec["bbw_compression_ratio"] is None
        # BBW itself can be computed (≥ 20 closes)
        assert rec["bbw_20d"] is not None
        # ATR_10 itself is computable but quantile is None — so atr_10d_pct present
        assert rec["atr_10d_pct"] is not None
        # Resistance levels — both None (history < 120)
        assert rec["high_120d"] is None
        assert rec["high_250d"] is None
        assert rec["low_120d"] is None
        assert rec["dist_to_120d_high_pct"] is None

    def test_15d_history_minimal_no_exception(self) -> None:
        h = _make_history(15, daily_range_pct=2.0)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
        )
        assert rec["bbw_20d"] is None
        assert rec["atr_10d_pct"] is None or isinstance(rec["atr_10d_pct"], float)
        assert rec["high_120d"] is None
        # MAs partially computable
        assert rec["ma5"] is not None  # 15 ≥ 5
        assert rec["ma60"] is None     # 15 < 60

    def test_close_at_250d_high_is_above_flag(self) -> None:
        # Hand-craft a strictly monotonic rising close → last close IS the 250d max
        dates = _make_dates(250)
        h = [
            _row(
                trade_date=dates[i],
                open_=10.0 + i * 0.05,
                high=10.0 + i * 0.05 + 0.05,
                low=10.0 + i * 0.05 - 0.05,
                close=10.0 + i * 0.05,
                pct_chg=0.0,
            )
            for i in range(250)
        ]
        last_close = float(h[-1]["close"])
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
        )
        # last close == high_250d → dist ≈ 0; is_above_250d_high == False
        # (strict > comparison) since they're equal at this rounding.
        assert rec["high_250d"] == pytest.approx(round(last_close, 3))
        assert rec["dist_to_250d_high_pct"] == pytest.approx(0.0, abs=1e-3)
        assert rec["is_above_250d_high"] is False

    def test_volatility_collapse_low_quantile(self) -> None:
        # First 90 days highly volatile, last 30 days calm → trailing-60 BBW
        # window straddles the transition so bbw_now (calm) < mean_60d (mixed).
        first = _make_history(90, daily_range_pct=4.0, drift=0.0)
        second = _make_history(30, daily_range_pct=0.3, drift=0.0)
        # Renumber dates so they are contiguous and unique
        dates = _make_dates(120)
        for i, r in enumerate(first + second):
            r["trade_date"] = dates[i]
        h = first + second
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
        )
        # Calm tail → recent ATR_10 should be low → low quantile within 60d window
        assert rec["atr_10d_quantile_in_60d"] is not None
        # bbw_compression_ratio < 1 means current BBW is below trailing 60d mean
        assert rec["bbw_compression_ratio"] is not None
        assert rec["bbw_compression_ratio"] < 1.0
