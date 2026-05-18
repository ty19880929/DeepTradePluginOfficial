"""compute_returns_and_labels — T5.2."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.data import compute_returns_and_labels


def _quotes(closes: list[float], highs: list[float] | None = None,
            lows: list[float] | None = None, code: str = "600000.SH") -> pd.DataFrame:
    n = len(closes)
    highs = highs or [c * 1.02 for c in closes]
    lows = lows or [c * 0.98 for c in closes]
    return pd.DataFrame([
        {"ts_code": code,
         "trade_date": pd.Timestamp("2026-01-01").strftime("%Y%m%d") if i == 0
                        else (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).strftime("%Y%m%d"),
         "close": closes[i], "high": highs[i], "low": lows[i]}
        for i in range(n)
    ])


class TestComputeReturnsAndLabels:
    def test_strong_uptrend_labels_t5_one(self):
        closes = [10.0, 10.5, 11.0, 11.5, 11.7, 11.9, 12.1, 12.0, 11.8, 11.5, 11.3]
        df = _quotes(closes)
        result = compute_returns_and_labels(
            ts_code="600000.SH", signal_date="20260101", quotes=df,
            label_t5_high_return_pct=8, label_t5_max_drawdown_pct=8,
        )
        assert result["data_status"] == "complete"
        # T+5 close = 11.9 → 19% return; max_high in T+1..5 ~12% above T
        assert result["ret_t5_pct"] is not None
        assert result["max_high_t5_pct"] > 8.0
        assert result["label_launch_t5"] == 1

    def test_high_return_but_big_drawdown_labels_zero(self):
        # High enough but with severe intraday drawdown (low breaks -10%)
        closes = [10.0, 10.5, 11.0, 11.5, 11.7, 11.9, 12.1, 12.0, 11.8, 11.5, 11.3]
        lows = [10.0, 8.9, 8.8, 8.7, 11.0, 11.5, 11.8, 11.9, 11.6, 11.4, 11.2]
        df = _quotes(closes, lows=lows)
        result = compute_returns_and_labels(
            ts_code="600000.SH", signal_date="20260101", quotes=df,
            label_t5_high_return_pct=8, label_t5_max_drawdown_pct=8,
        )
        # max_drawdown_t5 = (10 - 8.7) / 10 * 100 = 13% > 8 → label 0
        assert result["label_launch_t5"] == 0

    def test_low_return_labels_zero(self):
        closes = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        df = _quotes(closes)
        result = compute_returns_and_labels(
            ts_code="600000.SH", signal_date="20260101", quotes=df,
        )
        assert result["label_launch_t5"] == 0
        assert result["label_launch_t10"] == 0

    def test_partial_when_t10_missing(self):
        # Only 6 trading days — T+10 not reached
        closes = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5]
        df = _quotes(closes)
        result = compute_returns_and_labels(
            ts_code="600000.SH", signal_date="20260101", quotes=df,
        )
        assert result["data_status"] == "partial"
        assert result["close_t10"] is None
        assert result["label_launch_t10"] is None

    def test_missing_data_when_no_rows(self):
        result = compute_returns_and_labels(
            ts_code="600000.SH", signal_date="20260101", quotes=pd.DataFrame(),
        )
        assert result["data_status"] == "missing"
