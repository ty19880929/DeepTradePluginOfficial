"""volume-anomaly v0.5.0 — RPS / 大盘相对 alpha 单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from volume_anomaly.data import (
    ALPHA_LAGGING_THRESHOLD,
    ALPHA_LEADING_THRESHOLD,
    DEFAULT_BASELINE_INDEX_CODE,
    _build_candidate_row,
    _classify_rel_strength,
    _compute_alpha_pct,
)


def _make_history(n: int, *, start: float, daily_drift: float, end: str = "20261231") -> list[dict[str, Any]]:
    end_dt = datetime.strptime(end, "%Y%m%d")
    out = []
    close = start
    for i in range(n):
        date = (end_dt - timedelta(days=n - 1 - i)).strftime("%Y%m%d")
        new_close = close + daily_drift
        out.append(
            {
                "trade_date": date,
                "open": close,
                "high": max(close, new_close) * 1.005,
                "low": min(close, new_close) * 0.995,
                "close": new_close,
                "pct_chg": (new_close - close) / max(close, 1e-9) * 100,
                "vol": 1_000_000,
            }
        )
        close = new_close
    return out


def _make_baseline_lookup(history: list[dict[str, Any]], *, start: float, daily_drift: float) -> dict[str, float]:
    """Return a parallel baseline close-by-date map matching ``history``'s dates."""
    out: dict[str, float] = {}
    close = start
    for row in history:
        out[str(row["trade_date"])] = close
        close = close + daily_drift
    return out


# ---------------------------------------------------------------------------
# _compute_alpha_pct
# ---------------------------------------------------------------------------


class TestComputeAlphaPct:
    def test_stock_outperforms_baseline(self) -> None:
        # Stock: rises 10% over 20 days; baseline: rises 5% over 20 days
        # → alpha_20d ≈ 5%
        h = _make_history(25, start=10.0, daily_drift=0.05)
        # baseline: starts 100, drift makes it +5% after 20 steps
        # We need baseline_close on the same trade_dates as the history.
        baseline = _make_baseline_lookup(h, start=100.0, daily_drift=0.25)
        alpha = _compute_alpha_pct(h, baseline, 20)
        assert alpha is not None
        # stock_ret_20d = (h[-1].close / h[-21].close - 1) * 100
        # baseline_ret_20d = (baseline[h[-1].date] / baseline[h[-21].date] - 1) * 100
        stock_ret = (h[-1]["close"] / h[-21]["close"] - 1) * 100
        base_ret = (baseline[h[-1]["trade_date"]] / baseline[h[-21]["trade_date"]] - 1) * 100
        assert alpha == pytest.approx(stock_ret - base_ret, abs=0.01)

    def test_baseline_falls_stock_rises_strong_alpha(self) -> None:
        # Stock +5%, baseline -5% over 20 days → alpha ≈ 10%
        h = _make_history(25, start=10.0, daily_drift=0.025)  # ~+5% over 20d
        baseline = _make_baseline_lookup(h, start=100.0, daily_drift=-0.25)  # -5%
        alpha = _compute_alpha_pct(h, baseline, 20)
        assert alpha is not None
        assert alpha > 5.0

    def test_history_too_short_returns_none(self) -> None:
        h = _make_history(15, start=10.0, daily_drift=0.05)
        baseline = _make_baseline_lookup(h, start=100.0, daily_drift=0.0)
        # Need 21 rows for alpha_20d; only 15 → None
        assert _compute_alpha_pct(h, baseline, 20) is None

    def test_baseline_missing_returns_none(self) -> None:
        h = _make_history(25, start=10.0, daily_drift=0.05)
        # Empty baseline → can't compute
        assert _compute_alpha_pct(h, {}, 20) is None

    def test_baseline_missing_at_endpoint(self) -> None:
        # Baseline has data for early dates but not for the most recent → None
        h = _make_history(25, start=10.0, daily_drift=0.05)
        baseline = _make_baseline_lookup(h, start=100.0, daily_drift=0.1)
        # Drop the endpoint
        del baseline[h[-1]["trade_date"]]
        assert _compute_alpha_pct(h, baseline, 20) is None


# ---------------------------------------------------------------------------
# _classify_rel_strength
# ---------------------------------------------------------------------------


class TestClassifyRelStrength:
    def test_leading(self) -> None:
        assert _classify_rel_strength(ALPHA_LEADING_THRESHOLD + 0.1) == "leading"

    def test_in_line_at_lower_boundary(self) -> None:
        assert _classify_rel_strength(0.0) == "in_line"

    def test_in_line_at_threshold_inclusive(self) -> None:
        # threshold value itself is "in_line" (strict > / <)
        assert _classify_rel_strength(ALPHA_LEADING_THRESHOLD) == "in_line"
        assert _classify_rel_strength(ALPHA_LAGGING_THRESHOLD) == "in_line"

    def test_lagging(self) -> None:
        assert _classify_rel_strength(ALPHA_LAGGING_THRESHOLD - 0.1) == "lagging"

    def test_none_passes_through(self) -> None:
        assert _classify_rel_strength(None) is None


# ---------------------------------------------------------------------------
# _build_candidate_row — integration of alpha fields
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


class TestBuildCandidateRowAlphaFields:
    def test_emits_alpha_fields_when_baseline_present(self) -> None:
        h = _make_history(70, start=10.0, daily_drift=0.05)
        baseline = _make_baseline_lookup(h, start=100.0, daily_drift=0.10)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
            baseline_close_by_date=baseline,
        )
        assert rec["alpha_5d_pct"] is not None
        assert rec["alpha_20d_pct"] is not None
        assert rec["alpha_60d_pct"] is not None
        assert rec["baseline_index_code"] == DEFAULT_BASELINE_INDEX_CODE
        assert rec["rel_strength_label"] in {"leading", "in_line", "lagging"}

    def test_alpha_none_when_baseline_missing(self) -> None:
        h = _make_history(70, start=10.0, daily_drift=0.05)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
            baseline_close_by_date=None,
        )
        assert rec["alpha_5d_pct"] is None
        assert rec["alpha_20d_pct"] is None
        assert rec["alpha_60d_pct"] is None
        # baseline_index_code is metadata; emitted regardless
        assert rec["baseline_index_code"] == DEFAULT_BASELINE_INDEX_CODE
        # rel_strength_label is None when alpha_20d is None
        assert rec["rel_strength_label"] is None

    def test_strong_relative_strength_label_leading(self) -> None:
        # Stock rises while baseline falls → strong leading
        h = _make_history(25, start=10.0, daily_drift=0.05)
        baseline = _make_baseline_lookup(h, start=100.0, daily_drift=-0.30)
        rec = _build_candidate_row(
            watchlist_row=_watchlist_row(),
            trade_date=h[-1]["trade_date"],
            history=h,
            daily_basic={},
            moneyflow_5d=[],
            limit_up_dates=[],
            baseline_close_by_date=baseline,
        )
        assert rec["alpha_20d_pct"] is not None
        assert rec["alpha_20d_pct"] > ALPHA_LEADING_THRESHOLD
        assert rec["rel_strength_label"] == "leading"
