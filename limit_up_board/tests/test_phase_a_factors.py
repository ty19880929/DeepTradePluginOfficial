"""Phase A — derived-factor unit tests.

Covers:
    A1 — amplitude_pct / fd_amount_ratio / ma_metrics / up_count_30d
    A2 — limit_step_trend / yesterday_failure_rate / yesterday_winners_today

Pure-function tests; no tushare / DB / LLM in scope.
"""

from __future__ import annotations

import pandas as pd

from limit_up_board.data import (
    _amplitude_pct,
    _fd_amount_ratio,
    _limit_step_trend,
    _ma_metrics,
    _max_height,
    _trailing_closes,
    _up_count_30d,
    _yesterday_failure_rate,
    _yesterday_winners_today,
)

# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------


class TestAmplitudePct:
    def test_normal(self) -> None:
        row = {"high": 11.0, "low": 9.5, "pre_close": 10.0}
        assert _amplitude_pct(row) == 15.0

    def test_returns_none_when_pre_close_zero(self) -> None:
        assert _amplitude_pct({"high": 1.0, "low": 0.5, "pre_close": 0}) is None

    def test_returns_none_on_missing(self) -> None:
        assert _amplitude_pct(None) is None
        assert _amplitude_pct({"high": 1.0, "low": None, "pre_close": 1.0}) is None

    def test_returns_none_on_nan(self) -> None:
        assert _amplitude_pct({"high": float("nan"), "low": 0.5, "pre_close": 1.0}) is None


class TestFdAmountRatio:
    def test_normal(self) -> None:
        # 5e7 / 5e8 → 10.0%
        assert _fd_amount_ratio(5e7, 5e8) == 10.0

    def test_zero_amount(self) -> None:
        assert _fd_amount_ratio(1e7, 0) is None

    def test_none(self) -> None:
        assert _fd_amount_ratio(None, 1e8) is None
        assert _fd_amount_ratio(1e7, None) is None


class TestMaMetrics:
    def test_full_history_bull(self) -> None:
        # 25 ascending closes — bull pattern
        closes = [10.0 + i * 0.1 for i in range(25)]
        out = _ma_metrics(closes)
        assert out["ma5"] is not None
        assert out["ma10"] is not None
        assert out["ma20"] is not None
        assert out["ma_bull_aligned"] is True

    def test_full_history_not_bull(self) -> None:
        # Latest close < ma5 → not aligned
        closes = [10.0] * 24 + [9.0]
        out = _ma_metrics(closes)
        assert out["ma_bull_aligned"] is False

    def test_short_history_partial_null(self) -> None:
        closes = [10.0] * 8
        out = _ma_metrics(closes)
        assert out["ma5"] == 10.0
        assert out["ma10"] is None
        assert out["ma20"] is None
        assert out["ma_bull_aligned"] is None

    def test_empty(self) -> None:
        out = _ma_metrics([])
        assert all(v is None for v in out.values())


class TestUpCount30d:
    def test_under_30_days_returns_none(self) -> None:
        rows = [{"pct_chg": 10.0}] * 20
        assert _up_count_30d(rows) is None

    def test_counts_only_last_30(self) -> None:
        # Older 5 rows of 10% are outside the 30-day window
        rows = [{"pct_chg": 10.0}] * 5 + [{"pct_chg": 0.0}] * 28 + [{"pct_chg": 10.0}] * 2
        assert _up_count_30d(rows) == 2

    def test_threshold_at_9_8(self) -> None:
        rows = [{"pct_chg": 9.8}] * 30
        assert _up_count_30d(rows) == 30

    def test_handles_none(self) -> None:
        rows = [{"pct_chg": None}] * 30
        assert _up_count_30d(rows) == 0


class TestTrailingCloses:
    def test_drops_nans(self) -> None:
        rows = [{"close": 1.0}, {"close": None}, {"close": 2.0}]
        assert _trailing_closes(rows) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# A2
# ---------------------------------------------------------------------------


class TestMaxHeight:
    def test_basic(self) -> None:
        assert _max_height({"1": 10, "2": 5, "4": 1}) == 4

    def test_empty(self) -> None:
        assert _max_height({}) == 0

    def test_skips_non_int_keys(self) -> None:
        assert _max_height({"1": 10, "abc": 5, "3": 2}) == 3


class TestLimitStepTrend:
    def test_lifting(self) -> None:
        today = {"1": 50, "2": 20, "4": 3}
        prev = {"1": 30, "2": 10, "3": 1}
        out = _limit_step_trend(today, prev)
        assert out["interpretation"] == "spectrum_lifting"
        assert out["high_board_delta"] == 1
        assert out["total_limit_up_delta"] > 0

    def test_collapsing_high_drop(self) -> None:
        today = {"1": 30, "2": 5}
        prev = {"1": 40, "2": 10, "5": 2}
        out = _limit_step_trend(today, prev)
        assert out["interpretation"] == "spectrum_collapsing"
        assert out["high_board_delta"] < 0

    def test_collapsing_total_drop_past_threshold(self) -> None:
        today = {"1": 10, "2": 2}
        prev = {"1": 30, "2": 5}  # delta = -23
        out = _limit_step_trend(today, prev)
        assert out["interpretation"] == "spectrum_collapsing"

    def test_stable(self) -> None:
        today = {"1": 25, "2": 5}
        prev = {"1": 25, "2": 5}
        out = _limit_step_trend(today, prev)
        assert out["interpretation"] == "stable"


class TestYesterdayFailureRate:
    def test_high(self) -> None:
        df = pd.DataFrame({"limit": ["U"] * 60 + ["Z"] * 25, "ts_code": ["x"] * 85})
        out = _yesterday_failure_rate("20260506", df)
        # 25 / 85 ≈ 29.41 → high
        assert out["interpretation"] == "high"
        assert out["u_count"] == 60
        assert out["z_count"] == 25

    def test_low(self) -> None:
        df = pd.DataFrame({"limit": ["U"] * 95 + ["Z"] * 5, "ts_code": ["x"] * 100})
        out = _yesterday_failure_rate("20260506", df)
        assert out["interpretation"] == "low"

    def test_moderate(self) -> None:
        df = pd.DataFrame({"limit": ["U"] * 80 + ["Z"] * 15, "ts_code": ["x"] * 95})
        out = _yesterday_failure_rate("20260506", df)
        # 15/95 ≈ 15.79 → moderate
        assert out["interpretation"] == "moderate"

    def test_empty(self) -> None:
        out = _yesterday_failure_rate("20260506", pd.DataFrame())
        assert out["interpretation"] is None
        assert out["u_count"] == 0
        assert out["z_count"] == 0
        assert out["rate_pct"] is None

    def test_none(self) -> None:
        out = _yesterday_failure_rate("20260506", None)
        assert out["interpretation"] is None


class TestYesterdayWinnersToday:
    def test_strong_money_effect(self) -> None:
        prev = pd.DataFrame(
            {
                "limit": ["U"] * 10,
                "ts_code": [f"00000{i}.SZ" for i in range(10)],
            }
        )
        # 6 of them limit-up today (≥ 9.8); avg roughly 5%
        today = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(10)],
                "pct_chg": [10.0] * 6 + [3.0] * 4,
            }
        )
        out = _yesterday_winners_today("20260506", prev, today)
        assert out["interpretation"] == "strong_money_effect"
        assert out["n_continued_today"] == 6
        assert out["continuation_rate_pct"] == 60.0

    def test_weak_money_effect_negative_avg(self) -> None:
        prev = pd.DataFrame(
            {"limit": ["U"] * 10, "ts_code": [f"00000{i}.SZ" for i in range(10)]}
        )
        today = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(10)],
                "pct_chg": [-5.0] * 10,
            }
        )
        out = _yesterday_winners_today("20260506", prev, today)
        assert out["interpretation"] == "weak_money_effect"

    def test_neutral(self) -> None:
        prev = pd.DataFrame(
            {"limit": ["U"] * 10, "ts_code": [f"00000{i}.SZ" for i in range(10)]}
        )
        today = pd.DataFrame(
            {
                "ts_code": [f"00000{i}.SZ" for i in range(10)],
                "pct_chg": [3.0] * 4 + [1.0] * 6,
            }
        )
        out = _yesterday_winners_today("20260506", prev, today)
        # cont 0%, avg 1.8 → cont ≤25 → weak (not neutral)
        assert out["interpretation"] == "weak_money_effect"

    def test_no_yesterday_winners(self) -> None:
        prev = pd.DataFrame({"limit": ["Z"] * 5, "ts_code": ["x"] * 5})
        today = pd.DataFrame({"ts_code": ["x"], "pct_chg": [1.0]})
        out = _yesterday_winners_today("20260506", prev, today)
        assert out["n_winners"] == 0
        assert out["interpretation"] is None

    def test_today_daily_missing(self) -> None:
        prev = pd.DataFrame({"limit": ["U"], "ts_code": ["000001.SZ"]})
        out = _yesterday_winners_today("20260506", prev, pd.DataFrame())
        assert out["n_winners"] == 1
        assert out["n_continued_today"] == 0
        assert out["interpretation"] is None
