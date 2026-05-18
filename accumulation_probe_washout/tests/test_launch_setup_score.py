"""compute_launch_setup — T2.7."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import compute_launch_setup, compute_washout, detect_probe_day
from tests.conftest import make_quotes


class TestLaunchSetupScore:
    def test_above_mas_and_volume_pickup_scores_high(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        # Healthy washout + last-day pickup
        for i in range(111, 129):
            qdf.at[i, "close"] = qdf.at[110, "close"] * 0.97
            qdf.at[i, "high"] = qdf.at[i, "close"] * 1.015
            qdf.at[i, "low"] = qdf.at[i, "close"] * 0.985
            qdf.at[i, "vol"] = qdf.at[110, "vol"] * 0.4
        # Last day: heavy pickup above all MAs
        qdf.at[129, "close"] = qdf.at[110, "close"] * 1.02
        qdf.at[129, "high"] = qdf.at[129, "close"] * 1.03
        qdf.at[129, "low"] = qdf.at[128, "close"] * 1.00
        qdf.at[129, "vol"] = qdf.at[110, "vol"] * 0.9  # vs the small avg of recent: big ratio

        probe = detect_probe_day(qdf, default_cfg)
        wash = compute_washout(qdf, pd.DataFrame(), probe, default_cfg)
        result = compute_launch_setup(qdf, probe, wash, default_cfg)
        assert result["above_ma5"] is True
        # Most key checks
        assert result["launch_setup_score"] >= 50.0
        assert result["current_volume_ratio_5d"] > 1.0

    def test_below_mas_scores_low(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="downtrend", n=130)
        wash = {"washout_days": 0}
        result = compute_launch_setup(qdf, None, wash, default_cfg)
        # In a downtrend the last close sits below mid-window MAs.
        assert result["above_ma20"] is False
        assert result["launch_setup_score"] < 60.0

    def test_empty_returns_zero(self, default_cfg: ApwConfig) -> None:
        wash = {"washout_days": 0}
        result = compute_launch_setup(pd.DataFrame(), None, wash, default_cfg)
        assert result["launch_setup_score"] == 0.0


class TestRelativeStrength:
    """P2-1 regression — relative_strength_20d must subtract the index baseline."""

    def _make_index_df(self, n: int, return_pct: float) -> pd.DataFrame:
        """Build an index_daily frame that drifts ``return_pct`` over its window."""
        base_date = pd.Timestamp("2024-01-01")
        rows = []
        start_close = 4000.0
        end_close = start_close * (1.0 + return_pct / 100.0)
        for i in range(n):
            frac = i / max(1, n - 1)
            close = start_close + (end_close - start_close) * frac
            rows.append({
                "ts_code": "000300.SH",
                "trade_date": (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d"),
                "close": round(close, 3),
            })
        return pd.DataFrame(rows)

    def test_uses_index_baseline_when_provided(self, default_cfg: ApwConfig) -> None:
        """Stock outperforming the index by ~5pp → rs20 ≈ +5."""
        qdf = make_quotes(pattern="uptrend", n=130)  # stock returns ~+15% over 20d
        # Index outperforms stock by 3% over the same window — drag rs20 down.
        idx_df = self._make_index_df(n=130, return_pct=200.0)
        wash = {"washout_days": 0}
        result = compute_launch_setup(qdf, None, wash, default_cfg, index_df=idx_df)
        assert result["relative_strength_20d"] is not None
        # Index 20d return ≈ 200% * 19/129 ≈ 29.5; stock 20d ≈ +10. rs20 should
        # be strongly negative — proving the subtraction actually happened.
        assert result["relative_strength_20d"] < 0.0

    def test_marks_missing_when_index_unavailable(self, default_cfg: ApwConfig) -> None:
        """Without index_df rs20 is None — no silent fallback to absolute return."""
        qdf = make_quotes(pattern="uptrend", n=130)
        wash = {"washout_days": 0}
        result = compute_launch_setup(qdf, None, wash, default_cfg, index_df=None)
        assert result["relative_strength_20d"] is None

    def test_marks_missing_when_index_too_short(self, default_cfg: ApwConfig) -> None:
        """Index frame with < 21 rows can't yield a 20d return → rs20 stays None."""
        qdf = make_quotes(pattern="uptrend", n=130)
        idx_df = self._make_index_df(n=5, return_pct=2.0)
        wash = {"washout_days": 0}
        result = compute_launch_setup(qdf, None, wash, default_cfg, index_df=idx_df)
        assert result["relative_strength_20d"] is None


class TestCurrentMoneyflow:
    """Round-2 P2-A — current_moneyflow_net_yi must reflect ``mf_df`` and steer
    capital_score; previously the field was hard-coded ``None`` and the score
    was stuck at a neutral 50 regardless of input."""

    def _mf_df(
        self, *, n: int, daily_wan: float, col: str = "net_mf_amount"
    ) -> pd.DataFrame:
        base_date = pd.Timestamp("2024-01-01")
        return pd.DataFrame({
            "ts_code": ["600000.SH"] * n,
            "trade_date": [
                (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d")
                for i in range(n)
            ],
            col: [daily_wan] * n,
        })

    def test_positive_inflow_sets_field_and_lifts_capital_score(
        self, default_cfg: ApwConfig
    ) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        # 1 亿/天 净流入 × 3 天 = 3 亿 → capital_score = 50 + 3 * 20 = 110 → clip 100
        mf = self._mf_df(n=130, daily_wan=10000.0)  # 10000 万 = 1 亿
        wash = {"washout_days": 0}

        baseline = compute_launch_setup(qdf, None, wash, default_cfg)
        with_mf = compute_launch_setup(qdf, None, wash, default_cfg, mf_df=mf)

        assert with_mf["current_moneyflow_net_yi"] == 3.0
        assert baseline["current_moneyflow_net_yi"] is None
        # capital_score lift propagates into launch_setup_score (weight 0.20):
        # ΔScore ≥ 0.20 * (100 - 50) = 10.
        assert with_mf["launch_setup_score"] > baseline["launch_setup_score"] + 5.0

    def test_negative_outflow_drops_capital_score(
        self, default_cfg: ApwConfig
    ) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        # -1 亿/天 × 3 天 = -3 亿 → capital_score = 50 + (-3) * 20 = -10 → clip 0
        mf = self._mf_df(n=130, daily_wan=-10000.0)
        wash = {"washout_days": 0}

        baseline = compute_launch_setup(qdf, None, wash, default_cfg)
        with_mf = compute_launch_setup(qdf, None, wash, default_cfg, mf_df=mf)

        assert with_mf["current_moneyflow_net_yi"] == -3.0
        # Capital score drops by ~50 → ΔScore ≤ -0.20 * 50 = -10
        assert with_mf["launch_setup_score"] < baseline["launch_setup_score"] - 5.0

    def test_alt_net_amount_column_supported(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        mf = self._mf_df(n=130, daily_wan=10000.0, col="net_amount")
        wash = {"washout_days": 0}
        result = compute_launch_setup(qdf, None, wash, default_cfg, mf_df=mf)
        assert result["current_moneyflow_net_yi"] == 3.0

    def test_missing_mf_keeps_field_none_and_neutral_score(
        self, default_cfg: ApwConfig
    ) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        wash = {"washout_days": 0}

        none_result = compute_launch_setup(qdf, None, wash, default_cfg, mf_df=None)
        empty_result = compute_launch_setup(
            qdf, None, wash, default_cfg, mf_df=pd.DataFrame()
        )
        # Column name we don't recognise — treated as missing.
        bad_col = pd.DataFrame({
            "trade_date": ["20240101"], "ts_code": ["600000.SH"], "other": [1.0],
        })
        bad_result = compute_launch_setup(qdf, None, wash, default_cfg, mf_df=bad_col)

        for r in (none_result, empty_result, bad_result):
            assert r["current_moneyflow_net_yi"] is None

        # All three reduce to the same neutral score.
        assert (
            none_result["launch_setup_score"]
            == empty_result["launch_setup_score"]
            == bad_result["launch_setup_score"]
        )

    def test_respects_launch_moneyflow_days_window(
        self, default_cfg: ApwConfig
    ) -> None:
        """Only the last ``cfg.launch_moneyflow_days`` rows count."""
        qdf = make_quotes(pattern="flat", n=130)
        rows = []
        base_date = pd.Timestamp("2024-01-01")
        # 10 historic days of +5 亿/天 + 3 final days of +1 亿/天
        for i in range(10):
            rows.append({
                "ts_code": "600000.SH",
                "trade_date": (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d"),
                "net_mf_amount": 50000.0,  # 5 亿
            })
        for i in range(10, 13):
            rows.append({
                "ts_code": "600000.SH",
                "trade_date": (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d"),
                "net_mf_amount": 10000.0,  # 1 亿
            })
        mf = pd.DataFrame(rows)
        wash = {"washout_days": 0}
        result = compute_launch_setup(qdf, None, wash, default_cfg, mf_df=mf)
        # tail(3) → 3 × 1 亿 = 3 亿
        assert result["current_moneyflow_net_yi"] == 3.0


class TestBreakWashoutHigh:
    """P1-2 regression — washout_high must exclude the current trade day."""

    def _build_breakout_frame(self) -> pd.DataFrame:
        """probe @ idx=110 (high 11.0) → 18 washout days with platform high 10.50
        → last day closes 10.80 above the platform but below the probe high."""
        qdf = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        # Force probe day high so probe is detected with a clean upper bound.
        qdf.at[110, "high"] = 11.0
        qdf.at[110, "close"] = 10.50
        qdf.at[110, "low"] = 9.80
        # 18 washout days — sideways platform with max high == 10.50
        for i in range(111, 129):
            qdf.at[i, "close"] = 10.30
            qdf.at[i, "high"] = 10.50
            qdf.at[i, "low"] = 10.10
            qdf.at[i, "vol"] = qdf.at[110, "vol"] * 0.4
        # Current day — close 10.80 (> platform high 10.50, < probe high 11.0).
        qdf.at[129, "close"] = 10.80
        qdf.at[129, "high"] = 10.95
        qdf.at[129, "low"] = 10.60
        qdf.at[129, "vol"] = qdf.at[110, "vol"] * 0.9
        return qdf

    def test_true_when_close_above_prior_platform(self, default_cfg: ApwConfig) -> None:
        qdf = self._build_breakout_frame()
        probe = detect_probe_day(qdf, default_cfg)
        assert probe is not None, "fixture must trigger probe detection"
        wash = compute_washout(qdf, pd.DataFrame(), probe, default_cfg)
        result = compute_launch_setup(qdf, probe, wash, default_cfg)
        # Before fix: result["break_washout_high"] is False because the slice
        # included the current day (high 10.95 > close 10.80). After fix the
        # slice is [probe+1, last) so max==10.50 and 10.80 > 10.50 → True.
        assert result["break_washout_high"] is True

    def test_false_when_close_below_prior_platform(self, default_cfg: ApwConfig) -> None:
        qdf = self._build_breakout_frame()
        # Push the last close back under the 10.50 platform top.
        qdf.at[129, "close"] = 10.40
        qdf.at[129, "high"] = 10.48
        probe = detect_probe_day(qdf, default_cfg)
        assert probe is not None
        wash = compute_washout(qdf, pd.DataFrame(), probe, default_cfg)
        result = compute_launch_setup(qdf, probe, wash, default_cfg)
        assert result["break_washout_high"] is False
