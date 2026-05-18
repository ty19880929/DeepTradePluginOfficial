"""compute_accumulation — T2.4."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import compute_accumulation
from tests.conftest import make_quotes


class TestAccumulationScore:
    def test_low_flat_with_positive_moneyflow_scores_high(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        # Synthesise moneyflow with steady positive net inflow.
        mf = pd.DataFrame(
            [
                {"trade_date": d, "ts_code": "600000.SH", "net_mf_amount": 5000.0}
                for d in qdf["trade_date"].tail(30)
            ]
        )
        result = compute_accumulation(qdf, mf, default_cfg)
        assert result["accumulation_score"] >= 50.0
        assert result["accumulation_net_mf_yi"] is not None
        assert result["accumulation_net_mf_yi"] > 0
        assert "missing_data" in result

    def test_uptrend_already_run_scores_low(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="uptrend", n=130)
        result = compute_accumulation(qdf, pd.DataFrame(), default_cfg)
        # Steeply uptrending = already broken out, low position score should drop
        assert result["accumulation_score"] < 50.0

    def test_missing_moneyflow_does_not_crash(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        result = compute_accumulation(qdf, None, default_cfg)
        assert result["accumulation_net_mf_yi"] is None
        assert "moneyflow" in result["missing_data"]
        # Still produces a score (just downgraded)
        assert 0 <= result["accumulation_score"] <= 100

    def test_empty_window_returns_zero_score(self, default_cfg: ApwConfig) -> None:
        result = compute_accumulation(pd.DataFrame(), pd.DataFrame(), default_cfg)
        assert result["accumulation_score"] == 0.0
        assert result["accumulation_days"] == 0
