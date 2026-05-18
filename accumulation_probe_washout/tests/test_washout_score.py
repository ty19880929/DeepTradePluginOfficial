"""compute_washout — T2.6."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import compute_washout, detect_probe_day
from tests.conftest import make_quotes


class TestWashoutScore:
    def test_healthy_washout_scores_high(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        # Sane post-probe — shallow pullback that does NOT break the probe low.
        # Probe_low ≈ prev_close * 0.99; we hold post-probe lows comfortably above that.
        probe_close = qdf.at[110, "close"]
        for i in range(111, 130):
            qdf.at[i, "close"] = probe_close * 0.985
            qdf.at[i, "high"] = qdf.at[i, "close"] * 1.005
            qdf.at[i, "low"] = qdf.at[i, "close"] * 0.995
            qdf.at[i, "vol"] = qdf.at[110, "vol"] * 0.4
        probe = detect_probe_day(qdf, default_cfg)
        assert probe is not None
        result = compute_washout(qdf, pd.DataFrame(), probe, default_cfg)
        assert not result["post_probe_low_broken"]
        assert result["washout_days"] == 19  # 110 → 129 exclusive
        assert result["washout_score"] >= 50.0

    def test_low_broken_drops_score(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        # Crash post-probe — break below the probe's low
        probe_low = qdf.at[110, "low"]
        for i in range(111, 130):
            qdf.at[i, "close"] = probe_low * 0.85
            qdf.at[i, "high"] = qdf.at[i, "close"] * 1.01
            qdf.at[i, "low"] = qdf.at[i, "close"] * 0.95
            qdf.at[i, "vol"] = qdf.at[110, "vol"] * 0.8
        probe = detect_probe_day(qdf, default_cfg)
        assert probe is not None
        result = compute_washout(qdf, pd.DataFrame(), probe, default_cfg)
        assert result["post_probe_low_broken"] is True
        # Should be significantly dragged down by the broken support
        assert result["washout_score"] < 60.0

    def test_no_probe_returns_zero(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        result = compute_washout(qdf, pd.DataFrame(), None, default_cfg)
        assert result["washout_score"] == 0.0
        assert result["washout_days"] == 0
