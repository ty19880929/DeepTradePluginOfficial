"""detect_probe_day — T2.5, including D2 'most recent' semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import detect_probe_day
from tests.conftest import make_quotes


class TestDetectProbeDay:
    def test_no_probe_returns_none(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130)
        assert detect_probe_day(qdf, default_cfg) is None

    def test_single_probe_detected(self, default_cfg: ApwConfig) -> None:
        qdf = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        result = detect_probe_day(qdf, default_cfg)
        assert result is not None
        assert result["probe_volume_ratio_5d"] >= default_cfg.probe_volume_ratio_5d_min
        assert result["probe_quality_score"] > 0
        # Probe is at index 110, last day at 129 → days ago = 19
        assert result["probe_days_ago"] == 19

    def test_multiple_probes_takes_most_recent_D2(self, default_cfg: ApwConfig) -> None:
        """D2: two天量天 in the window — must return the LATER one."""
        qdf = make_quotes(pattern="flat", n=130, probe_index=100, probe_multiplier=5.0)
        # Insert a second probe at index 120
        base_vol = qdf.at[100, "vol"]
        qdf.at[120, "vol"] = base_vol  # roughly same big-volume day
        qdf.at[120, "turnover_rate"] = 9.0
        qdf.at[120, "high"] = qdf.at[120, "close"] * 1.08
        qdf.at[120, "low"] = qdf.at[119, "close"] * 0.99
        qdf.at[120, "open"] = qdf.at[119, "close"]

        result = detect_probe_day(qdf, default_cfg)
        assert result is not None
        # Must be the later one (index 120 → 9 days ago)
        assert result["probe_days_ago"] == 9

    def test_probe_below_threshold_rejected(self, default_cfg: ApwConfig) -> None:
        # 1.5x volume — well below 2.5x threshold
        qdf = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=1.5)
        assert detect_probe_day(qdf, default_cfg) is None
