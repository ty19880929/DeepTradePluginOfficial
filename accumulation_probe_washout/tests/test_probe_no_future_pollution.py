"""Round-2 P2-B regression: probe_volume_rank_pct_60d must not be influenced
by sessions after the candidate probe day.

Before the fix, ``detect_probe_day`` computed the rank percentile against
``df.tail(base_lookback)`` — a window anchored on the *last* row of the frame.
When the probe day sat in the middle of the lookback, post-probe trading
volume bled into the comparison set, dragging the historic probe day's rank
down and potentially flipping a real probe to ``None``.

These tests build two frames that differ only in volumes AFTER the probe day
and assert detection is identical.
"""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import detect_probe_day
from tests.conftest import make_quotes


class TestProbeRankIgnoresFuture:
    @staticmethod
    def _dampen_amplitude(df: pd.DataFrame, indices: range) -> None:
        """Squash amplitude so a high-volume day cannot itself qualify as a
        probe (amplitude_pct < probe_amplitude_pct_min)."""
        for i in indices:
            close = float(df.at[i, "close"])
            df.at[i, "open"] = round(close, 3)
            df.at[i, "high"] = round(close * 1.001, 3)
            df.at[i, "low"] = round(close * 0.999, 3)
            df.at[i, "turnover_rate"] = 0.5  # below 2.0 threshold

    def test_post_probe_volume_does_not_demote_probe(
        self, default_cfg: ApwConfig
    ) -> None:
        base = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        baseline = detect_probe_day(base, default_cfg)
        assert baseline is not None
        assert baseline["probe_days_ago"] == 19

        # Polluted frame: post-probe rows carry 5× the probe-day volume but
        # have squashed amplitude + low turnover so they can never qualify as
        # probes themselves. They exist solely to skew a tail()-based rank.
        polluted = base.copy()
        probe_vol = float(polluted.at[110, "vol"])
        for i in range(111, 130):
            polluted.at[i, "vol"] = probe_vol * 5.0
        self._dampen_amplitude(polluted, range(111, 130))

        result = detect_probe_day(polluted, default_cfg)
        assert result is not None, (
            "post-probe volumes must not influence the probe-day rank window"
        )
        assert result["probe_days_ago"] == 19, (
            "must still return the index-110 probe, not a post-probe spike"
        )
        # Probe-day vol is the max within [pos-base+1, pos] → top of the rank.
        assert (
            result["probe_volume_rank_pct_60d"]
            >= default_cfg.probe_volume_rank_pct_60d_min
        )

    def test_rank_uses_window_ending_at_probe_day(
        self, default_cfg: ApwConfig
    ) -> None:
        df = make_quotes(pattern="flat", n=130, probe_index=110, probe_multiplier=5.0)
        # One single massive post-probe spike. Under the old tail() window this
        # one day would crush the probe's rank below threshold and the probe
        # would be rejected; under the fixed slice it's outside the comparison
        # set.
        df.at[111, "vol"] = df.at[110, "vol"] * 50.0
        self._dampen_amplitude(df, range(111, 112))

        result = detect_probe_day(df, default_cfg)
        assert result is not None
        assert result["probe_days_ago"] == 19
        # Probe-day vol is the max in [pos-base+1, pos]; the only day < it is
        # the probe itself, so rank ≈ (N-1)/N × 100. With base_lookback=120
        # this is ~99.1 — well above the 90% threshold and unaffected by the
        # post-probe spike.
        assert result["probe_volume_rank_pct_60d"] >= 99.0
