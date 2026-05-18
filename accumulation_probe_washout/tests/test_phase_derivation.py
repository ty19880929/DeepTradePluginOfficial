"""derive_phase state machine — T2.8."""

from __future__ import annotations

import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import derive_phase
from accumulation_probe_washout.schemas import APWPhase


@pytest.fixture
def cfg() -> ApwConfig:
    return ApwConfig()


def _mk(acc: float = 0.0, probe: float | None = None, wash: float = 0.0, launch: float = 0.0,
        *, low_broken: bool = False, washout_days: int = 0,
        above_ma5: bool = False, above_ma10: bool = False,
        vol_ratio: float = 0.0):
    accumulation = {"accumulation_score": acc}
    probe_d = None if probe is None else {"probe_quality_score": probe}
    washout = {
        "washout_score": wash,
        "post_probe_low_broken": low_broken,
        "washout_days": washout_days,
    }
    launch_d = {
        "launch_setup_score": launch,
        "above_ma5": above_ma5,
        "above_ma10": above_ma10,
        "current_volume_ratio_5d": vol_ratio,
    }
    return accumulation, probe_d, washout, launch_d


class TestPhaseDerivation:
    def test_no_setup_when_accumulation_low(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(acc=30)
        assert derive_phase(a, p, w, l, cfg) == APWPhase.NO_SETUP

    def test_accumulating_when_only_accumulation(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(acc=70)
        assert derive_phase(a, p, w, l, cfg) == APWPhase.ACCUMULATING

    def test_accumulating_when_probe_too_weak(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(acc=70, probe=40)
        assert derive_phase(a, p, w, l, cfg) == APWPhase.ACCUMULATING

    def test_probe_seen_when_washout_fails(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(acc=70, probe=80, wash=30, washout_days=10)
        assert derive_phase(a, p, w, l, cfg) == APWPhase.PROBE_SEEN

    def test_probe_seen_when_low_broken(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(acc=70, probe=80, wash=80, washout_days=10, low_broken=True)
        assert derive_phase(a, p, w, l, cfg) == APWPhase.PROBE_SEEN

    def test_washing_after_probe_when_launch_not_ready(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(acc=70, probe=80, wash=80, washout_days=10, launch=30)
        assert derive_phase(a, p, w, l, cfg) == APWPhase.WASHING_AFTER_PROBE

    def test_launch_ready_when_all_pass(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(
            acc=70, probe=80, wash=80, washout_days=10,
            launch=70, above_ma5=True, above_ma10=True, vol_ratio=1.5,
        )
        assert derive_phase(a, p, w, l, cfg) == APWPhase.LAUNCH_READY

    def test_launch_ready_requires_volume_pickup(self, cfg: ApwConfig) -> None:
        a, p, w, l = _mk(
            acc=70, probe=80, wash=80, washout_days=10,
            launch=70, above_ma5=True, above_ma10=True, vol_ratio=0.8,
        )
        # vol_ratio < 1.2 → not launch_ready
        assert derive_phase(a, p, w, l, cfg) == APWPhase.WASHING_AFTER_PROBE
