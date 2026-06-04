"""VwapEngine 数学正确性（设计 §2.1/§2.2，P1 验收项）。纯 stdlib。"""

from __future__ import annotations

import math

import pytest

from vwap_reversion.engine.vwap import VwapEngine
from vwap_reversion.schemas import IntervalBar


def bar(ts: int, dv: float, da: float, last: float, cv: float, ca: float) -> IntervalBar:
    return IntervalBar(
        code="159518.SZ", trade_date="20260603", ts=ts,
        interval_vol=dv, interval_amount=da, last=last, cum_vol=cv, cum_amount=ca,
    )


# 三段成交：1000股@1.00、3000股@1.10、1000股@0.90
# V=5000, A=1000+3300+900=5200 → VWAP=1.04
# E[p²] = (1000·1 + 3000·1.21 + 1000·0.81)/5000 = (1000+3630+810)/5000 = 1.088
# Var = 1.088 − 1.04² = 0.0064 → σ = 0.08
BARS = [
    bar(1, 1000, 1000.0, 1.00, 1000, 1000.0),
    bar(2, 3000, 3300.0, 1.10, 4000, 4300.0),
    bar(3, 1000, 900.0, 0.90, 5000, 5200.0),
]


def test_vwap_sigma_hand_computed() -> None:
    eng = VwapEngine(band_k=2.0)
    for b in BARS:
        m = eng.push(b)
    assert eng.vwap == pytest.approx(1.04)
    assert eng.sigma == pytest.approx(0.08)
    assert m.band_upper == pytest.approx(1.04 + 2 * 0.08)
    assert m.band_lower == pytest.approx(1.04 - 2 * 0.08)
    # last=0.90 → z = (0.90 − 1.04)/0.08 = −1.75
    assert m.z == pytest.approx(-1.75)


def test_first_bar_sigma_zero_z_none() -> None:
    eng = VwapEngine(band_k=2.0)
    m = eng.push(BARS[0])  # 单一价位：Var = 1·1²−1² = 0
    assert eng.vwap == pytest.approx(1.0)
    assert eng.sigma == pytest.approx(0.0)
    assert m.z is None
    assert m.band_upper == pytest.approx(m.band_lower) == pytest.approx(1.0)


def test_rebuild_equals_incremental() -> None:
    inc = VwapEngine(band_k=2.0)
    for b in BARS:
        inc.push(b)
    re = VwapEngine(band_k=2.0)
    assert re.rebuild(BARS) == 3
    assert re.vwap == pytest.approx(inc.vwap)
    assert re.sigma == pytest.approx(inc.sigma)
    assert re.cum_vol == inc.cum_vol


def test_vwap_exactness_from_cumulative() -> None:
    # VWAP 永远等于 cum_amount/cum_vol —— 与增量路径无关（漏采样不影响）。
    eng = VwapEngine(band_k=2.0)
    eng.push(bar(1, 1000, 1000.0, 1.00, 1000, 1000.0))
    # 模拟漏采样：直接跳到更大的累计（区间跨缺口）
    eng.push(bar(9, 4000, 4200.0, 0.90, 5000, 5200.0))
    assert eng.vwap == pytest.approx(5200.0 / 5000.0)


def test_guards() -> None:
    eng = VwapEngine(band_k=2.0)
    with pytest.raises(ValueError, match="interval_vol"):
        eng.push(bar(1, 0.0, 0.0, 1.0, 1000, 1000.0))
    eng.push(BARS[0])
    with pytest.raises(ValueError, match="回退"):
        eng.push(bar(2, 100, 100.0, 1.0, 500, 500.0))  # 累计量回退
    with pytest.raises(ValueError, match="尚无成交量"):
        _ = VwapEngine(band_k=2.0).vwap
    with pytest.raises(ValueError, match="band_k"):
        VwapEngine(band_k=0)


def test_sigma_never_nan_on_fp_noise() -> None:
    # 浮点噪声可能让 Var 微负 —— σ 必须钳为 0 而不是 NaN。
    eng = VwapEngine(band_k=2.0)
    eng.push(bar(1, 3, 0.3, 0.1, 3, 0.3))
    eng.push(bar(2, 7, 0.7, 0.1, 10, 1.0))
    assert eng.sigma >= 0.0
    assert not math.isnan(eng.sigma)
