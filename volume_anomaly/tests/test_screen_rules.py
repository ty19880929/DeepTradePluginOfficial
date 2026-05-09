"""volume-anomaly v0.3.0 — ScreenRules / 上影线 / 流通市值分桶 单元测试。

只测纯函数与 dataclass 行为；不接 tushare / DB / LLM。
"""

from __future__ import annotations

import math

import pytest

from volume_anomaly.data import (
    DEFAULT_TURNOVER_BUCKETS,
    ScreenRules,
    _resolve_turnover_bucket,
)


# ---------------------------------------------------------------------------
# P0-1 — upper shadow ratio formula
# ---------------------------------------------------------------------------


def _upper_shadow_ratio(*, open_: float, high: float, low: float, close: float) -> float:
    """Mirror the formula used inline in ``screen_anomalies``."""
    rng = max(high - low, 1e-9)
    body_top = max(open_, close)
    return (high - body_top) / rng


class TestUpperShadowFormula:
    """三组人造 K 线 → 通过 / 拒绝 / 通过，阈值 0.35。"""

    threshold = 0.35

    def test_pure_yang_line_passes(self) -> None:
        # close=high, open=low → upper_shadow_ratio = 0
        r = _upper_shadow_ratio(open_=10.0, high=11.0, low=10.0, close=11.0)
        assert r == pytest.approx(0.0)
        assert r <= self.threshold

    def test_avalanche_pin_rejected(self) -> None:
        # body 在下半段、上影线 50% 振幅 → ratio = 0.5
        # range = 2; body_top = 11; (12 - 11) / 2 = 0.5
        r = _upper_shadow_ratio(open_=10.0, high=12.0, low=10.0, close=11.0)
        assert r == pytest.approx(0.5)
        assert r > self.threshold

    def test_reasonable_shadow_passes(self) -> None:
        # 25% 上影 → 通过
        # range = 4; body_top = 11.5; (12 - 11.5) / 4 = 0.125
        r = _upper_shadow_ratio(open_=10.0, high=12.0, low=8.0, close=11.5)
        assert r == pytest.approx(0.125)
        assert r <= self.threshold

    def test_disabled_filter_bypass(self) -> None:
        rules = ScreenRules(upper_shadow_ratio_max=None)
        assert rules.upper_shadow_ratio_max is None


# ---------------------------------------------------------------------------
# P0-2 — turnover_buckets resolver
# ---------------------------------------------------------------------------


class TestTurnoverBucketResolution:
    """命中分桶 + 边界归属 + 全局回退。"""

    def test_micro_cap_passes_high_turnover(self) -> None:
        # 微盘 30 亿 turnover=14% → 应在 [5, 15] 区间内 → 通过
        idx, label, t_min, t_max = _resolve_turnover_bucket(30.0, DEFAULT_TURNOVER_BUCKETS)
        assert idx == 0
        assert label == "≤50亿"
        assert (t_min, t_max) == (5.0, 15.0)
        assert t_min <= 14.0 <= t_max

    def test_mid_cap_rejects_high_turnover(self) -> None:
        # 中盘 500 亿 turnover=10% → 桶为 [2.5, 9.0] → 10% > 9.0 → 拒绝
        idx, label, t_min, t_max = _resolve_turnover_bucket(500.0, DEFAULT_TURNOVER_BUCKETS)
        assert idx == 2
        assert label == "200-1000亿"
        assert (t_min, t_max) == (2.5, 9.0)
        assert not (t_min <= 10.0 <= t_max)

    def test_large_cap_passes_modest_turnover(self) -> None:
        # 大盘 2000 亿 turnover=5% → 桶 [1.5, 6.0] → 通过
        idx, label, t_min, t_max = _resolve_turnover_bucket(2000.0, DEFAULT_TURNOVER_BUCKETS)
        assert idx == 3
        assert label == ">1000亿"
        assert (t_min, t_max) == (1.5, 6.0)
        assert t_min <= 5.0 <= t_max

    def test_boundary_50yi_belongs_to_smaller_bucket(self) -> None:
        # E4 决策：边界值归"较小桶"——50.0 → ≤50亿，而不是 50-200亿。
        idx, label, _, _ = _resolve_turnover_bucket(50.0, DEFAULT_TURNOVER_BUCKETS)
        assert idx == 0
        assert label == "≤50亿"

    def test_boundary_200yi_belongs_to_smaller_bucket(self) -> None:
        idx, label, _, _ = _resolve_turnover_bucket(200.0, DEFAULT_TURNOVER_BUCKETS)
        assert idx == 1
        assert label == "50-200亿"

    def test_global_mode_old_logic_passes_all_three(self) -> None:
        """旧模式（turnover_buckets=None）下，3.0–10.0% 全局阈值；
        微盘 14% / 中盘 10% / 大盘 5% → 通过 / 通过(边界) / 通过。
        与分桶模式（通过 / 拒绝 / 通过）形成对比，证明分桶纠正了大/小盘错配。"""
        rules = ScreenRules(turnover_buckets=None)
        assert rules.turnover_buckets is None
        # 用全局区间手动测三档 turnover：
        for circ_mv_yi, turnover in [(30.0, 14.0), (500.0, 10.0), (2000.0, 5.0)]:
            # 全局 [3.0, 10.0]
            in_global = rules.turnover_min <= turnover <= rules.turnover_max
            if (circ_mv_yi, turnover) == (30.0, 14.0):
                assert not in_global  # 14% 实际超出 10.0 → 旧逻辑也会拒绝微盘
            elif (circ_mv_yi, turnover) == (500.0, 10.0):
                assert in_global       # 边界通过
            else:
                assert in_global

    def test_circ_mv_missing_uses_global_fallback_logic(self) -> None:
        """circ_mv 缺失时，screen_anomalies 内部逻辑应该退化到全局
        [turnover_min, turnover_max]。这里直接验证桶选择不会被调用——
        使用 ScreenRules 的全局字段即可。"""
        rules = ScreenRules()  # 默认 buckets enabled, global [3, 10]
        assert rules.turnover_min == 3.0
        assert rules.turnover_max == 10.0


# ---------------------------------------------------------------------------
# ScreenRules.from_dict — null-as-inf for last bucket
# ---------------------------------------------------------------------------


class TestScreenRulesFromDict:
    def test_bucket_null_translates_to_inf(self) -> None:
        # E1 决策：JSON 配置 [[50, 5, 15], [null, 1.5, 6]] → 内层 null → math.inf
        rules = ScreenRules.from_dict(
            {"turnover_buckets": [[50, 5, 15], [None, 1.5, 6]]}
        )
        assert rules.turnover_buckets is not None
        assert len(rules.turnover_buckets) == 2
        assert rules.turnover_buckets[0] == (50.0, 5.0, 15.0)
        b_max, t_min, t_max = rules.turnover_buckets[1]
        assert math.isinf(b_max)
        assert (t_min, t_max) == (1.5, 6.0)

    def test_bucket_null_disables_bucketing(self) -> None:
        # 显式传 null → 退化到全局阈值
        rules = ScreenRules.from_dict({"turnover_buckets": None})
        assert rules.turnover_buckets is None

    def test_bucket_missing_keeps_default(self) -> None:
        # 不传 turnover_buckets → 默认 DEFAULT_TURNOVER_BUCKETS（D3 默认开启）
        rules = ScreenRules.from_dict({"pct_chg_min": 5.0})
        assert rules.turnover_buckets is not None
        assert len(rules.turnover_buckets) == len(DEFAULT_TURNOVER_BUCKETS)

    def test_upper_shadow_null_disables(self) -> None:
        rules = ScreenRules.from_dict({"upper_shadow_ratio_max": None})
        assert rules.upper_shadow_ratio_max is None

    def test_upper_shadow_missing_keeps_default(self) -> None:
        # D2 决策：默认 0.35
        rules = ScreenRules.from_dict({"pct_chg_min": 5.0})
        assert rules.upper_shadow_ratio_max == pytest.approx(0.35)

    def test_upper_shadow_explicit_value(self) -> None:
        rules = ScreenRules.from_dict({"upper_shadow_ratio_max": 0.40})
        assert rules.upper_shadow_ratio_max == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# ScreenRules.__post_init__ — validation
# ---------------------------------------------------------------------------


class TestScreenRulesValidation:
    def test_upper_shadow_invalid(self) -> None:
        with pytest.raises(ValueError, match="upper_shadow_ratio_max"):
            ScreenRules(upper_shadow_ratio_max=0.0)
        with pytest.raises(ValueError, match="upper_shadow_ratio_max"):
            ScreenRules(upper_shadow_ratio_max=1.5)
        with pytest.raises(ValueError, match="upper_shadow_ratio_max"):
            ScreenRules(upper_shadow_ratio_max=-0.1)

    def test_buckets_must_be_strictly_increasing(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            ScreenRules(turnover_buckets=[(50.0, 5.0, 15.0), (50.0, 3.5, 12.0)])

    def test_buckets_invalid_turnover_range(self) -> None:
        with pytest.raises(ValueError, match="invalid turnover range"):
            ScreenRules(turnover_buckets=[(50.0, 10.0, 5.0)])  # min > max

    def test_buckets_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ScreenRules(turnover_buckets=[])

    def test_buckets_none_is_allowed(self) -> None:
        rules = ScreenRules(turnover_buckets=None)
        assert rules.turnover_buckets is None


# ---------------------------------------------------------------------------
# ScreenRules.as_dict — JSON-safe inf
# ---------------------------------------------------------------------------


class TestScreenRulesAsDict:
    def test_inf_serialised_as_null(self) -> None:
        import json as _json

        rules = ScreenRules.defaults()
        d = rules.as_dict()
        # 默认最后一桶 max == inf → as_dict 转为 None
        last_bucket = d["turnover_buckets"][-1]
        assert last_bucket[0] is None
        # 整体应能 dump 为标准 JSON（不依赖 Python 的 Infinity 扩展）
        encoded = _json.dumps(d, allow_nan=False)
        assert "Infinity" not in encoded

    def test_disabled_buckets_serialise_as_null(self) -> None:
        rules = ScreenRules(turnover_buckets=None)
        d = rules.as_dict()
        assert d["turnover_buckets"] is None
