"""PR-1.1 — T+1 标签构造单元测试。

覆盖：
* 阈值边界（9.69 / 9.70 / 9.71）
* 缺失字段 → None
* pre_close 非正 → None（防御性）
* batch label_dataframe 输出 nullable Int64
* compute_max_upside_pct 辅助
"""

from __future__ import annotations

import pandas as pd
import pytest

from limit_up_board.lgb.labels import (
    DEFAULT_LABEL_THRESHOLD_PCT,
    compute_label_for_t1,
    compute_max_upside_pct,
    label_dataframe,
)


def _row(pre_close: float | None, high: float | None) -> dict:
    return {"pre_close": pre_close, "high": high}


class TestComputeLabel:
    def test_below_threshold_zero(self) -> None:
        # (10.969 - 10) / 10 * 100 = 9.69 < 9.7
        assert compute_label_for_t1(_row(10.0, 10.969)) == 0

    def test_at_threshold_one(self) -> None:
        # (10.97 - 10) / 10 * 100 = 9.7 → label 1
        assert compute_label_for_t1(_row(10.0, 10.97)) == 1

    def test_above_threshold_one(self) -> None:
        # (10.971 - 10) / 10 * 100 = 9.71 → label 1
        assert compute_label_for_t1(_row(10.0, 10.971)) == 1

    def test_far_above_threshold(self) -> None:
        # 30% 高开 → label 1
        assert compute_label_for_t1(_row(10.0, 13.0)) == 1

    def test_negative_open(self) -> None:
        # high 小于 pre_close（不可能但要 robust） → label 0
        assert compute_label_for_t1(_row(10.0, 9.5)) == 0

    def test_missing_pre_close(self) -> None:
        assert compute_label_for_t1(_row(None, 12.0)) is None

    def test_missing_high(self) -> None:
        assert compute_label_for_t1(_row(10.0, None)) is None

    def test_none_row(self) -> None:
        assert compute_label_for_t1(None) is None

    def test_zero_pre_close(self) -> None:
        # 防御性：pre_close = 0 → None（除零保护）
        assert compute_label_for_t1(_row(0.0, 10.0)) is None

    def test_nan_pre_close(self) -> None:
        assert compute_label_for_t1(_row(float("nan"), 10.0)) is None

    def test_custom_threshold(self) -> None:
        # 阈值改成 5%，原本 label=0 的 7% 涨幅 → label=1
        assert compute_label_for_t1(_row(10.0, 10.7), threshold_pct=5.0) == 1
        assert compute_label_for_t1(_row(10.0, 10.49), threshold_pct=5.0) == 0

    def test_default_threshold_constant_consistent(self) -> None:
        # 防止 design 阈值漂移
        assert DEFAULT_LABEL_THRESHOLD_PCT == pytest.approx(9.7)


class TestComputeMaxUpside:
    def test_happy_path(self) -> None:
        assert compute_max_upside_pct(_row(10.0, 12.0)) == pytest.approx(20.0)

    def test_missing_returns_none(self) -> None:
        assert compute_max_upside_pct(_row(None, 12.0)) is None
        assert compute_max_upside_pct(_row(10.0, None)) is None
        assert compute_max_upside_pct(None) is None

    def test_zero_pre_close(self) -> None:
        assert compute_max_upside_pct(_row(0.0, 5.0)) is None


class TestLabelDataframeBatch:
    def test_basic_mix(self) -> None:
        samples = pd.DataFrame(
            [
                {"ts_code": "A.SZ", "trade_date": "20260529", "next_trade_date": "20260530"},
                {"ts_code": "B.SZ", "trade_date": "20260529", "next_trade_date": "20260530"},
                {"ts_code": "C.SZ", "trade_date": "20260529", "next_trade_date": "20260530"},
            ]
        )
        lookup = {
            ("A.SZ", "20260530"): _row(10.0, 11.5),   # +15% → 1
            ("B.SZ", "20260530"): _row(10.0, 10.5),   # +5% → 0
            # C 没有 T+1 数据
        }
        out = label_dataframe(samples, lookup)
        assert out.iloc[0] == 1
        assert out.iloc[1] == 0
        assert pd.isna(out.iloc[2])
        # nullable Int64
        assert str(out.dtype) == "Int64"
        assert out.name == "label"

    def test_index_alignment(self) -> None:
        samples = pd.DataFrame(
            {"ts_code": ["X.SZ"], "next_trade_date": ["20260530"]},
            index=[42],
        )
        lookup = {("X.SZ", "20260530"): _row(10.0, 12.0)}
        out = label_dataframe(samples, lookup)
        assert list(out.index) == [42]

    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame([{"ts_code": "A.SZ"}])  # 缺 next_trade_date
        with pytest.raises(ValueError, match="next_trade_date"):
            label_dataframe(bad, {})

    def test_threshold_override(self) -> None:
        samples = pd.DataFrame(
            [{"ts_code": "A.SZ", "next_trade_date": "20260530"}]
        )
        lookup = {("A.SZ", "20260530"): _row(10.0, 10.6)}  # +6%
        out_default = label_dataframe(samples, lookup)
        out_strict = label_dataframe(samples, lookup, threshold_pct=5.0)
        assert out_default.iloc[0] == 0
        assert out_strict.iloc[0] == 1
