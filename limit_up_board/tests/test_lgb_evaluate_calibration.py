"""P2-2 Part 2 (v0.13.1)：``evaluate --show-calibration`` 输出 Brier + reliability。

直接构造 :class:`EvaluateResult` 测渲染路径 + 单测内部 ``_safe_brier`` 走通；
真实 ``evaluate_model`` 端到端（要拉 tushare）放在 slow 集合，CI 跳过。
"""

from __future__ import annotations

import numpy as np

from limit_up_board.lgb.calibration import brier_score, reliability_table
from limit_up_board.lgb.evaluate import EvaluateResult, format_evaluate_table


def test_format_evaluate_table_renders_calibration_section() -> None:
    res = EvaluateResult(
        model_id="m-1",
        window_start="20260501",
        window_end="20260520",
        label_threshold_pct=9.7,
        n_samples=500,
        n_labeled=480,
        n_positive=120,
        n_trade_dates=12,
        auc=0.71,
        logloss=0.58,
        topk=[],
        calibration_method="isotonic",
        brier=0.1234,
        reliability=[
            {"bin_lo": 0.0, "bin_hi": 0.1, "n": 60, "mean_pred": 0.05, "observed": 0.08, "gap": 0.03},
            {"bin_lo": 0.1, "bin_hi": 0.2, "n": 55, "mean_pred": 0.15, "observed": 0.18, "gap": 0.03},
        ],
    )
    text = format_evaluate_table(res)
    assert "Calibration · method=isotonic" in text
    assert "Brier=0.1234" in text
    assert "bins=2" in text
    # bins 表头 + 数据行
    assert "mean_pred" in text and "observed" in text and "gap" in text


def test_format_evaluate_table_omits_calibration_when_brier_none() -> None:
    res = EvaluateResult(
        model_id="m-1",
        window_start="20260501",
        window_end="20260520",
        label_threshold_pct=9.7,
        n_samples=100, n_labeled=80, n_positive=20, n_trade_dates=5,
        auc=0.7, logloss=0.6, topk=[],
    )
    text = format_evaluate_table(res)
    assert "Calibration" not in text


def test_brier_and_reliability_consistent() -> None:
    """Brier 与 reliability_table 的 mean_pred / observed 由同一份 y_true/y_pred 计算。"""
    rng = np.random.default_rng(123)
    n = 1000
    y = rng.binomial(1, 0.4, size=n)
    p = np.clip(rng.normal(0.4, 0.15, size=n), 0.0, 1.0)
    b = brier_score(y, p)
    table = reliability_table(y, p, n_bins=10)
    assert b is not None
    total = sum(int(row["n"]) for row in table)
    assert total == n
    # Brier 的 lower-bound 由各 bin 的 var(y|bin) 加权决定；这里只断言数值在 (0, 1)
    assert 0.0 < b < 1.0
