"""PR-3.1 — :func:`lgb.evaluate.evaluate_model` 单元测试。

测试目标：
* Top-K 聚合逻辑：score 排序、当日 < K 时 picks 收缩、NaN score 跳过。
* AUC / logloss 计算（含单类别退化分支）。
* schema mismatch / 模型未加载 → 结果 schema_version_match=False。
* Baseline = 当日全 candidate 平均 hit rate / 平均 upside（与"随机 K 抽取"等价期望）。

evaluate_model 端到端需要 tushare/calendar，本测试直接调用纯函数
``_compute_topk_metrics``、``_safe_auc``、``_safe_logloss`` 与
``format_evaluate_table`` 完成验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from limit_up_board.lgb.evaluate import (
    DEFAULT_K_VALUES,
    EvaluateResult,
    TopKMetrics,
    _compute_topk_metrics,
    _safe_auc,
    _safe_logloss,
    format_evaluate_table,
)


# ---------------------------------------------------------------------------
# Top-K aggregation
# ---------------------------------------------------------------------------


def _build_eval_df() -> pd.DataFrame:
    """Synthetic eval_df mirroring the shape that ``evaluate_model`` produces.

    Three trade dates, each with 6 candidates; lgb_score correlates with label
    so top-5 should beat the baseline.
    """
    rows: list[dict] = []
    for d, td in enumerate(["20260520", "20260521", "20260522"]):
        for i in range(6):
            # Higher score = higher label probability; tweak by day to differentiate.
            score = 0.05 + 0.15 * i + d * 0.01
            label = 1 if score > 0.55 else 0
            # Upside loosely tracks the label so top-K mean upside > baseline mean.
            upside = 9.0 + i * 0.5 if label == 1 else 1.0 + i * 0.2
            rows.append(
                {
                    "ts_code": f"S{d}{i:02d}.SH",
                    "trade_date": td,
                    "next_trade_date": "20260601",
                    "pct_chg_t1": upside,
                    "lgb_score": score,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_topk_aggregates_per_day_and_compares_to_baseline() -> None:
    df = _build_eval_df()
    metrics = _compute_topk_metrics(df, k_values=(5, 10, 20))
    by_k = {m.k: m for m in metrics}

    m5 = by_k[5]
    assert m5.n_days_evaluated == 3
    # 5 picks/day × 3 days = 15
    assert m5.pick_count == 15
    # Top-5 should out-hit the baseline (6-candidate baseline includes 2 zeros).
    assert m5.hit_rate_pct is not None
    assert m5.baseline_hit_rate_pct is not None
    assert m5.hit_rate_pct > m5.baseline_hit_rate_pct
    assert m5.delta_hit_rate_pct == round(
        m5.hit_rate_pct - m5.baseline_hit_rate_pct, 2
    )

    # K=10 > n_day → picks compress to 6/day.
    m10 = by_k[10]
    assert m10.pick_count == 18
    # When K saturates everyone, top-K hit rate equals the baseline.
    assert m10.hit_rate_pct == m10.baseline_hit_rate_pct
    # K=20: also saturates everyone — same outcome.
    m20 = by_k[20]
    assert m20.pick_count == 18
    assert m20.hit_rate_pct == m20.baseline_hit_rate_pct


def test_topk_skips_days_with_only_nan_scores() -> None:
    df = pd.DataFrame(
        [
            {"trade_date": "20260520", "lgb_score": np.nan, "label": 1, "pct_chg_t1": 5.0},
            {"trade_date": "20260520", "lgb_score": np.nan, "label": 0, "pct_chg_t1": 1.0},
            {"trade_date": "20260521", "lgb_score": 0.8, "label": 1, "pct_chg_t1": 9.0},
            {"trade_date": "20260521", "lgb_score": 0.2, "label": 0, "pct_chg_t1": 1.0},
        ]
    )
    metrics = _compute_topk_metrics(df, k_values=(1,))
    m = metrics[0]
    # Only one day had any usable scores → n_days_evaluated == 1.
    assert m.n_days_evaluated == 1
    assert m.pick_count == 1
    assert m.hit_count == 1
    assert m.hit_rate_pct == 100.0


def test_topk_empty_eval_df_returns_zero_metrics() -> None:
    metrics = _compute_topk_metrics(
        pd.DataFrame(columns=["trade_date", "lgb_score", "label", "pct_chg_t1"]),
        k_values=(5, 10),
    )
    by_k = {m.k: m for m in metrics}
    assert by_k[5].n_days_evaluated == 0
    assert by_k[5].hit_rate_pct is None
    assert by_k[5].delta_hit_rate_pct is None


# ---------------------------------------------------------------------------
# AUC / logloss helpers
# ---------------------------------------------------------------------------


def test_safe_auc_returns_none_for_single_class() -> None:
    y = np.array([0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert _safe_auc(y, s) is None


def test_safe_auc_high_separability_gives_near_one() -> None:
    y = np.array([0] * 20 + [1] * 20)
    s = np.concatenate([np.full(20, 0.1), np.full(20, 0.9)])
    auc = _safe_auc(y, s)
    assert auc is not None and auc > 0.99


def test_safe_logloss_clips_extremes() -> None:
    y = np.array([0, 1, 0, 1])
    s = np.array([0.0, 1.0, 0.0, 1.0])  # raw values would explode without clipping
    ll = _safe_logloss(y, s)
    assert ll is not None and ll < 1.0


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------


def test_format_evaluate_table_smoke() -> None:
    result = EvaluateResult(
        model_id="20260530_1_demo",
        window_start="20260101",
        window_end="20260530",
        label_threshold_pct=9.7,
        n_samples=120,
        n_labeled=110,
        n_positive=33,
        n_trade_dates=20,
        auc=0.71,
        logloss=0.55,
        topk=[
            TopKMetrics(
                k=5, n_days_evaluated=20, hit_count=40, pick_count=100,
                hit_rate_pct=40.0, avg_upside_pct=3.4,
                baseline_hit_rate_pct=30.0, baseline_avg_upside_pct=2.0,
                delta_hit_rate_pct=10.0, delta_avg_upside_pct=1.4,
            )
        ],
    )
    text = format_evaluate_table(result)
    assert "AUC = 0.7100" in text
    assert "Top-K vs baseline" in text
    # Header / first data row mention K and the deltas
    assert " 5 " in text
    assert "+10.0" in text  # delta hit rate column


def test_format_evaluate_table_shows_schema_mismatch() -> None:
    result = EvaluateResult(
        model_id="ghost",
        window_start="20260101",
        window_end="20260530",
        label_threshold_pct=9.7,
        n_samples=0,
        n_labeled=0,
        n_positive=0,
        n_trade_dates=0,
        auc=None,
        logloss=None,
        topk=[],
        schema_version_match=False,
        schema_mismatch_detail="schema_mismatch: model_features(3) != FEATURE_NAMES(50)",
    )
    text = format_evaluate_table(result)
    assert "scorer not loaded" in text
    assert "schema_mismatch" in text


def test_default_k_values_constant() -> None:
    assert DEFAULT_K_VALUES == (5, 10, 20)
