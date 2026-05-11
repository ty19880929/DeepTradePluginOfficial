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


# ---------------------------------------------------------------------------
# PR-3.3 — Feature drift detection
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_zero() -> None:
    from limit_up_board.lgb.evaluate import _psi

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    psi = _psi(x, x.copy(), n_bins=10)
    assert psi is not None
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_distribution_high() -> None:
    from limit_up_board.lgb.evaluate import _psi

    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 1000)
    shifted = rng.normal(2.0, 1, 1000)  # mean shift by 2 std
    psi = _psi(base, shifted, n_bins=10)
    assert psi is not None
    assert psi > 0.5, f"Expected significant drift PSI, got {psi}"


def test_psi_insufficient_data_returns_none() -> None:
    from limit_up_board.lgb.evaluate import _psi

    psi = _psi(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), n_bins=10)
    assert psi is None


def test_psi_status_thresholds() -> None:
    from limit_up_board.lgb.evaluate import _psi_status

    assert _psi_status(0.05) == "stable"
    assert _psi_status(0.15) == "moderate"
    assert _psi_status(0.30) == "shift"
    assert _psi_status(None) == "insufficient_data"


def test_compute_drift_returns_sorted_features() -> None:
    from limit_up_board.lgb.evaluate import compute_drift

    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "stable_a": rng.normal(0, 1, 500),
            "stable_b": rng.normal(0, 1, 500),
            "drifted":  rng.normal(0, 1, 500),
        }
    )
    cur = pd.DataFrame(
        {
            "stable_a": rng.normal(0, 1, 500),
            "stable_b": rng.normal(0, 1, 500),
            "drifted":  rng.normal(3, 1, 500),
        }
    )
    res = compute_drift(
        baseline_feature_matrix=base,
        current_feature_matrix=cur,
        baseline_model_id="20260530_demo",
        window_start="20260601",
        window_end="20260630",
    )
    assert res.n_features_compared == 3
    # Drifted feature should bubble to the top.
    assert res.features[0].feature == "drifted"
    assert res.features[0].status == "shift"
    # Stable features after the drifted one.
    statuses = [f.status for f in res.features]
    assert statuses[0] == "shift"
    assert all(s in ("stable", "moderate") for s in statuses[1:])


def test_compute_drift_no_overlap_returns_note() -> None:
    from limit_up_board.lgb.evaluate import compute_drift

    base = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    cur = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
    res = compute_drift(
        baseline_feature_matrix=base,
        current_feature_matrix=cur,
        baseline_model_id="m",
        window_start="20260601",
        window_end="20260630",
    )
    assert res.n_features_compared == 0
    assert res.features == []
    assert any("no overlapping" in n for n in res.notes)


def test_format_drift_table_truncates() -> None:
    from limit_up_board.lgb.evaluate import (
        DriftResult,
        FeatureDrift,
        format_drift_table,
    )

    features = [
        FeatureDrift(
            feature=f"f{i}",
            psi=0.5 - 0.01 * i,
            baseline_mean=0.0,
            current_mean=0.5,
            baseline_std=1.0,
            current_std=1.0,
            n_baseline=100,
            n_current=100,
            status="shift",
        )
        for i in range(25)
    ]
    res = DriftResult(
        baseline_model_id="m",
        window_start="20260601",
        window_end="20260630",
        n_features_compared=25,
        features=features,
    )
    text = format_drift_table(res, top_n=10)
    assert "Feature drift" in text
    assert "+15 more" in text


def test_load_baseline_feature_matrix_missing(tmp_path) -> None:
    from limit_up_board.lgb.evaluate import load_baseline_feature_matrix

    assert load_baseline_feature_matrix(tmp_path / "nope.parquet") is None
