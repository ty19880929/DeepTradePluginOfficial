"""PR-3.1 / PR-3.3 — evaluate metrics (AUC / Top-K) + PSI drift detection.

These exercise the pure functions in ``volume_anomaly.lgb.evaluate`` without
going through Tushare or the full ``evaluate_model`` pipeline — that path is
covered indirectly via the lower-level ``collect_training_window`` tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from volume_anomaly.lgb.evaluate import (
    DEFAULT_K_VALUES,
    PSI_THRESHOLD_SHIFT,
    PSI_THRESHOLD_STABLE,
    DriftResult,
    EvaluateResult,
    FeatureDrift,
    TopKMetrics,
    _compute_topk_metrics,
    _psi,
    _psi_status,
    _safe_auc,
    _safe_logloss,
    compute_drift,
    format_drift_table,
    format_evaluate_table,
)


# ---------------------------------------------------------------------------
# AUC / logloss
# ---------------------------------------------------------------------------


def test_safe_auc_returns_none_for_single_class() -> None:
    assert _safe_auc(np.array([1, 1, 1]), np.array([0.5, 0.6, 0.7])) is None


def test_safe_auc_perfect_ranking() -> None:
    auc = _safe_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert auc == pytest.approx(1.0)


def test_safe_logloss_zero_for_perfect_pred() -> None:
    # eps=1e-7 → not exactly zero but very small.
    ll = _safe_logloss(np.array([0, 1]), np.array([0.0, 1.0]))
    assert ll is not None and ll < 1e-5


# ---------------------------------------------------------------------------
# Top-K aggregation
# ---------------------------------------------------------------------------


def _eval_df() -> pd.DataFrame:
    # Two anomaly_dates. Day 1: 5 candidates (2 labels positive in top-2).
    # Day 2: 5 candidates (1 label positive in top-2).
    rows = []
    for i, (score, label, upside) in enumerate(
        [(0.9, 1, 8.0), (0.8, 1, 6.0), (0.5, 0, 1.0), (0.3, 0, 0.5), (0.1, 0, 0.0)]
    ):
        rows.append(
            {
                "anomaly_date": "20260601",
                "ts_code": f"00000{i}.SZ",
                "lgb_score": score,
                "label": label,
                "max_ret_5d": upside,
            }
        )
    for i, (score, label, upside) in enumerate(
        [(0.95, 0, 2.0), (0.7, 1, 7.0), (0.4, 0, 1.0), (0.3, 0, 0.0), (0.1, 0, -1.0)]
    ):
        rows.append(
            {
                "anomaly_date": "20260602",
                "ts_code": f"10000{i}.SZ",
                "lgb_score": score,
                "label": label,
                "max_ret_5d": upside,
            }
        )
    return pd.DataFrame(rows)


def test_topk_aggregation_known_distribution() -> None:
    metrics = _compute_topk_metrics(_eval_df(), k_values=(2,))
    assert len(metrics) == 1
    tk = metrics[0]
    assert tk.k == 2
    assert tk.n_days_evaluated == 2
    # Day1 top-2 hits = 2; Day2 top-2 hits = 1 (the 0.7 row). Total picks = 4.
    assert tk.hit_count == 3
    assert tk.pick_count == 4
    assert tk.hit_rate_pct == pytest.approx(75.0)
    # Avg upside over 4 picks = (8 + 6 + 2 + 7) / 4 = 5.75.
    assert tk.avg_upside_pct == pytest.approx(5.75)
    # Day-wise baseline upside = (mean Day1, mean Day2) / 2.
    # Day1 mean = (8 + 6 + 1 + 0.5 + 0) / 5 = 3.1; Day2 = (2 + 7 + 1 + 0 + -1) / 5 = 1.8.
    # Mean of day means = 2.45.
    assert tk.baseline_avg_upside_pct == pytest.approx(2.45)


def test_topk_empty_input_returns_zero_rows() -> None:
    metrics = _compute_topk_metrics(
        pd.DataFrame(
            columns=["anomaly_date", "lgb_score", "label", "max_ret_5d"]
        ),
        k_values=(5, 10),
    )
    assert len(metrics) == 2
    assert all(tk.n_days_evaluated == 0 for tk in metrics)
    assert all(tk.hit_rate_pct is None for tk in metrics)


def test_topk_skips_days_with_all_nan_scores() -> None:
    df = pd.DataFrame(
        {
            "anomaly_date": ["20260601"] * 3,
            "ts_code": ["a", "b", "c"],
            "lgb_score": [np.nan, np.nan, np.nan],
            "label": [1, 0, 1],
            "max_ret_5d": [5.0, 1.0, 6.0],
        }
    )
    metrics = _compute_topk_metrics(df, k_values=(2,))
    assert metrics[0].n_days_evaluated == 0
    assert metrics[0].hit_count == 0


def test_default_k_values_match_design() -> None:
    assert DEFAULT_K_VALUES == (5, 10, 20)


# ---------------------------------------------------------------------------
# format_evaluate_table — output shape
# ---------------------------------------------------------------------------


def test_format_evaluate_table_handles_no_scorer() -> None:
    r = EvaluateResult(
        model_id="m1",
        window_start="20260101",
        window_end="20260131",
        label_threshold_pct=5.0,
        label_source="max_ret_5d",
        n_samples=10,
        n_labeled=10,
        n_positive=4,
        n_anomaly_dates=2,
        auc=None,
        logloss=None,
        schema_version_match=False,
        schema_mismatch_detail="no_active_model",
    )
    out = format_evaluate_table(r)
    assert "no_active_model" in out
    assert "AUC" not in out  # short-circuit


def test_format_evaluate_table_includes_topk_lines() -> None:
    r = EvaluateResult(
        model_id="m1",
        window_start="20260101",
        window_end="20260131",
        label_threshold_pct=5.0,
        label_source="max_ret_5d",
        n_samples=20,
        n_labeled=20,
        n_positive=8,
        n_anomaly_dates=4,
        auc=0.65,
        logloss=0.55,
        topk=[
            TopKMetrics(
                k=5,
                n_days_evaluated=4,
                hit_count=10,
                pick_count=20,
                hit_rate_pct=50.0,
                avg_upside_pct=4.5,
                baseline_hit_rate_pct=40.0,
                baseline_avg_upside_pct=3.0,
                delta_hit_rate_pct=10.0,
                delta_avg_upside_pct=1.5,
            )
        ],
    )
    out = format_evaluate_table(r)
    assert "AUC = 0.6500" in out
    assert "logloss = 0.5500" in out
    assert "Top-K" in out
    assert " 5 |" in out


# ---------------------------------------------------------------------------
# PSI (drift) — pure math
# ---------------------------------------------------------------------------


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    assert _psi(x, x.copy()) == pytest.approx(0.0, abs=1e-6)


def test_psi_increases_with_distribution_shift() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 1000)
    shifted = rng.normal(2.0, 1, 1000)
    psi_small = _psi(base, rng.normal(0, 1, 1000))
    psi_large = _psi(base, shifted)
    assert psi_small is not None and psi_large is not None
    assert psi_large > psi_small
    assert psi_large > PSI_THRESHOLD_SHIFT


def test_psi_returns_none_for_small_samples() -> None:
    assert _psi(np.array([1.0, 2.0]), np.array([3.0, 4.0])) is None


def test_psi_status_thresholds() -> None:
    assert _psi_status(None) == "insufficient_data"
    assert _psi_status(PSI_THRESHOLD_STABLE - 0.01) == "stable"
    assert _psi_status(PSI_THRESHOLD_STABLE) == "moderate"
    assert _psi_status(PSI_THRESHOLD_SHIFT) == "shift"


# ---------------------------------------------------------------------------
# compute_drift
# ---------------------------------------------------------------------------


def test_compute_drift_sorts_features_by_psi_desc() -> None:
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "f_stable": rng.normal(0, 1, 500),
            "f_shift": rng.normal(0, 1, 500),
        }
    )
    cur = pd.DataFrame(
        {
            "f_stable": rng.normal(0, 1, 500),     # ~no shift
            "f_shift": rng.normal(2.0, 1, 500),    # large shift
        }
    )
    result = compute_drift(
        baseline_feature_matrix=base,
        current_feature_matrix=cur,
        baseline_model_id="m_base",
        window_start="20260301",
        window_end="20260330",
    )
    assert result.n_features_compared == 2
    # The shifted feature must come first.
    assert result.features[0].feature == "f_shift"
    assert result.features[0].status == "shift"
    assert result.features[1].feature == "f_stable"
    assert result.features[1].status == "stable"


def test_compute_drift_handles_disjoint_columns() -> None:
    base = pd.DataFrame({"a": np.arange(100).astype(float)})
    cur = pd.DataFrame({"b": np.arange(100).astype(float)})
    result = compute_drift(
        baseline_feature_matrix=base,
        current_feature_matrix=cur,
        baseline_model_id="m",
        window_start="20260101",
        window_end="20260131",
    )
    assert result.n_features_compared == 0
    assert "no overlapping feature columns" in result.notes


def test_format_drift_table_handles_empty_features() -> None:
    result = DriftResult(
        baseline_model_id="m",
        window_start="20260101",
        window_end="20260131",
        n_features_compared=0,
        notes=["nothing"],
    )
    out = format_drift_table(result)
    assert "nothing" in out


def test_format_drift_table_truncates_to_top_n() -> None:
    features = [
        FeatureDrift(
            feature=f"f_{i:02d}",
            psi=0.5 - i * 0.01,
            baseline_mean=0.0,
            current_mean=1.0,
            baseline_std=1.0,
            current_std=1.0,
            n_baseline=100,
            n_current=100,
            status="shift",
        )
        for i in range(30)
    ]
    result = DriftResult(
        baseline_model_id="m",
        window_start="20260101",
        window_end="20260131",
        n_features_compared=30,
        features=features,
    )
    out = format_drift_table(result, top_n=10)
    # 10 visible + 1 "(+20 more...)" footer.
    assert "+20 more" in out
    # Only the first 10 feature names should appear directly.
    for i in range(10):
        assert f"f_{i:02d}" in out
    assert "f_29" not in out
