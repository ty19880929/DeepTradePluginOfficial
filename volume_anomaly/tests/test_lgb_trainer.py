"""PR-1.3 — trainer parameter / output format checks.

Light-weight tests that import lightgbm only to verify the wiring (param
dict, CV path, importance ranking); the slow end-to-end "synthetic 50 days"
smoke test lives in ``test_lgb_train_smoke.py`` (marked ``@pytest.mark.slow``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from volume_anomaly.lgb.dataset import VaLgbDataset
from volume_anomaly.lgb.features import FEATURE_NAMES
from volume_anomaly.lgb.trainer import (
    DEFAULT_NUM_BOOST_ROUND,
    LGB_PARAMS,
    TrainResult,
    train_lightgbm,
)


def _synthetic_dataset(
    *, n_days: int = 6, n_per_day: int = 20, seed: int = 0
) -> VaLgbDataset:
    """A small but lightgbm-trainable dataset with a real signal in
    f_vol_anomaly_pct_chg → label."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    meta: list[dict[str, str | float]] = []
    labels: list[int] = []
    anomaly_dates: list[str] = []
    for d in range(n_days):
        anomaly_date = f"2026010{d + 1}" if d < 9 else f"202601{d + 1}"
        anomaly_dates.append(anomaly_date)
        for i in range(n_per_day):
            feat = {col: rng.normal(0, 1) for col in FEATURE_NAMES}
            # Inject a clean signal: high anomaly_pct_chg → likely positive.
            signal = rng.normal(0, 1) + (1 if i % 2 == 0 else -1) * 3.0
            feat["f_vol_anomaly_pct_chg"] = signal
            rows.append(feat)
            labels.append(1 if signal > 0 else 0)
            meta.append(
                {
                    "ts_code": f"{i:06d}.SZ",
                    "anomaly_date": anomaly_date,
                    "max_ret_5d": signal,
                    "data_status": "complete",
                }
            )
    feature_matrix = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float)
    labels_series = pd.Series(labels, dtype="Int64", name="label")
    sample_index = pd.DataFrame(meta)
    split_groups = pd.Series(
        sample_index["anomaly_date"].astype(int), dtype="Int64", name="split_group"
    )
    return VaLgbDataset(
        feature_matrix=feature_matrix.reset_index(drop=True),
        labels=labels_series.reset_index(drop=True),
        sample_index=sample_index.reset_index(drop=True),
        split_groups=split_groups.reset_index(drop=True),
        anomaly_dates=anomaly_dates,
    )


def test_default_params_are_design_compliant() -> None:
    assert LGB_PARAMS["objective"] == "binary"
    assert LGB_PARAMS["seed"] == 42
    assert LGB_PARAMS["deterministic"] is True
    assert LGB_PARAMS["is_unbalance"] is True
    assert "auc" in LGB_PARAMS["metric"]


def test_train_lightgbm_returns_trainresult_with_expected_shape() -> None:
    ds = _synthetic_dataset(n_days=4, n_per_day=15)
    result = train_lightgbm(ds, folds=2, num_boost_round=40, early_stopping_rounds=5)
    assert isinstance(result, TrainResult)
    assert result.n_samples == ds.n_samples
    assert result.folds == 2
    assert result.hyperparams["seed"] == 42
    assert isinstance(result.feature_importance, list)
    # Top features must be (str, float) tuples sorted desc by gain.
    scores = [s for _, s in result.feature_importance]
    assert scores == sorted(scores, reverse=True)


def test_train_lightgbm_learns_signal() -> None:
    ds = _synthetic_dataset(n_days=6, n_per_day=20)
    # min_data_in_leaf default = 80 chokes on the 120-sample fixture; the
    # design default is fine for production scale (≥1000) but we override
    # here so the booster actually splits.
    result = train_lightgbm(
        ds,
        folds=3,
        num_boost_round=80,
        early_stopping_rounds=20,
        hyperparams={"min_data_in_leaf": 5},
    )
    assert result.cv_auc_mean is not None
    assert result.cv_auc_mean > 0.8


def test_train_lightgbm_rejects_empty_dataset() -> None:
    empty = VaLgbDataset(
        feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
        labels=pd.Series([], dtype="Int64", name="label"),
        sample_index=pd.DataFrame(),
        split_groups=pd.Series([], dtype="Int64"),
    )
    with pytest.raises(ValueError, match="empty"):
        train_lightgbm(empty)


def test_train_lightgbm_rejects_unlabeled_rows() -> None:
    ds = _synthetic_dataset(n_days=3, n_per_day=10)
    ds.labels.iloc[0] = pd.NA
    with pytest.raises(ValueError, match="unlabeled"):
        train_lightgbm(ds)


def test_train_lightgbm_falls_back_when_groups_lt_folds() -> None:
    # 2 anomaly_dates but 5 folds requested → CV skipped; final fit still runs.
    ds = _synthetic_dataset(n_days=2, n_per_day=30)
    result = train_lightgbm(ds, folds=5, num_boost_round=30, early_stopping_rounds=5)
    assert result.cv_auc_mean is None
    assert result.cv_auc_std is None
    # Final fit still produces a booster + importance.
    assert result.feature_importance


def test_default_num_boost_round_is_design_default() -> None:
    assert DEFAULT_NUM_BOOST_ROUND == 1500


def test_top_features_returns_at_most_n() -> None:
    ds = _synthetic_dataset(n_days=3, n_per_day=10)
    result = train_lightgbm(ds, folds=2, num_boost_round=30, early_stopping_rounds=5)
    top5 = result.top_features(5)
    assert len(top5) <= 5
    assert all(isinstance(name, str) and not math.isnan(score) for name, score in top5)
