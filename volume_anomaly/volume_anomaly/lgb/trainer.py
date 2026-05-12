"""LightGBM 训练：拟合 + GroupKFold 交叉验证 + 落盘 (PR-1.3).

设计文档 §6.2 / §6.3 / §6.4。本模块依赖 ``lightgbm`` + ``scikit-learn``，
仅在调用 :func:`train_lightgbm` 时才 import，**模块顶层不引入硬依赖**——
让无 lightgbm 环境也能 import :mod:`volume_anomaly.lgb` (features / labels /
dataset 不依赖 LGB)。

输入：:class:`VaLgbDataset`（已 filter_labeled）
输出：:class:`TrainResult`（拟合后的 booster + CV 指标 + 特征重要性 +
      hyperparam 快照）

设计与 ``limit-up-board`` 的同名模块基本对齐；VA 仅在调用方传入
``VaLgbDataset`` 而非 ``LgbDataset``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import VaLgbDataset

logger = logging.getLogger(__name__)


# 设计文档 §6.4 — v1 默认超参；如需调整请同时 bump SCHEMA_VERSION。
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "is_unbalance": True,
    "deterministic": True,
    "seed": 42,
    "force_col_wise": True,
}

DEFAULT_NUM_BOOST_ROUND: int = 1500
DEFAULT_EARLY_STOPPING_ROUNDS: int = 100


@dataclass
class TrainResult:
    """Output of one ``train_lightgbm`` invocation."""

    model: Any  # lightgbm.Booster — kept untyped to preserve soft dependency
    n_samples: int
    n_positive: int
    cv_auc_mean: float | None
    cv_auc_std: float | None
    cv_logloss_mean: float | None
    feature_importance: list[tuple[str, float]]
    hyperparams: dict[str, Any] = field(default_factory=dict)
    best_iteration: int | None = None
    folds: int = 0

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        return self.feature_importance[:n]


def _import_lgb() -> Any:
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "lightgbm 未安装：pip install 'lightgbm>=4.3'（v0.7+ 的 lgb train "
            "与 LgbScorer 依赖）"
        ) from e
    return lgb


def _build_dataset(
    lgb_mod: Any,
    feature_matrix: Any,
    labels: Any,
    *,
    feature_names: list[str],
) -> Any:
    return lgb_mod.Dataset(
        feature_matrix.to_numpy(dtype="float64"),
        label=labels.to_numpy(dtype="float64"),
        feature_name=feature_names,
        free_raw_data=False,
    )


def _crossval_metrics(
    lgb_mod: Any,
    dataset: VaLgbDataset,
    *,
    params: dict[str, Any],
    folds: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[list[float], list[float], int]:
    """Return ``(auc_per_fold, logloss_per_fold, mean_best_iter)``.

    GroupKFold by anomaly_date — same anomaly_date never spans train+val.
    """
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415

    splits = list(
        GroupKFold(n_splits=folds).split(
            np.zeros(len(dataset.feature_matrix)),
            dataset.labels.astype("int").to_numpy(),
            groups=dataset.split_groups.astype("int").to_numpy(),
        )
    )
    aucs: list[float] = []
    loglosses: list[float] = []
    best_iters: list[int] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_X = dataset.feature_matrix.iloc[train_idx]
        train_y = dataset.labels.iloc[train_idx].astype("int")
        val_X = dataset.feature_matrix.iloc[val_idx]
        val_y = dataset.labels.iloc[val_idx].astype("int")

        train_ds = _build_dataset(
            lgb_mod,
            train_X,
            train_y,
            feature_names=list(dataset.feature_matrix.columns),
        )
        val_ds = _build_dataset(
            lgb_mod,
            val_X,
            val_y,
            feature_names=list(dataset.feature_matrix.columns),
        )

        booster = lgb_mod.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=[
                lgb_mod.early_stopping(early_stopping_rounds, verbose=False)
            ],
        )
        pred = booster.predict(val_X.to_numpy(dtype="float64"))
        aucs.append(_safe_auc(val_y.to_numpy(), pred))
        loglosses.append(_safe_logloss(val_y.to_numpy(), pred))
        best_iters.append(int(booster.best_iteration or num_boost_round))
        logger.info(
            "fold %d/%d: AUC=%.4f logloss=%.4f best_iter=%d",
            fold_idx,
            folds,
            aucs[-1],
            loglosses[-1],
            best_iters[-1],
        )
    return aucs, loglosses, int(np.mean(best_iters)) if best_iters else 0


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def _safe_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import log_loss  # noqa: PLC0415

    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, y_pred_clipped, labels=[0, 1]))


def train_lightgbm(
    dataset: VaLgbDataset,
    *,
    folds: int = 5,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    hyperparams: dict[str, Any] | None = None,
    top_importance: int = 20,
) -> TrainResult:
    """Fit LightGBM on the whole dataset + run GroupKFold CV for metrics.

    The returned ``model`` is fit on **all** labeled samples for
    ``num_boost_round`` rounds (no early stopping on the final fit; CV gave
    us the generalization estimate already).
    """
    if dataset.n_samples == 0:
        raise ValueError("dataset is empty — nothing to train")
    if dataset.labels.isna().any():
        raise ValueError(
            "dataset has unlabeled rows — call VaLgbDataset.filter_labeled() "
            "before training"
        )

    lgb_mod = _import_lgb()
    params = {**LGB_PARAMS, **(hyperparams or {})}
    feature_names = list(dataset.feature_matrix.columns)

    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None
    cv_logloss_mean: float | None = None
    if folds >= 2 and dataset.split_groups.nunique() >= folds:
        aucs, loglosses, _ = _crossval_metrics(
            lgb_mod,
            dataset,
            params=params,
            folds=folds,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        finite_aucs = [a for a in aucs if not np.isnan(a)]
        if finite_aucs:
            cv_auc_mean = float(np.mean(finite_aucs))
            cv_auc_std = float(np.std(finite_aucs))
        if loglosses:
            cv_logloss_mean = float(np.mean(loglosses))
    else:
        logger.warning(
            "Skipping CV (folds=%d, unique groups=%d) — caller should supply "
            "more anomaly_dates.",
            folds,
            int(dataset.split_groups.nunique()),
        )

    full_ds = _build_dataset(
        lgb_mod,
        dataset.feature_matrix,
        dataset.labels.astype("int"),
        feature_names=feature_names,
    )
    model = lgb_mod.train(params, full_ds, num_boost_round=num_boost_round)

    importance = list(
        zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain"),
            strict=False,
        )
    )
    importance.sort(key=lambda kv: kv[1], reverse=True)
    if top_importance:
        importance = importance[:top_importance]
    importance_typed: list[tuple[str, float]] = [
        (name, float(score)) for name, score in importance
    ]

    return TrainResult(
        model=model,
        n_samples=dataset.n_samples,
        n_positive=dataset.n_positive,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_logloss_mean=cv_logloss_mean,
        feature_importance=importance_typed,
        hyperparams=params,
        best_iteration=getattr(model, "best_iteration", None),
        folds=folds,
    )


__all__ = [
    "DEFAULT_EARLY_STOPPING_ROUNDS",
    "DEFAULT_NUM_BOOST_ROUND",
    "LGB_PARAMS",
    "TrainResult",
    "train_lightgbm",
]
