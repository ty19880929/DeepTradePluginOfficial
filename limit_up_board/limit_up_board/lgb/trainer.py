"""LightGBM 训练：拟合 + GroupKFold 交叉验证 + 落盘。

设计文档 §6.2 / §6.3 / §6.4。本模块依赖 ``lightgbm`` + ``scikit-learn``，
仅在调用 :func:`train_lightgbm` 时才 import，**模块顶层不引入硬依赖**——
让无 lightgbm 环境也能 import :mod:`limit_up_board.lgb`（features / labels /
dataset 不依赖 LGB）。

输入：:class:`LgbDataset`（已 filter_labeled）
输出：:class:`TrainResult`（拟合后的 booster + CV 指标 + 特征重要性 +
      hyperparam 快照）

注意事项
--------
* ``num_boost_round`` 上限 1500；``early_stopping_rounds=100``。
* 折数 < 2 时退化到普通 holdout split（仅 smoke 测试用）；正式训练折
  数应 ≥ 3。
* 训练 deterministic=True + seed=42，便于复现。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .dataset import LgbDataset

if TYPE_CHECKING:  # pragma: no cover
    import lightgbm as lgb

logger = logging.getLogger(__name__)


# 设计文档 §6.3 — v1 默认超参。后续若需调参，请新建 schema_version 并重训。
# v0.18.3 (因子过拟合优化方案 §3.1)：收敛模型复杂度——
#   num_leaves 63→31、min_gain_to_split 0→0.02、lambda_l1 0→0.5、lambda_l2 1.0→2.0。
# 不动 SCHEMA_VERSION（仅超参变更，FEATURE_NAMES 未改）；既有落盘模型按旧
# hyperparams 继续推理，需重训后 `lgb activate` 才切到新参。
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "min_gain_to_split": 0.02,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "is_unbalance": True,
    "deterministic": True,
    "seed": 42,
    "force_col_wise": True,
}

DEFAULT_NUM_BOOST_ROUND: int = 1500
DEFAULT_EARLY_STOPPING_ROUNDS: int = 100
CV_SCHEMES: tuple[str, ...] = ("groupkfold", "walk_forward", "purged")
DEFAULT_EMBARGO_DAYS: int = 1


@dataclass
class TrainResult:
    """完整训练产出。"""

    model: Any  # lightgbm.Booster — 不在顶层 import lgb，保持软依赖
    n_samples: int
    n_positive: int
    cv_auc_mean: float | None
    cv_auc_std: float | None
    cv_logloss_mean: float | None
    train_auc: float | None
    feature_importance: list[tuple[str, float]]
    feature_health: list[dict[str, Any]] = field(default_factory=list)
    hyperparams: dict[str, Any] = field(default_factory=dict)
    best_iteration: int | None = None
    folds: int = 0
    # P1-2: 最终全量 fit 实际跑的轮次。当 CV 内 early-stopping 给出 mean best_iter
    # 时优先用它；否则回退到 ``num_boost_round`` 上限（1500）。值同样会写入
    # ``hyperparams_json`` 便于 `lgb info` 输出。
    final_num_boost_round: int = 0
    # v0.13.1 (P2-2)：CV out-of-fold 拟合的 isotonic 校准器 + 校准前后 Brier。
    # CV 被跳过 / 全单类 / sklearn 缺包时校准器为 None，scorer 自然走 raw 路径。
    calibrator: Any | None = None
    calibration_method: str | None = None
    calibration_brier: float | None = None  # 校准 *后* 的 Brier（写库）
    calibration_brier_pre: float | None = None  # 仅用于 CLI 输出对比
    calibration_samples: int = 0

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        return self.feature_importance[:n]


def _import_lgb() -> Any:
    """Lazy-import lightgbm；缺包时抛友好错误。"""
    try:
        import lightgbm as lgb  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "lightgbm 未安装：pip install 'lightgbm>=4.3'（限定 limit-up-board "
            "v0.5+ 的 train CLI 与 LgbScorer 依赖）"
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


def _ordered_trade_dates(dataset: LgbDataset) -> list[str]:
    if "trade_date" in dataset.sample_index.columns:
        values = dataset.sample_index["trade_date"].astype(str).tolist()
    else:
        values = dataset.split_groups.astype(str).tolist()
    return sorted(set(values))


def _date_positions(dataset: LgbDataset) -> tuple[np.ndarray, dict[str, int]]:
    ordered = _ordered_trade_dates(dataset)
    pos_by_date = {d: i for i, d in enumerate(ordered)}
    dates = dataset.sample_index["trade_date"].astype(str).to_numpy()
    positions = np.asarray([pos_by_date[d] for d in dates], dtype=int)
    return positions, pos_by_date


def _walk_forward_splits(
    dataset: LgbDataset,
    *,
    folds: int,
    embargo_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits with validation dates strictly after training."""
    ordered = _ordered_trade_dates(dataset)
    if folds < 2 or len(ordered) < folds + 1:
        return []
    positions, _pos_by_date = _date_positions(dataset)
    date_chunks = [list(chunk) for chunk in np.array_split(np.asarray(ordered), folds + 1)]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for chunk_idx in range(1, len(date_chunks)):
        val_dates = date_chunks[chunk_idx]
        if not val_dates:
            continue
        val_start_pos = ordered.index(val_dates[0])
        train_max_pos = val_start_pos - max(0, embargo_days) - 1
        if train_max_pos < 0:
            continue
        val_positions = {ordered.index(d) for d in val_dates}
        train_idx = np.flatnonzero(positions <= train_max_pos)
        val_idx = np.flatnonzero(np.isin(positions, list(val_positions)))
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def _purged_group_splits(
    base_splits: list[tuple[np.ndarray, np.ndarray]],
    dataset: LgbDataset,
    *,
    embargo_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold splits with train rows near validation dates removed."""
    if embargo_days <= 0:
        return base_splits
    positions, _pos_by_date = _date_positions(dataset)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in base_splits:
        val_positions = np.unique(positions[val_idx])
        train_positions = positions[train_idx]
        keep = np.ones(len(train_idx), dtype=bool)
        for val_pos in val_positions:
            keep &= np.abs(train_positions - int(val_pos)) > embargo_days
        purged_train_idx = train_idx[keep]
        if len(purged_train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((purged_train_idx, val_idx))
    return splits


def _make_cv_splits(
    dataset: LgbDataset,
    *,
    folds: int,
    cv_scheme: Literal["groupkfold", "walk_forward", "purged"],
    embargo_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415

    n = len(dataset.feature_matrix)
    if cv_scheme == "walk_forward":
        return _walk_forward_splits(
            dataset,
            folds=folds,
            embargo_days=max(0, embargo_days),
        )

    base_splits = list(
        GroupKFold(n_splits=folds).split(
            np.zeros(n),
            dataset.labels.astype("int").to_numpy(),
            groups=dataset.split_groups.astype("int").to_numpy(),
        )
    )
    if cv_scheme == "purged":
        return _purged_group_splits(
            base_splits,
            dataset,
            embargo_days=max(0, embargo_days),
        )
    return base_splits


def _crossval_metrics(
    lgb_mod: Any,
    dataset: LgbDataset,
    *,
    params: dict[str, Any],
    folds: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    cv_scheme: Literal["groupkfold", "walk_forward", "purged"] = "groupkfold",
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[list[float], list[float], int, np.ndarray]:
    """Return (auc_per_fold, logloss_per_fold, mean_best_iter, oof_preds).

    默认使用 GroupKFold by trade_date，避免同一交易日的样本跨折泄漏。
    ``walk_forward`` 与 ``purged`` 用 embargo 切断 T/T+1 标签邻接泄漏。

    v0.13.1 (P2-2)：``oof_preds`` 是 OOF 拼接预测向量（长度 == 数据集行数）；
    每个 fold 的 val 切片由该 fold 的 booster 推出，组合后用于训练校准器。
    GroupKFold 保证每个样本仅作为某一个 fold 的 val，所以 OOF 没有数据泄漏。
    """
    n = len(dataset.feature_matrix)
    splits = _make_cv_splits(
        dataset,
        folds=folds,
        cv_scheme=cv_scheme,
        embargo_days=embargo_days,
    )
    aucs: list[float] = []
    loglosses: list[float] = []
    best_iters: list[int] = []
    oof_preds = np.full(n, np.nan, dtype="float64")
    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        train_X = dataset.feature_matrix.iloc[train_idx]
        train_y = dataset.labels.iloc[train_idx].astype("int")
        val_X = dataset.feature_matrix.iloc[val_idx]
        val_y = dataset.labels.iloc[val_idx].astype("int")

        train_ds = _build_dataset(
            lgb_mod, train_X, train_y, feature_names=list(dataset.feature_matrix.columns)
        )
        val_ds = _build_dataset(
            lgb_mod, val_X, val_y, feature_names=list(dataset.feature_matrix.columns)
        )

        booster = lgb_mod.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=[lgb_mod.early_stopping(early_stopping_rounds, verbose=False)],
        )
        pred = booster.predict(val_X.to_numpy(dtype="float64"))
        oof_preds[val_idx] = pred
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
    mean_best = int(np.mean(best_iters)) if best_iters else 0
    return aucs, loglosses, mean_best, oof_preds


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sklearn roc_auc_score；单类别 val 集合 → 返回 NaN 避免崩溃。"""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def _safe_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import log_loss  # noqa: PLC0415

    eps = 1e-7
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, y_pred_clipped, labels=[0, 1]))


def _feature_health(
    feature_matrix: Any,
    gain_by_feature: dict[str, float],
    *,
    variance_epsilon: float = 1e-12,
    missing_rate_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Diagnostic-only feature health snapshot; callers must not auto-drop."""
    rows: list[dict[str, Any]] = []
    if feature_matrix.empty:
        return rows
    for name in feature_matrix.columns:
        s = feature_matrix[name].astype("float64")
        missing_rate = float(s.isna().mean())
        variance = float(s.var(skipna=True)) if s.notna().sum() > 1 else 0.0
        gain = float(gain_by_feature.get(name, 0.0))
        reasons: list[str] = []
        if gain <= 0.0:
            reasons.append("zero_gain")
        if variance <= variance_epsilon:
            reasons.append("near_constant")
        if missing_rate >= missing_rate_threshold:
            reasons.append("high_missing")
        if reasons:
            rows.append(
                {
                    "feature": name,
                    "gain": round(gain, 6),
                    "variance": round(variance, 8),
                    "missing_rate": round(missing_rate, 4),
                    "reasons": reasons,
                }
            )
    rows.sort(
        key=lambda r: (
            "zero_gain" in r["reasons"],
            "high_missing" in r["reasons"],
            "near_constant" in r["reasons"],
            r["missing_rate"],
            -r["variance"],
            r["feature"],
        ),
        reverse=True,
    )
    return rows


def train_lightgbm(
    dataset: LgbDataset,
    *,
    folds: int = 5,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    hyperparams: dict[str, Any] | None = None,
    top_importance: int = 20,
    cv_scheme: Literal["groupkfold", "walk_forward", "purged"] = "groupkfold",
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> TrainResult:
    """Fit LightGBM on the entire dataset + run GroupKFold CV for metrics.

    Implementation note
    -------------------
    The returned ``model`` is fit on **all** labeled samples for `num_boost_round`
    rounds (no early stopping when training on the full set — CV already gave us
    a generalization estimate). CV is only used to report AUC / logloss for the
    operator's confidence.
    """
    if dataset.n_samples == 0:
        raise ValueError("dataset is empty — nothing to train")
    if dataset.labels.isna().any():
        raise ValueError(
            "dataset has unlabeled rows — call LgbDataset.filter_labeled() before training"
        )

    lgb_mod = _import_lgb()
    params = {**LGB_PARAMS, **(hyperparams or {})}
    if cv_scheme not in CV_SCHEMES:
        raise ValueError(f"unsupported cv_scheme: {cv_scheme!r}")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")

    feature_names = list(dataset.feature_matrix.columns)

    # ---- CV ----
    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None
    cv_logloss_mean: float | None = None
    mean_best_iter: int = 0
    oof_preds: np.ndarray | None = None
    if folds >= 2 and dataset.split_groups.nunique() >= folds:
        aucs, loglosses, mean_best_iter, oof_preds = _crossval_metrics(
            lgb_mod,
            dataset,
            params=params,
            folds=folds,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            cv_scheme=cv_scheme,
            embargo_days=embargo_days,
        )
        finite_aucs = [a for a in aucs if not np.isnan(a)]
        if finite_aucs:
            cv_auc_mean = float(np.mean(finite_aucs))
            cv_auc_std = float(np.std(finite_aucs))
        if loglosses:
            cv_logloss_mean = float(np.mean(loglosses))
    else:
        logger.warning(
            "Skipping CV (folds=%d, unique groups=%d) — caller should supply more days.",
            folds,
            int(dataset.split_groups.nunique()),
        )

    # ---- Final fit on all labeled data ----
    # P1-2: 优先使用 CV 各折 early-stopping 给出的 mean best_iter；CV 被跳过或
    # 全部折数命中 num_boost_round 上限时回退到 num_boost_round。这样最终模型
    # 不会盲目 fit 1500 轮（过拟合风险 + 训练时间浪费）。
    final_num_boost_round = mean_best_iter if mean_best_iter > 0 else num_boost_round
    logger.info(
        "final fit num_boost_round=%d (cv mean_best_iter=%d, ceiling=%d)",
        final_num_boost_round,
        mean_best_iter,
        num_boost_round,
    )
    full_ds = _build_dataset(
        lgb_mod,
        dataset.feature_matrix,
        dataset.labels.astype("int"),
        feature_names=feature_names,
    )
    model = lgb_mod.train(
        params,
        full_ds,
        num_boost_round=final_num_boost_round,
    )

    train_pred = model.predict(dataset.feature_matrix.to_numpy(dtype="float64"))
    train_auc = _safe_auc(dataset.labels.astype("int").to_numpy(), train_pred)

    # ---- Feature importance ----
    importance_all = list(
        zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain"),
            strict=False,
        )
    )
    importance_all.sort(key=lambda kv: kv[1], reverse=True)
    gain_by_feature = {name: float(score) for name, score in importance_all}
    feature_health = _feature_health(dataset.feature_matrix, gain_by_feature)
    importance = list(importance_all)
    importance.sort(key=lambda kv: kv[1], reverse=True)
    if top_importance:
        importance = importance[:top_importance]
    # cast np.int → float for JSON friendliness
    importance_typed: list[tuple[str, float]] = [(name, float(score)) for name, score in importance]

    # 把 final_num_boost_round 一并写入 hyperparams 快照，下游 `lgb info`
    # 直接展示 hyperparams_json 即能反映"全量 fit 实际跑了多少轮"。
    hyperparams_snapshot = {
        **params,
        "final_num_boost_round": final_num_boost_round,
        "cv_scheme": cv_scheme,
        "embargo_days": embargo_days,
    }

    # ---- Calibrator (P2-2)：用 OOF 预测拟合 isotonic regression ----
    calibrator: Any | None = None
    calibration_method: str | None = None
    calibration_brier: float | None = None
    calibration_brier_pre: float | None = None
    calibration_samples: int = 0
    if oof_preds is not None:
        from .calibration import train_isotonic_calibrator  # noqa: PLC0415

        calibrator, brier_pre, brier_post, n_used = train_isotonic_calibrator(
            dataset.labels.astype("int").to_numpy(),
            oof_preds,
        )
        calibration_brier_pre = brier_pre
        calibration_samples = n_used
        if calibrator is not None:
            calibration_method = "isotonic"
            calibration_brier = brier_post
            logger.info(
                "calibrator fitted on %d OOF samples — Brier pre=%.4f post=%.4f",
                n_used,
                brier_pre if brier_pre is not None else float("nan"),
                brier_post if brier_post is not None else float("nan"),
            )
        else:
            logger.warning(
                "calibrator skipped (single-class or sklearn missing); "
                "scorer will fall back to raw predictions",
            )

    return TrainResult(
        model=model,
        n_samples=dataset.n_samples,
        n_positive=dataset.n_positive,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_logloss_mean=cv_logloss_mean,
        train_auc=train_auc,
        feature_importance=importance_typed,
        feature_health=feature_health,
        hyperparams=hyperparams_snapshot,
        best_iteration=getattr(model, "best_iteration", None),
        folds=folds,
        final_num_boost_round=final_num_boost_round,
        calibrator=calibrator,
        calibration_method=calibration_method,
        calibration_brier=calibration_brier,
        calibration_brier_pre=calibration_brier_pre,
        calibration_samples=calibration_samples,
    )
