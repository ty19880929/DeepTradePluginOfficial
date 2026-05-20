"""LightGBM training over a fully-assembled :class:`ApwLgbDataset`.

The trainer is intentionally narrow:

1. Drop unlabeled rows (NaN ``label``).
2. ``GroupKFold(n_splits=folds)`` keyed by ``signal_date`` so same-day
   samples never split across train/val — the only way to avoid temporal
   leakage in this regime.
3. Fit ``lightgbm.LGBMClassifier`` with mild defaults; collect per-fold
   AUC / log-loss.
4. Refit on all labeled rows; ``booster.save_model(file_path)``.
5. Save the training matrix as parquet next to the booster so future
   ``lgb evaluate --drift`` can PSI it.
6. Insert a :class:`ModelRecord` via :func:`registry.insert_model`.

The trainer is *not* responsible for Phase-1 data collection — caller hands
in a :class:`ApwLgbDataset` already produced by ``dataset.collect_training_window``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, SCHEMA_VERSION
from .paths import datasets_dir, models_dir
from . import registry as _registry

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from .dataset import ApwLgbDataset

logger = logging.getLogger(__name__)


_DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "n_estimators": 200,
    "random_state": 42,
    "verbose": -1,
}


@dataclass
class TrainResult:
    """Return value from :func:`train_lightgbm`."""

    model_id: str
    record: "_registry.ModelRecord"
    cv_auc_mean: float
    cv_auc_std: float
    cv_logloss_mean: float
    n_samples: int
    n_positive: int
    booster_path: Path
    dataset_path: Path


class LgbTrainError(RuntimeError):
    """Raised when the dataset is too small / too imbalanced to train."""


def train_lightgbm(
    db: "Database",
    *,
    dataset: "ApwLgbDataset",
    cfg: Any,
    plugin_version: str,
    activate: bool = True,
    hyperparam_overrides: dict[str, Any] | None = None,
) -> TrainResult:
    """Fit + register a new APW LGB booster."""
    import lightgbm as lgb  # noqa: PLC0415 — heavy dep, defer
    from sklearn.metrics import log_loss, roc_auc_score  # noqa: PLC0415
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415

    labels = dataset.labels
    mask = labels.notna() & labels.isin([0, 1])
    if not mask.any():
        raise LgbTrainError(
            "no labeled rows in training window — did `evaluate` run?"
        )

    features = dataset.feature_matrix.loc[mask].reset_index(drop=True)
    y = labels.loc[mask].astype(int).reset_index(drop=True)
    groups = dataset.split_groups.loc[mask].astype(str).reset_index(drop=True)

    n_samples = len(features)
    n_positive = int((y == 1).sum())
    min_samples = int(getattr(cfg, "lgb_train_min_samples", 500))
    if n_samples < min_samples:
        raise LgbTrainError(
            f"only {n_samples} labeled samples, need >= {min_samples} "
            f"(tune lgb_train_min_samples to override)"
        )
    if n_positive == 0 or n_positive == n_samples:
        raise LgbTrainError(
            f"degenerate label distribution: positives={n_positive}/{n_samples}"
        )

    # Sanity: training matrix must carry the canonical FEATURE_NAMES order.
    if list(features.columns) != list(FEATURE_NAMES):
        # reindex defensively — features are produced by build_feature_frame
        # so this should already hold; the assert just guards future drift.
        features = features.reindex(columns=FEATURE_NAMES)

    folds = max(2, int(getattr(cfg, "lgb_train_folds", 5)))
    folds = min(folds, max(2, len(groups.unique())))

    hyper = dict(_DEFAULT_HYPERPARAMS)
    if hyperparam_overrides:
        hyper.update(hyperparam_overrides)

    # ---- CV
    gkf = GroupKFold(n_splits=folds)
    aucs: list[float] = []
    logs: list[float] = []
    for tr_idx, va_idx in gkf.split(features, y, groups=groups):
        clf = lgb.LGBMClassifier(**hyper)
        clf.fit(features.iloc[tr_idx], y.iloc[tr_idx])
        probs = clf.predict_proba(features.iloc[va_idx])[:, 1]
        try:
            aucs.append(float(roc_auc_score(y.iloc[va_idx], probs)))
        except ValueError:
            # Single-class fold — skip metric, don't bail the whole CV.
            continue
        logs.append(float(log_loss(y.iloc[va_idx], probs, labels=[0, 1])))

    cv_auc_mean = float(np.mean(aucs)) if aucs else float("nan")
    cv_auc_std = float(np.std(aucs)) if aucs else float("nan")
    cv_logloss_mean = float(np.mean(logs)) if logs else float("nan")

    # ---- Refit on all labeled rows
    final = lgb.LGBMClassifier(**hyper)
    final.fit(features, y)

    # ---- Mint id + save artefacts
    git_commit = _git_short_sha()
    base_id = _registry.mint_model_id(
        train_end_date=str(max(dataset.signal_dates) if dataset.signal_dates else "00000000"),
        schema_version=SCHEMA_VERSION,
        git_commit=git_commit,
    )
    model_id = _registry.ensure_unique_model_id(db, base_id)

    booster_path = models_dir() / f"{model_id}.txt"
    dataset_path = datasets_dir() / f"{model_id}.parquet"
    final.booster_.save_model(str(booster_path))
    # Stash the training matrix (features + label + signal_date) so a future
    # `lgb evaluate --drift` can do per-feature PSI without re-loading the DB.
    snap = features.copy()
    snap["__label__"] = y.values
    snap["__signal_date__"] = groups.values
    snap.to_parquet(dataset_path, index=False)

    record = _registry.ModelRecord(
        model_id=model_id,
        schema_version=SCHEMA_VERSION,
        train_start_date=str(min(dataset.signal_dates)) if dataset.signal_dates else "",
        train_end_date=str(max(dataset.signal_dates)) if dataset.signal_dates else "",
        n_samples=n_samples,
        n_positive=n_positive,
        cv_auc_mean=None if np.isnan(cv_auc_mean) else cv_auc_mean,
        cv_auc_std=None if np.isnan(cv_auc_std) else cv_auc_std,
        cv_logloss_mean=None if np.isnan(cv_logloss_mean) else cv_logloss_mean,
        feature_count=len(FEATURE_NAMES),
        feature_list_json=json.dumps(FEATURE_NAMES, ensure_ascii=False),
        hyperparams_json=json.dumps(hyper, ensure_ascii=False, sort_keys=True),
        label_source=dataset.label_source,
        label_threshold_pct=dataset.label_threshold_pct,
        framework_version=_framework_version(),
        plugin_version=plugin_version,
        git_commit=git_commit,
        file_path=str(booster_path),
    )
    _registry.insert_model(db, record, activate=activate)

    return TrainResult(
        model_id=model_id,
        record=record,
        cv_auc_mean=cv_auc_mean,
        cv_auc_std=cv_auc_std,
        cv_logloss_mean=cv_logloss_mean,
        n_samples=n_samples,
        n_positive=n_positive,
        booster_path=booster_path,
        dataset_path=dataset_path,
    )


# ---------------------------------------------------------------------------
# Auxiliaries
# ---------------------------------------------------------------------------


def _git_short_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        sha = out.stdout.strip()
        return sha or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _framework_version() -> str | None:
    try:
        import deeptrade  # noqa: PLC0415

        return getattr(deeptrade, "__version__", None)
    except Exception:  # noqa: BLE001 — best-effort metadata
        return None
