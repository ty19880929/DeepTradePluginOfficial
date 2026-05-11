"""LightGBM 模型版本登记 — ``lub_lgb_models`` 表的 CRUD。

设计文档 §3.5：

* 一行 = 一个落盘模型 booster 的元数据快照；
* 同一时刻最多一行 ``is_active=TRUE``，由 :func:`set_active` 在单事务内
  完成切换；
* 训练 / 激活 / prune 三个 CLI 命令都走本模块。

注：表 schema 由 ``migrations/20260601_001_lgb_tables.sql`` 创建。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    """A single row in ``lub_lgb_models``."""

    model_id: str
    schema_version: int
    train_start_date: str
    train_end_date: str
    n_samples: int
    n_positive: int
    feature_count: int
    feature_list_json: str
    hyperparams_json: str
    plugin_version: str
    file_path: str

    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None
    cv_logloss_mean: float | None = None
    framework_version: str | None = None
    git_commit: str | None = None
    is_active: bool = False
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_record(row: tuple[Any, ...]) -> ModelRecord:
    """Map a SELECT * row → ModelRecord."""
    (
        model_id,
        schema_version,
        train_start_date,
        train_end_date,
        n_samples,
        n_positive,
        cv_auc_mean,
        cv_auc_std,
        cv_logloss_mean,
        feature_count,
        feature_list_json,
        hyperparams_json,
        framework_version,
        plugin_version,
        git_commit,
        file_path,
        is_active,
        created_at,
    ) = row
    return ModelRecord(
        model_id=str(model_id),
        schema_version=int(schema_version),
        train_start_date=str(train_start_date),
        train_end_date=str(train_end_date),
        n_samples=int(n_samples),
        n_positive=int(n_positive),
        cv_auc_mean=None if cv_auc_mean is None else float(cv_auc_mean),
        cv_auc_std=None if cv_auc_std is None else float(cv_auc_std),
        cv_logloss_mean=None if cv_logloss_mean is None else float(cv_logloss_mean),
        feature_count=int(feature_count),
        feature_list_json=str(feature_list_json),
        hyperparams_json=str(hyperparams_json),
        framework_version=None if framework_version is None else str(framework_version),
        plugin_version=str(plugin_version),
        git_commit=None if git_commit is None else str(git_commit),
        file_path=str(file_path),
        is_active=bool(is_active),
        created_at=created_at if isinstance(created_at, datetime) else None,
    )


_SELECT_ALL_COLS = (
    "model_id, schema_version, train_start_date, train_end_date, "
    "n_samples, n_positive, cv_auc_mean, cv_auc_std, cv_logloss_mean, "
    "feature_count, feature_list_json, hyperparams_json, framework_version, "
    "plugin_version, git_commit, file_path, is_active, created_at"
)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def insert_model(db: Database, record: ModelRecord, *, activate: bool = True) -> None:
    """Insert a row + optionally flip ``is_active`` to this model in one transaction."""
    with db.transaction():
        if activate:
            db.execute("UPDATE lub_lgb_models SET is_active = FALSE WHERE is_active = TRUE")
        db.execute(
            "INSERT INTO lub_lgb_models("
            "model_id, schema_version, train_start_date, train_end_date, "
            "n_samples, n_positive, cv_auc_mean, cv_auc_std, cv_logloss_mean, "
            "feature_count, feature_list_json, hyperparams_json, framework_version, "
            "plugin_version, git_commit, file_path, is_active"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.model_id,
                record.schema_version,
                record.train_start_date,
                record.train_end_date,
                record.n_samples,
                record.n_positive,
                record.cv_auc_mean,
                record.cv_auc_std,
                record.cv_logloss_mean,
                record.feature_count,
                record.feature_list_json,
                record.hyperparams_json,
                record.framework_version,
                record.plugin_version,
                record.git_commit,
                record.file_path,
                bool(activate),
            ),
        )


def set_active(db: Database, model_id: str) -> bool:
    """Atomic toggle of the active model. Returns False when id not found."""
    with db.transaction():
        row = db.fetchone("SELECT 1 FROM lub_lgb_models WHERE model_id = ?", (model_id,))
        if row is None:
            return False
        db.execute("UPDATE lub_lgb_models SET is_active = FALSE WHERE is_active = TRUE")
        db.execute("UPDATE lub_lgb_models SET is_active = TRUE WHERE model_id = ?", (model_id,))
        return True


def deactivate_all(db: Database) -> None:
    """Force all rows to is_active = FALSE (operational kill-switch)."""
    db.execute("UPDATE lub_lgb_models SET is_active = FALSE")


def list_models(db: Database) -> list[ModelRecord]:
    rows = db.fetchall(
        f"SELECT {_SELECT_ALL_COLS} FROM lub_lgb_models ORDER BY created_at DESC"
    )
    return [_row_to_record(r) for r in rows]


def get_model(db: Database, model_id: str) -> ModelRecord | None:
    row = db.fetchone(
        f"SELECT {_SELECT_ALL_COLS} FROM lub_lgb_models WHERE model_id = ?",
        (model_id,),
    )
    return None if row is None else _row_to_record(row)


def get_active(db: Database) -> ModelRecord | None:
    row = db.fetchone(
        f"SELECT {_SELECT_ALL_COLS} FROM lub_lgb_models "
        f"WHERE is_active = TRUE ORDER BY created_at DESC LIMIT 1"
    )
    return None if row is None else _row_to_record(row)


def delete_model(db: Database, model_id: str) -> bool:
    """Delete the model row. Returns True iff a row was removed."""
    with db.transaction():
        existing = db.fetchone("SELECT 1 FROM lub_lgb_models WHERE model_id = ?", (model_id,))
        if existing is None:
            return False
        db.execute("DELETE FROM lub_lgb_models WHERE model_id = ?", (model_id,))
        return True


# ---------------------------------------------------------------------------
# Model-id minting
# ---------------------------------------------------------------------------


@dataclass
class _MintInputs:
    train_end_date: str
    schema_version: int
    git_commit: str | None
    extras: list[str] = field(default_factory=list)


def mint_model_id(
    *,
    train_end_date: str,
    schema_version: int,
    git_commit: str | None,
) -> str:
    """``<YYYYMMDD>_<schema_version>_<git_short_or_nogit>`` per design §3.4。

    被 train CLI 调用前会再追加一个递增后缀，确保同一天多次训练不冲突。
    """
    short = git_commit if git_commit else "nogit"
    return f"{train_end_date}_{schema_version}_{short}"


def ensure_unique_model_id(db: Database, base_id: str) -> str:
    """同一天多次训练 → 追加 ``-2`` / ``-3`` …直到 unique。"""
    if db.fetchone("SELECT 1 FROM lub_lgb_models WHERE model_id = ?", (base_id,)) is None:
        return base_id
    n = 2
    while True:
        candidate = f"{base_id}-{n}"
        if db.fetchone("SELECT 1 FROM lub_lgb_models WHERE model_id = ?", (candidate,)) is None:
            return candidate
        n += 1
        if n > 999:
            raise RuntimeError(f"unable to mint unique model_id starting from {base_id!r}")
