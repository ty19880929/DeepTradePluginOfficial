"""LightGBM 资产清理（destructive）。

与 :mod:`registry` 的 ``prune`` 不同：``prune`` 是日常维护（保留最近 N 个 +
active），``cleanup.purge_lgb_artifacts`` 是"scorched earth"——按调用方
指定的范围彻底清空。所有 destructive 行为通过本模块的 dataclass-style 参数
显式表达，避免 CLI 层直接散落 ``DELETE FROM`` / ``unlink`` 调用。

调用方契约
----------

* DB 操作放在一个 ``db.transaction()`` 里，失败时整体回滚——避免出现
  "DB 行删了但文件还在"的半状态。
* 文件删除按"尽力而为"逐文件单独 try/except，统计实际删除的数量；磁盘
  权限错误不阻塞剩余清理。
* 返回 ``PurgeReport`` 让 CLI / 测试两侧都能确认副作用范围。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import paths as lgb_paths

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)


@dataclass
class PurgeReport:
    """Summary of what :func:`purge_lgb_artifacts` actually removed."""

    n_model_files: int = 0
    n_meta_files: int = 0
    n_dataset_files: int = 0
    n_model_rows: int = 0
    n_prediction_rows: int = 0
    latest_pointer_removed: bool = False
    n_checkpoint_dirs: int = 0
    n_checkpoint_shards: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def total_files_removed(self) -> int:
        return (
            self.n_model_files
            + self.n_meta_files
            + self.n_dataset_files
            + self.n_checkpoint_shards
            + (1 if self.latest_pointer_removed else 0)
        )


def _delete_dir_glob(directory: Path, pattern: str) -> tuple[int, list[str]]:
    """Delete files matching ``pattern`` under ``directory``. Returns ``(n, errors)``."""
    if not directory.is_dir():
        return 0, []
    n = 0
    errs: list[str] = []
    for f in directory.glob(pattern):
        if not f.is_file():
            continue
        try:
            f.unlink()
            n += 1
        except OSError as e:  # noqa: BLE001 — keep going on per-file failures
            errs.append(f"unlink {f.name}: {e}")
    return n, errs


def _safe_count(db: Database, sql: str) -> int:
    try:
        row = db.fetchone(sql)
    except Exception:  # noqa: BLE001 — table missing on legacy DBs is fine
        return 0
    if row is None:
        return 0
    return int(row[0] or 0)


def count_artifacts(db: Database) -> PurgeReport:
    """Inspect what *would* be removed by a full purge. Side-effect free."""
    from . import checkpoint as lgb_checkpoint  # noqa: PLC0415

    report = PurgeReport()
    models_dir = lgb_paths.models_dir()
    datasets_dir = lgb_paths.datasets_dir()
    if models_dir.is_dir():
        report.n_model_files = sum(
            1 for _ in models_dir.glob("lgb_model_*.txt") if _.is_file()
        )
        report.n_meta_files = sum(
            1 for _ in models_dir.glob("lgb_model_*.meta.json") if _.is_file()
        )
        report.latest_pointer_removed = lgb_paths.latest_pointer().is_file()
    if datasets_dir.is_dir():
        report.n_dataset_files = sum(
            1 for _ in datasets_dir.glob("lgb_dataset_*.parquet") if _.is_file()
        )
    n_ck, n_shards = lgb_checkpoint.count_checkpoints()
    report.n_checkpoint_dirs = n_ck
    report.n_checkpoint_shards = n_shards
    report.n_model_rows = _safe_count(db, "SELECT COUNT(*) FROM lub_lgb_models")
    report.n_prediction_rows = _safe_count(
        db, "SELECT COUNT(*) FROM lub_lgb_predictions"
    )
    return report


def purge_lgb_artifacts(
    db: Database,
    *,
    datasets: bool = False,
    models: bool = False,
    predictions: bool = False,
    checkpoints: bool = False,
) -> PurgeReport:
    """Erase the requested LGB artifact sets.

    Parameters
    ----------
    datasets
        Delete training-matrix parquet snapshots in ``datasets/``.
    models
        Delete model files + meta.json + ``latest.txt`` and truncate
        ``lub_lgb_models``. After this the scorer enters the
        "no_active_model" branch on the next run.
    predictions
        Truncate ``lub_lgb_predictions`` (per-run scoring audit history).
    checkpoints
        Delete all in-progress training checkpoints under ``checkpoints/``.
        典型用于 "上次 train 崩了，磁盘上留了 shard，但我想重训"——本旗标
        把 ``checkpoints/<digest>/`` 全部目录一并清掉。

    At least one flag must be True; otherwise the call is a no-op and the
    returned report has all-zero counts.

    Returns
    -------
    PurgeReport
        Counts of files removed / DB rows truncated, plus a non-fatal
        ``errors`` list (per-file IO failures, etc.).
    """
    from . import checkpoint as lgb_checkpoint  # noqa: PLC0415

    report = PurgeReport()
    if not (datasets or models or predictions or checkpoints):
        return report

    # DB side first (single transaction so a half-state can't survive).
    try:
        with db.transaction():
            if models:
                row = db.fetchone("SELECT COUNT(*) FROM lub_lgb_models")
                report.n_model_rows = int((row[0] if row else 0) or 0)
                db.execute("DELETE FROM lub_lgb_models")
            if predictions:
                row = db.fetchone("SELECT COUNT(*) FROM lub_lgb_predictions")
                report.n_prediction_rows = int((row[0] if row else 0) or 0)
                db.execute("DELETE FROM lub_lgb_predictions")
    except Exception as e:  # noqa: BLE001
        report.errors.append(f"db purge: {type(e).__name__}: {e}")
        logger.warning("purge_lgb_artifacts: DB cleanup failed: %s", e)

    # File side.
    if models:
        n_models, errs = _delete_dir_glob(lgb_paths.models_dir(), "lgb_model_*.txt")
        report.n_model_files = n_models
        report.errors.extend(errs)
        n_meta, errs = _delete_dir_glob(
            lgb_paths.models_dir(), "lgb_model_*.meta.json"
        )
        report.n_meta_files = n_meta
        report.errors.extend(errs)
        latest = lgb_paths.latest_pointer()
        if latest.is_file():
            try:
                latest.unlink()
                report.latest_pointer_removed = True
            except OSError as e:
                report.errors.append(f"unlink latest.txt: {e}")

    if datasets:
        n_ds, errs = _delete_dir_glob(
            lgb_paths.datasets_dir(), "lgb_dataset_*.parquet"
        )
        report.n_dataset_files = n_ds
        report.errors.extend(errs)

    if checkpoints:
        # 在删除前快照一下要被清掉的 shard 数（删除后无法再数）
        _, n_shards = lgb_checkpoint.count_checkpoints()
        n_ck, errs = lgb_checkpoint.purge_all_checkpoints()
        report.n_checkpoint_dirs = n_ck
        report.n_checkpoint_shards = n_shards
        report.errors.extend(errs)

    return report
