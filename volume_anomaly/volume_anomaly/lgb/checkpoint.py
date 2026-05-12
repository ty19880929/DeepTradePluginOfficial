"""Phase-1 训练数据收集 checkpoint —— 支撑 ``lgb train`` 长窗口续训。

设计动机
--------
``lgb train`` 最耗时的环节是按 anomaly_date 抓取 + 特征化（1 年 ~240 个交易
日 × 多个 Tushare 接口 + 每日 ``build_feature_frame``）。一旦中途崩溃，
既有流程必须从头跑。LightGBM 拟合本身是 ``deterministic=True + seed=42``
的、相对短的 CPU 任务，不在续训范围内。

本模块把每日处理结果按 anomaly_date 落成 parquet shard，附带一份 state.json
记录"指纹"（训练窗口 + 标签参数 + schema_version + lookbacks）和已完成日期
列表；下次同配置启动时跳过磁盘上已有的日，仅补漏。训练成功后整个
checkpoint 目录被清理。

与 ``limit-up-board`` 的同名模块设计同源；VA 的差异主要在 fingerprint
字段（多了 ``label_source`` / ``main_board_only`` / ``baseline_index_code``，
没有 LUB 的市值过滤）和 META_COLUMNS（VA 用 ``anomaly_date / max_ret_5d /
data_status``，LUB 用 ``trade_date / next_trade_date / pct_chg_t1``）。

存储布局
--------
``~/.deeptrade/volume_anomaly/checkpoints/<digest>/``::

    days/<YYYYMMDD>.parquet     # 单日 shard：FEATURE_NAMES + label + meta
    state.json                  # 指纹 + completed_dates + 版本元信息

shard 列顺序固定为 :data:`SHARD_COLUMNS`。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import paths as lgb_paths
from .features import FEATURE_NAMES, SCHEMA_VERSION

logger = logging.getLogger(__name__)


STATE_FILENAME = "state.json"
DAYS_DIRNAME = "days"
SHARD_SUFFIX = ".parquet"

# Per-sample metadata columns kept alongside features + label in each shard.
# Distinct from LUB's META_COLUMNS — VA's label provenance is anomaly_date +
# data_status from va_realized_returns, not (trade_date, next_trade_date,
# pct_chg_t1).
META_COLUMNS: list[str] = [
    "ts_code",
    "anomaly_date",
    "max_ret_5d",
    "data_status",
]
SHARD_COLUMNS: list[str] = FEATURE_NAMES + ["label"] + META_COLUMNS


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointFingerprint:
    """训练参数稳定指纹。

    成员仅包含会**改变 dataset 内容**的参数：训练窗口、标签配置、特征
    schema 版本、历史窗口长度、主板筛选开关、baseline index code。
    ``force_sync`` 不影响样本，故不在指纹内。
    """

    start_date: str
    end_date: str
    schema_version: int
    label_threshold_pct: float
    label_source: str
    daily_lookback: int
    moneyflow_lookback: int
    main_board_only: bool
    baseline_index_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "schema_version": int(self.schema_version),
            "label_threshold_pct": float(self.label_threshold_pct),
            "label_source": self.label_source,
            "daily_lookback": int(self.daily_lookback),
            "moneyflow_lookback": int(self.moneyflow_lookback),
            "main_board_only": bool(self.main_board_only),
            "baseline_index_code": self.baseline_index_code,
        }

    def digest(self) -> str:
        """12-char hex — directory suffix; uses BLAKE2b-128 truncated."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()[:12]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class CheckpointState:
    fingerprint: CheckpointFingerprint
    completed_dates: list[str] = field(default_factory=list)
    plugin_version: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "digest": self.fingerprint.digest(),
            "completed_dates": sorted(set(self.completed_dates)),
            "plugin_version": self.plugin_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointState:
        fp_dict = d.get("fingerprint") or {}
        fp = CheckpointFingerprint(
            start_date=str(fp_dict["start_date"]),
            end_date=str(fp_dict["end_date"]),
            schema_version=int(fp_dict["schema_version"]),
            label_threshold_pct=float(fp_dict["label_threshold_pct"]),
            label_source=str(fp_dict["label_source"]),
            daily_lookback=int(fp_dict["daily_lookback"]),
            moneyflow_lookback=int(fp_dict["moneyflow_lookback"]),
            main_board_only=bool(fp_dict["main_board_only"]),
            baseline_index_code=str(fp_dict["baseline_index_code"]),
        )
        return cls(
            fingerprint=fp,
            completed_dates=[str(x) for x in d.get("completed_dates") or []],
            plugin_version=d.get("plugin_version"),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


class CheckpointMismatch(RuntimeError):
    """On-disk fingerprint disagrees with the one requested by this run."""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def checkpoints_root() -> Path:
    return lgb_paths.checkpoints_dir()


def checkpoint_dir(digest: str) -> Path:
    return checkpoints_root() / digest


def days_dir(digest: str) -> Path:
    return checkpoint_dir(digest) / DAYS_DIRNAME


def state_path(digest: str) -> Path:
    return checkpoint_dir(digest) / STATE_FILENAME


def shard_path(digest: str, anomaly_date: str) -> Path:
    return days_dir(digest) / f"{anomaly_date}{SHARD_SUFFIX}"


def ensure_layout(digest: str) -> None:
    days_dir(digest).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(digest: str) -> CheckpointState | None:
    sp = state_path(digest)
    if not sp.is_file():
        return None
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CheckpointMismatch(
            f"checkpoint state unreadable ({sp}): {e}; rm the dir or --fresh"
        ) from e
    state = CheckpointState.from_dict(data)
    if state.fingerprint.digest() != digest:
        raise CheckpointMismatch(
            f"checkpoint digest mismatch: dir={digest}, "
            f"state.digest={state.fingerprint.digest()}; rerun with --fresh"
        )
    return state


def save_state(state: CheckpointState) -> None:
    digest = state.fingerprint.digest()
    ensure_layout(digest)
    sp = state_path(digest)
    if not state.created_at:
        state.created_at = _now_iso()
    state.updated_at = _now_iso()
    tmp = sp.with_suffix(sp.suffix + ".tmp")
    tmp.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, sp)


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------


def completed_dates(digest: str) -> set[str]:
    """Disk-as-truth: scan ``days/*.parquet`` for the set of completed dates."""
    dd = days_dir(digest)
    if not dd.is_dir():
        return set()
    out: set[str] = set()
    for f in dd.glob(f"*{SHARD_SUFFIX}"):
        name = f.stem
        if len(name) == 8 and name.isdigit():
            out.add(name)
    return out


def save_day_shard(
    digest: str, anomaly_date: str, shard_df: pd.DataFrame
) -> None:
    """Persist one anomaly_date's shard. Empty frames are allowed."""
    ensure_layout(digest)
    missing = [c for c in SHARD_COLUMNS if c not in shard_df.columns]
    if missing:
        raise ValueError(
            f"shard missing required columns: {missing} "
            f"(have {list(shard_df.columns)})"
        )
    df = shard_df[SHARD_COLUMNS].copy()
    target = shard_path(digest, anomaly_date)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)


def load_day_shard(digest: str, anomaly_date: str) -> pd.DataFrame | None:
    sp = shard_path(digest, anomaly_date)
    if not sp.is_file():
        return None
    return pd.read_parquet(sp)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def open_or_create(
    fingerprint: CheckpointFingerprint,
    *,
    plugin_version: str | None = None,
) -> CheckpointState:
    digest = fingerprint.digest()
    existing = load_state(digest)
    if existing is not None:
        if existing.fingerprint != fingerprint:
            raise CheckpointMismatch(
                f"checkpoint fingerprint mismatch under digest={digest}: "
                f"on-disk={existing.fingerprint}, requested={fingerprint}; --fresh"
            )
        return existing
    state = CheckpointState(
        fingerprint=fingerprint,
        completed_dates=[],
        plugin_version=plugin_version,
        created_at=_now_iso(),
        updated_at=_now_iso(),
    )
    save_state(state)
    return state


def record_day_done(digest: str, anomaly_date: str) -> None:
    state = load_state(digest)
    if state is None:
        logger.warning(
            "record_day_done: state.json missing for digest=%s; "
            "relying on disk shards for self-repair",
            digest,
        )
        return
    if anomaly_date in state.completed_dates:
        return
    state.completed_dates = sorted({*state.completed_dates, anomaly_date})
    save_state(state)


def delete_checkpoint(digest: str) -> None:
    cd = checkpoint_dir(digest)
    if cd.is_dir():
        shutil.rmtree(cd, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def assemble_full_dataset(
    digest: str,
    *,
    label_threshold_pct: float,
    label_source: str,
    daily_lookback: int,
    moneyflow_lookback: int,
    anomaly_dates: list[str] | None = None,
):
    """Read every shard → :class:`VaLgbDataset` (imported lazily)."""
    from .dataset import VaLgbDataset  # noqa: PLC0415

    def _empty() -> VaLgbDataset:
        return VaLgbDataset(
            feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
            labels=pd.Series([], dtype="Int64", name="label"),
            sample_index=pd.DataFrame(columns=META_COLUMNS),
            split_groups=pd.Series([], dtype="Int64", name="split_group"),
            schema_version=SCHEMA_VERSION,
            daily_lookback=daily_lookback,
            moneyflow_lookback=moneyflow_lookback,
            label_threshold_pct=label_threshold_pct,
            label_source=label_source,
            anomaly_dates=list(anomaly_dates or []),
        )

    dd = days_dir(digest)
    if not dd.is_dir():
        return _empty()

    shard_files = sorted(dd.glob(f"*{SHARD_SUFFIX}"))
    frames: list[pd.DataFrame] = []
    for f in shard_files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return _empty()

    full = pd.concat(frames, ignore_index=True)
    feature_matrix = full.reindex(columns=FEATURE_NAMES)
    labels = full["label"].astype("Int64").rename("label")
    sample_index = full[META_COLUMNS].reset_index(drop=True)
    split_groups = (
        sample_index["anomaly_date"]
        .astype(int)
        .astype("Int64")
        .rename("split_group")
    )
    return VaLgbDataset(
        feature_matrix=feature_matrix.reset_index(drop=True),
        labels=labels.reset_index(drop=True),
        sample_index=sample_index,
        split_groups=split_groups.reset_index(drop=True),
        schema_version=SCHEMA_VERSION,
        daily_lookback=daily_lookback,
        moneyflow_lookback=moneyflow_lookback,
        label_threshold_pct=label_threshold_pct,
        label_source=label_source,
        anomaly_dates=list(anomaly_dates or []),
    )


# ---------------------------------------------------------------------------
# Shard <-> DayBundle adapters (consumed by dataset.collect_training_window)
# ---------------------------------------------------------------------------


def day_bundle_to_shard(
    *,
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    sample_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Pack the day's three tables into a shard DataFrame keyed on
    SHARD_COLUMNS. Empty input → an empty frame with the right columns."""
    if len(feature_matrix) == 0:
        return pd.DataFrame(columns=SHARD_COLUMNS)
    df = feature_matrix.reset_index(drop=True).copy()
    df["label"] = labels.reset_index(drop=True).astype("Int64").to_numpy()
    meta = sample_meta.reset_index(drop=True)
    for col in META_COLUMNS:
        if col not in meta.columns:
            raise ValueError(f"sample_meta missing column: {col}")
        df[col] = meta[col].to_numpy()
    return df[SHARD_COLUMNS]
