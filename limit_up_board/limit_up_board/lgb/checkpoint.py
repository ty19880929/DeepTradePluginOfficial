"""Phase-1 训练数据收集 checkpoint —— 支撑 ``lgb train`` 长窗口续训。

设计动机
--------
``lgb train`` 最耗时的环节是按日抓取 + 特征化（1 年 ~240 个交易日 × ~12 个
Tushare 接口 + 每日 ``build_feature_frame``）。一旦中途崩溃（网络、Ctrl-C、
机器掉电），既有流程必须从头跑。LightGBM 拟合本身是 ``deterministic=True
+ seed=42`` 的、相对短的 CPU 任务，不在续训范围内——再跑一次代价小。

本模块把每日处理结果按 trade_date 落成 parquet shard，附带一份 state.json
记录"指纹"（训练窗口 + 过滤参数 + schema_version + lookbacks）和已完成日
期列表；下次同配置启动时跳过磁盘上已有的日，仅补漏。训练成功后整个
checkpoint 目录被清理。

存储布局
--------
``~/.deeptrade/limit_up_board/checkpoints/<digest>/``::

    days/<YYYYMMDD>.parquet     # 单日 shard：FEATURE_NAMES + label + meta
    state.json                  # 指纹 + completed_dates + 版本元信息

shard 列顺序固定为 :data:`SHARD_COLUMNS`（``FEATURE_NAMES + ["label",
"ts_code", "trade_date", "next_trade_date", "pct_chg_t1"]``）—— 这是 the
schema 单一来源；``assemble_full_dataset`` 直接按此重建 :class:`LgbDataset`。

并发与一致性
------------
* state.json 用 ``os.replace`` 原子写入（Windows / POSIX 均原子）。
* 每天结束时 **先落 shard 再更新 state**：即使 state.json 比 shard 旧，
  下次启动通过 ``completed_dates``（扫 ``days/*.parquet`` 而非读 state）
  自我修复——磁盘是 truth source，state 仅是加速索引。
* 不支持同一 fingerprint 并发 ``lgb train``：调用方自行加锁/串行。
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
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import paths as lgb_paths
from .features import FEATURE_NAMES, SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover
    from .dataset import LgbDataset

logger = logging.getLogger(__name__)


STATE_FILENAME = "state.json"
DAYS_DIRNAME = "days"
SHARD_SUFFIX = ".parquet"

# Shard parquet 的列顺序（与 lgb_dataset_*.parquet 一致）
META_COLUMNS: list[str] = [
    "ts_code",
    "trade_date",
    "next_trade_date",
    "pct_chg_t1",
]
SHARD_COLUMNS: list[str] = FEATURE_NAMES + ["label"] + META_COLUMNS


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointFingerprint:
    """训练参数稳定指纹。

    成员仅包含会**改变 dataset 内容**的参数：训练窗口、市场筛选阈值、标签阈值、
    历史窗口长度、特征 schema 版本。``force_sync`` 不影响样本，故不在指纹内。
    """

    start_date: str
    end_date: str
    schema_version: int
    label_threshold_pct: float
    daily_lookback: int
    moneyflow_lookback: int
    min_float_mv_yi: float
    max_float_mv_yi: float
    max_close_yuan: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "schema_version": self.schema_version,
            "label_threshold_pct": float(self.label_threshold_pct),
            "daily_lookback": int(self.daily_lookback),
            "moneyflow_lookback": int(self.moneyflow_lookback),
            "min_float_mv_yi": float(self.min_float_mv_yi),
            "max_float_mv_yi": float(self.max_float_mv_yi),
            "max_close_yuan": float(self.max_close_yuan),
        }

    def digest(self) -> str:
        """12 位 hex，作为 checkpoint 目录名后缀的稳定 hash。"""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


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
            daily_lookback=int(fp_dict["daily_lookback"]),
            moneyflow_lookback=int(fp_dict["moneyflow_lookback"]),
            min_float_mv_yi=float(fp_dict["min_float_mv_yi"]),
            max_float_mv_yi=float(fp_dict["max_float_mv_yi"]),
            max_close_yuan=float(fp_dict["max_close_yuan"]),
        )
        return cls(
            fingerprint=fp,
            completed_dates=[str(x) for x in d.get("completed_dates") or []],
            plugin_version=d.get("plugin_version"),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )


class CheckpointMismatch(RuntimeError):
    """磁盘 state.json 的 fingerprint 与本次调用不一致。"""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def checkpoints_root() -> Path:
    """``~/.deeptrade/limit_up_board/checkpoints/`` —— 所有 checkpoint 的父目录。"""
    return lgb_paths.plugin_data_dir() / "checkpoints"


def checkpoint_dir(digest: str) -> Path:
    return checkpoints_root() / digest


def days_dir(digest: str) -> Path:
    return checkpoint_dir(digest) / DAYS_DIRNAME


def state_path(digest: str) -> Path:
    return checkpoint_dir(digest) / STATE_FILENAME


def shard_path(digest: str, trade_date: str) -> Path:
    return days_dir(digest) / f"{trade_date}{SHARD_SUFFIX}"


def ensure_layout(digest: str) -> None:
    days_dir(digest).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(digest: str) -> CheckpointState | None:
    """读 state.json；不存在返回 None；解析失败抛 ``CheckpointMismatch``。"""
    sp = state_path(digest)
    if not sp.is_file():
        return None
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CheckpointMismatch(
            f"checkpoint state file unreadable ({sp}): {e}; "
            "删除该 checkpoint 目录或 --fresh 重试"
        ) from e
    state = CheckpointState.from_dict(data)
    if state.fingerprint.digest() != digest:
        # 目录名 ≠ state 内部 digest：可能是磁盘损坏 / 手工拷贝
        raise CheckpointMismatch(
            f"checkpoint digest mismatch: dir={digest}, "
            f"state.digest={state.fingerprint.digest()}; 请 --fresh"
        )
    return state


def save_state(state: CheckpointState) -> None:
    """原子写 state.json（写 .tmp → os.replace）。"""
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
    """以磁盘 ``days/*.parquet`` 为 truth source 返回已完成日期集合。"""
    dd = days_dir(digest)
    if not dd.is_dir():
        return set()
    out: set[str] = set()
    for f in dd.glob(f"*{SHARD_SUFFIX}"):
        name = f.stem  # YYYYMMDD
        if len(name) == 8 and name.isdigit():
            out.add(name)
    return out


def save_day_shard(digest: str, trade_date: str, shard_df: pd.DataFrame) -> None:
    """落盘单日 parquet shard。

    Parameters
    ----------
    shard_df
        必须含 :data:`SHARD_COLUMNS` 所有列；多余列被丢弃。空 frame（当天 0
        样本）也允许，会落一个 0 行 parquet——下次启动通过 ``trade_date in
        completed_dates(...)`` 判定为已处理。
    """
    ensure_layout(digest)
    # 保留约定列序，过滤掉额外列
    missing = [c for c in SHARD_COLUMNS if c not in shard_df.columns]
    if missing:
        raise ValueError(
            f"shard missing required columns: {missing} (have {list(shard_df.columns)})"
        )
    df = shard_df[SHARD_COLUMNS].copy()
    target = shard_path(digest, trade_date)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)


def load_day_shard(digest: str, trade_date: str) -> pd.DataFrame | None:
    """读单日 shard；缺失返回 None。"""
    sp = shard_path(digest, trade_date)
    if not sp.is_file():
        return None
    return pd.read_parquet(sp)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def assemble_full_dataset(
    digest: str,
    *,
    label_threshold_pct: float,
    daily_lookback: int,
    moneyflow_lookback: int,
    trade_dates: list[str] | None = None,
) -> "LgbDataset":
    """读所有 shard，拼成可训练的 :class:`LgbDataset`。

    Notes
    -----
    - 列顺序按 :data:`FEATURE_NAMES` 锁死。
    - ``trade_dates`` 仅作为元信息塞进 ``LgbDataset.trade_dates``——不依赖它
      决定加载哪些 shard（磁盘是 truth source）。
    """
    # 局部 import 避免 paths.py / cleanup.py 引用本模块时的循环
    from .dataset import LgbDataset  # noqa: PLC0415

    def _empty() -> LgbDataset:
        return LgbDataset(
            feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
            labels=pd.Series([], dtype="Int64", name="label"),
            sample_index=pd.DataFrame(columns=META_COLUMNS),
            split_groups=pd.Series([], dtype="Int64", name="split_group"),
            schema_version=SCHEMA_VERSION,
            daily_lookback=daily_lookback,
            moneyflow_lookback=moneyflow_lookback,
            label_threshold_pct=label_threshold_pct,
            trade_dates=list(trade_dates or []),
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
        sample_index["trade_date"].astype(int).astype("Int64").rename("split_group")
    )
    return LgbDataset(
        feature_matrix=feature_matrix.reset_index(drop=True),
        labels=labels.reset_index(drop=True),
        sample_index=sample_index,
        split_groups=split_groups.reset_index(drop=True),
        schema_version=SCHEMA_VERSION,
        daily_lookback=daily_lookback,
        moneyflow_lookback=moneyflow_lookback,
        label_threshold_pct=label_threshold_pct,
        trade_dates=list(trade_dates or []),
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def open_or_create(
    fingerprint: CheckpointFingerprint,
    *,
    plugin_version: str | None = None,
) -> CheckpointState:
    """读已有 state；若 digest 不匹配或不存在则新建。

    抛 :class:`CheckpointMismatch` 当且仅当：磁盘上同 digest 目录的 state.json
    存在但损坏 / 内部 fingerprint 与目录名不符——CLI 应建议 ``--fresh``。
    """
    digest = fingerprint.digest()
    existing = load_state(digest)
    if existing is not None:
        # 严格相等才视作"接着跑"
        if existing.fingerprint != fingerprint:
            raise CheckpointMismatch(
                f"checkpoint fingerprint mismatch under digest={digest}: "
                f"on-disk={existing.fingerprint}, requested={fingerprint}; "
                "请 --fresh"
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


def record_day_done(digest: str, trade_date: str) -> None:
    """把 trade_date 写进 state.completed_dates 并刷盘。

    幂等：重复调用同日不会重复写入。state 缺失时静默放弃（shard 仍在磁盘上，
    下次启动会从 ``completed_dates(digest)`` 重建）。
    """
    state = load_state(digest)
    if state is None:
        logger.warning(
            "record_day_done called but state.json missing for digest=%s; "
            "依赖磁盘 shard 自我修复",
            digest,
        )
        return
    if trade_date in state.completed_dates:
        return
    state.completed_dates = sorted({*state.completed_dates, trade_date})
    save_state(state)


def delete_checkpoint(digest: str) -> None:
    """递归删除单个 checkpoint 目录。不存在则静默。"""
    cd = checkpoint_dir(digest)
    if cd.is_dir():
        shutil.rmtree(cd, ignore_errors=True)


def count_checkpoints() -> tuple[int, int]:
    """``(n_checkpoints, n_total_shards)``。供 ``cleanup.count_artifacts`` 调用。"""
    root = checkpoints_root()
    if not root.is_dir():
        return 0, 0
    n_ck = 0
    n_shards = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        n_ck += 1
        dd = entry / DAYS_DIRNAME
        if dd.is_dir():
            n_shards += sum(1 for _ in dd.glob(f"*{SHARD_SUFFIX}"))
    return n_ck, n_shards


def purge_all_checkpoints() -> tuple[int, list[str]]:
    """删除所有 checkpoint 目录，返回 ``(n_removed, errors)``。"""
    root = checkpoints_root()
    if not root.is_dir():
        return 0, []
    n = 0
    errs: list[str] = []
    for entry in list(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            shutil.rmtree(entry)
            n += 1
        except OSError as e:
            errs.append(f"rmtree {entry.name}: {e}")
    return n, errs


# ---------------------------------------------------------------------------
# Shard <-> DayBundle adapters (consumed by dataset.collect_training_window)
# ---------------------------------------------------------------------------


def day_bundle_to_shard(
    *,
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    sample_meta: pd.DataFrame,
) -> pd.DataFrame:
    """把 :class:`dataset._DayBundle` 三段拼成单日 shard DataFrame。

    分离在本模块（而非 dataset.py）是因为 shard 的列顺序契约属于
    checkpoint 子系统；dataset.py 只需 import 这个 helper 即可，免去
    重复声明。
    """
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
