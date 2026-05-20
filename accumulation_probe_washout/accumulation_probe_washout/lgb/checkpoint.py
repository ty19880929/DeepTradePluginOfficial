"""Per-day Phase-1 collection shards (resumable training).

A "checkpoint" is a directory under ``~/.deeptrade/accumulation_probe_washout/
checkpoints/<digest>/`` where ``<digest>`` is a BLAKE2b-64 hash of the
training parameters that affect the **data side** of the pipeline (window,
label config, schema version, lookbacks, baseline index, adj_factor flag).

Layout::

    <root>/checkpoints/<digest>/
    ├── meta.json
    └── days/
        ├── 20240101.parquet
        ├── 20240102.parquet
        └── ...

If a ``lgb train`` run crashes or is Ctrl-C'd halfway through Phase-1, the
next invocation with the same parameters re-uses any already-written shards
and only refills the missing days. The trainer deletes the directory on
success unless ``--keep-checkpoint`` is passed.

Note: the **same** digest is also embedded in the LightGBM hyperparameter
dict (via the trainer's ``hyperparams_json``) so a model record can be
matched back to the data slice that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import checkpoints_dir


META_COLUMNS = ["ts_code", "signal_date", "label", "data_status"]


@dataclass(frozen=True)
class CheckpointFingerprint:
    """Stable identifier for the data-side of a training run."""

    start_date: str
    end_date: str
    label_source: str
    label_threshold_pct: float
    label_drawdown_threshold_pct: float
    schema_version: int
    baseline_index_code: str
    volume_adjust_enabled: bool
    # Lookback parameters carry the same semantics as the v0.3 backfill loop;
    # encoded here so changing them invalidates old shards automatically.
    base_lookback_trade_days: int
    probe_lookback_trade_days: int
    accumulation_lookback_trade_days: int

    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


@dataclass
class DayShard:
    """One per-trade-date snapshot of (features, label, meta)."""

    signal_date: str
    feature_matrix: pd.DataFrame  # columns = FEATURE_NAMES, index = ts_code
    sample_meta: pd.DataFrame     # META_COLUMNS schema


@dataclass
class CheckpointWriter:
    """Filesystem handle for one fingerprint."""

    fingerprint: CheckpointFingerprint
    _root: Path = field(init=False)

    def __post_init__(self) -> None:
        self._root = checkpoints_dir() / self.fingerprint.digest()
        (self._root / "days").mkdir(parents=True, exist_ok=True)
        self._write_meta()

    def _write_meta(self) -> None:
        meta_path = self._root / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps(asdict(self.fingerprint), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def days_dir(self) -> Path:
        return self._root / "days"

    def existing_dates(self) -> set[str]:
        return {p.stem for p in self.days_dir.glob("*.parquet")}

    def write_day(self, shard: DayShard) -> None:
        out = self.days_dir / f"{shard.signal_date}.parquet"
        # Join meta + features inline so a single parquet file is the unit of
        # resume. Reader unpacks them back.
        meta = shard.sample_meta.copy()
        feat = shard.feature_matrix.copy()
        feat.index.name = "ts_code"
        feat = feat.reset_index()
        # Avoid double ts_code column on the join.
        meta = meta.drop(columns=[c for c in meta.columns if c == "ts_code"
                                   and c in feat.columns and len(meta) == len(feat)])
        joined = pd.concat([feat, meta.reset_index(drop=True)], axis=1)
        joined.to_parquet(out, index=False)

    def read_all(self) -> list[DayShard]:
        shards: list[DayShard] = []
        for p in sorted(self.days_dir.glob("*.parquet")):
            df = pd.read_parquet(p)
            # ``META_COLUMNS`` already contains ``ts_code``; build the slice
            # without duplicating that column when it lives in df.
            meta_cols = [c for c in META_COLUMNS if c in df.columns]
            feat_cols = [
                c for c in df.columns if c not in meta_cols and c != "ts_code"
            ]
            if "ts_code" in df.columns:
                feat = df.set_index("ts_code")[feat_cols]
                meta = df[meta_cols]
                if "ts_code" not in meta.columns:
                    meta = meta.assign(ts_code=df["ts_code"].values)
            else:
                feat = df[feat_cols]
                meta = df[meta_cols]
            shards.append(
                DayShard(
                    signal_date=p.stem,
                    feature_matrix=feat,
                    sample_meta=meta,
                )
            )
        return shards

    def discard(self) -> None:
        """Recursively delete the checkpoint dir (used on training success)."""
        if not self._root.exists():
            return
        for p in sorted(self._root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        self._root.rmdir()


def open_checkpoint(fp: CheckpointFingerprint) -> CheckpointWriter:
    return CheckpointWriter(fingerprint=fp)


def existing_checkpoint_digests() -> list[str]:
    root = checkpoints_dir()
    return sorted(p.name for p in root.iterdir() if p.is_dir())
