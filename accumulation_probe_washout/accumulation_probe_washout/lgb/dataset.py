"""Training-matrix assembly from ``apw_signal_history`` + ``apw_realized_returns``.

APW's key advantage over the legacy VA pipeline is that the screen path
**already** persists a fully-featured candidate dict in
``apw_signal_history.raw_candidate_json`` (PR-1 backfill writes it; PR-2
extended it with VCP / alpha / long-range / MA / volume_event_score).
Training therefore requires **zero** Tushare calls — just JSON-parse the
existing rows and feed them to :func:`build_feature_frame`.

Public entrypoint :func:`collect_training_window` returns an
:class:`ApwLgbDataset` ready for :mod:`trainer`. Per-day Phase-1 progress
lands in the checkpoint dir so a crashed train resumes cleanly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from .checkpoint import (
    META_COLUMNS,
    CheckpointFingerprint,
    CheckpointWriter,
    DayShard,
    open_checkpoint,
)
from .features import FEATURE_NAMES, SCHEMA_VERSION, build_feature_frame
from .labels import fetch_labels_for_window

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from ..config import ApwConfig

logger = logging.getLogger(__name__)


@dataclass
class ApwLgbDataset:
    """Training matrix + label + group + sample metadata."""

    feature_matrix: pd.DataFrame   # rows × FEATURE_NAMES, RangeIndex
    labels: pd.Series              # int8, parallel to feature_matrix
    sample_index: pd.DataFrame     # ts_code / signal_date / data_status / row index
    split_groups: pd.Series        # signal_date strings — GroupKFold key
    schema_version: int = SCHEMA_VERSION
    label_source: str = "label_launch_t5"
    label_threshold_pct: float | None = None
    label_drawdown_threshold_pct: float | None = None
    signal_dates: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(len(self.feature_matrix))

    @property
    def n_positive(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def n_labeled(self) -> int:
        return int(self.labels.notna().sum())


# ---------------------------------------------------------------------------
# Per-day shard build (pure read from apw_signal_history)
# ---------------------------------------------------------------------------


def enumerate_signal_dates(
    db: Database, *, start_date: str, end_date: str
) -> list[str]:
    rows = db.fetchall(
        "SELECT DISTINCT trade_date FROM apw_signal_history "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date),
    )
    return [str(r[0]) for r in (rows or [])]


def _candidates_for_signal_date(db: Database, signal_date: str) -> list[dict[str, Any]]:
    rows = db.fetchall(
        "SELECT raw_candidate_json FROM apw_signal_history WHERE trade_date = ?",
        (signal_date,),
    ) or []
    out: list[dict[str, Any]] = []
    for (raw,) in rows:
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "apw_signal_history(%s) row carries unparseable JSON; skipped",
                signal_date,
            )
            continue
        if isinstance(data, dict) and data.get("ts_code"):
            out.append(data)
    return out


def build_day_shard(db: Database, *, signal_date: str) -> DayShard | None:
    """Build one DayShard from ``apw_signal_history`` rows.

    Returns ``None`` when the day has no rows (caller skips). Label is NOT
    populated here — :func:`collect_training_window` joins labels in a single
    bulk pass per window so the per-day path stays I/O-cheap.
    """
    cands = _candidates_for_signal_date(db, signal_date)
    if not cands:
        return None
    feat = build_feature_frame(candidate_rows=cands)
    if feat.empty:
        return None
    meta = pd.DataFrame(
        {
            "ts_code": list(feat.index),
            "signal_date": [signal_date] * len(feat),
            "label": [pd.NA] * len(feat),
            "data_status": [None] * len(feat),
        },
        columns=META_COLUMNS,
    )
    return DayShard(signal_date=signal_date, feature_matrix=feat, sample_meta=meta)


# ---------------------------------------------------------------------------
# Window orchestration (with checkpoint resume)
# ---------------------------------------------------------------------------


def make_fingerprint(
    *,
    start_date: str,
    end_date: str,
    label_source: str,
    label_threshold_pct: float,
    label_drawdown_threshold_pct: float,
    cfg: ApwConfig,
) -> CheckpointFingerprint:
    return CheckpointFingerprint(
        start_date=start_date,
        end_date=end_date,
        label_source=label_source,
        label_threshold_pct=label_threshold_pct,
        label_drawdown_threshold_pct=label_drawdown_threshold_pct,
        schema_version=SCHEMA_VERSION,
        baseline_index_code=cfg.baseline_index_code,
        volume_adjust_enabled=bool(cfg.volume_adjust_enabled),
        base_lookback_trade_days=cfg.base_lookback_trade_days,
        probe_lookback_trade_days=cfg.probe_lookback_trade_days,
        accumulation_lookback_trade_days=cfg.accumulation_lookback_trade_days,
    )


def collect_training_window(
    db: Database,
    *,
    start_date: str,
    end_date: str,
    cfg: ApwConfig,
    label_source: str | None = None,
    label_threshold_pct: float | None = None,
    label_drawdown_threshold_pct: float | None = None,
    fresh: bool = False,
    on_progress: Any = None,
) -> tuple[ApwLgbDataset, CheckpointWriter]:
    """Build the full training matrix for ``[start_date, end_date]``.

    Returns ``(dataset, checkpoint_writer)`` so the caller (trainer) can
    decide whether to drop the checkpoint dir on success.

    ``on_progress`` is an optional ``Callable[[str, int, int], None]``
    invoked once per signal_date with ``(date, idx, total)`` — wired up to
    the runner's event stream.
    """
    src = label_source or cfg.lgb_label_source
    th = (
        label_threshold_pct
        if label_threshold_pct is not None
        else cfg.lgb_label_threshold_pct
    )
    dd = (
        label_drawdown_threshold_pct
        if label_drawdown_threshold_pct is not None
        else cfg.lgb_label_drawdown_threshold_pct
    )

    fp = make_fingerprint(
        start_date=start_date,
        end_date=end_date,
        label_source=src,
        label_threshold_pct=th,
        label_drawdown_threshold_pct=dd,
        cfg=cfg,
    )
    cp = open_checkpoint(fp)
    if fresh:
        cp.discard()
        cp = open_checkpoint(fp)

    # Phase 1: enumerate dates, fill missing shards.
    all_dates = enumerate_signal_dates(db, start_date=start_date, end_date=end_date)
    already = cp.existing_dates()
    for i, d in enumerate(all_dates, start=1):
        if d in already:
            continue
        shard = build_day_shard(db, signal_date=d)
        if shard is None:
            continue
        cp.write_day(shard)
        if on_progress is not None:
            on_progress(d, i, len(all_dates))

    # Phase 2: load all shards from disk; concat.
    shards = cp.read_all()
    if not shards:
        return (
            ApwLgbDataset(
                feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
                labels=pd.Series([], dtype="Int8", name="label"),
                sample_index=pd.DataFrame(columns=META_COLUMNS),
                split_groups=pd.Series([], dtype=object, name="signal_date"),
                label_source=src,
                label_threshold_pct=th if src == "custom_t5" else None,
                label_drawdown_threshold_pct=dd if src == "custom_t5" else None,
            ),
            cp,
        )

    feature_frames: list[pd.DataFrame] = []
    meta_frames: list[pd.DataFrame] = []
    for sh in shards:
        feat = sh.feature_matrix
        # Re-order to canonical column order so concat is safe across shards.
        feat = feat.reindex(columns=FEATURE_NAMES).reset_index(drop=False)
        feature_frames.append(feat)
        meta = sh.sample_meta.copy()
        if "signal_date" not in meta.columns:
            meta["signal_date"] = sh.signal_date
        meta_frames.append(meta)

    big_feat = pd.concat(feature_frames, ignore_index=True)
    big_meta = pd.concat(meta_frames, ignore_index=True)
    # Pull ts_code out of feature frame into the meta frame; drop from feat.
    if "ts_code" in big_feat.columns:
        if "ts_code" not in big_meta.columns or big_meta["ts_code"].isna().all():
            big_meta["ts_code"] = big_feat["ts_code"].values
        big_feat = big_feat.drop(columns=["ts_code"])

    # Phase 3: bulk label JOIN over the same window.
    label_df = fetch_labels_for_window(
        db,
        start_date=start_date,
        end_date=end_date,
        source=src,
        threshold_pct=th if src == "custom_t5" else None,
        drawdown_threshold_pct=dd if src == "custom_t5" else None,
    )
    # Align big_meta with label_df on (signal_date, ts_code).
    merged = big_meta.merge(
        label_df, how="left", on=["signal_date", "ts_code"], suffixes=("", "_lbl")
    )
    # Prefer the freshly-joined label over any stale meta label.
    final_label = merged["label_lbl"] if "label_lbl" in merged.columns else merged["label"]
    big_meta["label"] = final_label.astype("Int8")
    big_meta = big_meta[META_COLUMNS]

    dataset = ApwLgbDataset(
        feature_matrix=big_feat.reset_index(drop=True),
        labels=big_meta["label"].astype("Int8").reset_index(drop=True),
        sample_index=big_meta.reset_index(drop=True),
        split_groups=big_meta["signal_date"].astype(str).reset_index(drop=True),
        label_source=src,
        label_threshold_pct=th if src == "custom_t5" else None,
        label_drawdown_threshold_pct=dd if src == "custom_t5" else None,
        signal_dates=sorted(big_meta["signal_date"].astype(str).unique().tolist()),
    )
    return dataset, cp
