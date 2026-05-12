"""训练数据收集器 — 把多个 anomaly_date 的特征 + 标签拼成可训练矩阵。

设计文档 §6.1：本模块复用 :mod:`volume_anomaly.data` 的数据装配 helper +
:func:`features.build_feature_frame` + :func:`labels.fetch_labels_for_date`，
**不进入 LLM 阶段**。

主入口：:func:`collect_training_window`。返回 :class:`VaLgbDataset`，被
:func:`trainer.train_lightgbm` 直接消费（trainer 在 PR-1.3 实现）。

历史回放策略
-----------
对每个 anomaly_date ``T``：

1. 读 ``va_anomaly_history WHERE trade_date = T`` 取得当日异动列表
   （ts_code / name / industry / 异动当日量价指标）。
2. 拉 ``T`` 之前 ``daily_lookback`` 个交易日的 daily 行情（含 ``T`` 当天）。
3. 拉 ``T`` 之前 ``moneyflow_lookback`` 天的 moneyflow。
4. 拉沪深 300 (``baseline_index_code``) 同窗口 index_daily。
5. 拉 60 日窗口的 ``limit_list_d``（标记历史涨停）。
6. 对每只 (ts_code, T) 调 ``build_candidate_features``（与 analyze 路径同源）。
7. 调 ``build_feature_frame`` 转为特征矩阵。
8. JOIN ``labels.fetch_labels_for_date(db, T, source, threshold)``。
9. 写 checkpoint shard。

Phase-1 续训：相同 (fingerprint) 下重启时跳过 ``completed_dates(digest)``
中的 anomaly_date，仅补漏；详见 :mod:`.checkpoint`。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import checkpoint as ckpt
from .features import FEATURE_NAMES, SCHEMA_VERSION, build_feature_frame
from .labels import fetch_labels_for_date

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient

    from ..calendar import TradeCalendar


logger = logging.getLogger(__name__)


DEFAULT_DAILY_LOOKBACK = 250
DEFAULT_MONEYFLOW_LOOKBACK = 5
DEFAULT_BASELINE_INDEX_CODE = "000300.SH"


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class VaLgbDataset:
    """训练矩阵 + 标签 + 样本元信息。"""

    feature_matrix: pd.DataFrame
    labels: pd.Series
    sample_index: pd.DataFrame
    split_groups: pd.Series
    schema_version: int = SCHEMA_VERSION
    daily_lookback: int = DEFAULT_DAILY_LOOKBACK
    moneyflow_lookback: int = DEFAULT_MONEYFLOW_LOOKBACK
    label_threshold_pct: float = 5.0
    label_source: str = "max_ret_5d"
    anomaly_dates: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(len(self.feature_matrix))

    @property
    def n_positive(self) -> int:
        return int(self.labels.fillna(-1).eq(1).sum())

    @property
    def n_labeled(self) -> int:
        return int(self.labels.notna().sum())

    def filter_labeled(self) -> "VaLgbDataset":
        mask = self.labels.notna().to_numpy()
        return VaLgbDataset(
            feature_matrix=self.feature_matrix.loc[mask].reset_index(drop=True),
            labels=self.labels.loc[mask].reset_index(drop=True).astype("Int64"),
            sample_index=self.sample_index.loc[mask].reset_index(drop=True),
            split_groups=self.split_groups.loc[mask].reset_index(drop=True),
            schema_version=self.schema_version,
            daily_lookback=self.daily_lookback,
            moneyflow_lookback=self.moneyflow_lookback,
            label_threshold_pct=self.label_threshold_pct,
            label_source=self.label_source,
            anomaly_dates=list(self.anomaly_dates),
        )


# ---------------------------------------------------------------------------
# Anomaly-date enumeration
# ---------------------------------------------------------------------------


def enumerate_anomaly_dates(
    db: Database,
    *,
    start_date: str,
    end_date: str,
) -> list[str]:
    """Distinct anomaly trade_dates in ``va_anomaly_history`` within window."""
    rows = db.fetchall(
        "SELECT DISTINCT trade_date FROM va_anomaly_history "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date),
    )
    return [str(r[0]) for r in rows]


def _fetch_anomaly_rows(
    db: Database, anomaly_date: str
) -> list[dict[str, Any]]:
    """Read all anomaly hits for one date, normalized to dicts."""
    rows = db.fetchall(
        "SELECT trade_date, ts_code, name, industry, pct_chg, close, open, high, "
        "low, vol, amount, body_ratio, turnover_rate, vol_ratio_5d, max_vol_60d "
        "FROM va_anomaly_history WHERE trade_date = ?",
        (anomaly_date,),
    )
    cols = [
        "trade_date",
        "ts_code",
        "name",
        "industry",
        "pct_chg",
        "close",
        "open",
        "high",
        "low",
        "vol",
        "amount",
        "body_ratio",
        "turnover_rate",
        "vol_ratio_5d",
        "max_vol_60d",
    ]
    return [dict(zip(cols, r, strict=False)) for r in rows]


# ---------------------------------------------------------------------------
# Per-day feature assembly (the historical-replay core)
# ---------------------------------------------------------------------------


@dataclass
class _DayBundle:
    """Single anomaly_date's features + labels + meta."""

    feature_matrix: pd.DataFrame  # index = ts_code, columns = FEATURE_NAMES
    labels: pd.Series             # Int64, index = ts_code, name = 'label'
    sample_meta: pd.DataFrame     # ts_code / anomaly_date / max_ret_5d / data_status


def _empty_day() -> _DayBundle:
    return _DayBundle(
        feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
        labels=pd.Series([], dtype="Int64", name="label"),
        sample_meta=pd.DataFrame(columns=ckpt.META_COLUMNS),
    )


def collect_day_samples(
    *,
    tushare: TushareClient,
    db: Database,
    calendar: TradeCalendar,
    anomaly_date: str,
    label_source: str = "max_ret_5d",
    label_threshold_pct: float = 5.0,
    daily_lookback: int = DEFAULT_DAILY_LOOKBACK,
    moneyflow_lookback: int = DEFAULT_MONEYFLOW_LOOKBACK,
    baseline_index_code: str = DEFAULT_BASELINE_INDEX_CODE,
    force_sync: bool = False,
) -> _DayBundle:
    """Replay ``anomaly_date`` worth of features + labels into a _DayBundle.

    No Tushare token consumption beyond what the underlying client caches —
    repeated calls on the same trade_date hit the immutable cache.
    """
    # Local import keeps the heavy data.py module out of import-time cost when
    # the LGB subpackage is loaded for read-only CLI commands like `lgb list`.
    from ..data import (  # noqa: PLC0415
        _fetch_daily_history_by_date,
        _index_daily_by_code,
        _index_moneyflow_by_code,
        _last_n_trade_dates,
        _normalize_id_cols,
        _shift_calendar_days,
        _try_optional,
        build_candidate_features,
    )

    anomaly_rows = _fetch_anomaly_rows(db, anomaly_date)
    if not anomaly_rows:
        return _empty_day()

    candidate_codes = {r["ts_code"] for r in anomaly_rows}

    # 1. Historical daily window (250d ending on anomaly_date).
    history_dates = _last_n_trade_dates(calendar, anomaly_date, daily_lookback)
    daily_df, _ = _fetch_daily_history_by_date(
        tushare, history_dates, candidate_codes, force_sync=force_sync
    )

    # 2. Baseline index daily.
    baseline_close_by_date: dict[str, float] = {}
    if history_dates:
        idx_df, idx_err = _try_optional(
            tushare,
            "index_daily",
            params={
                "ts_code": baseline_index_code,
                "start_date": history_dates[0],
                "end_date": history_dates[-1],
            },
            force_sync=force_sync,
        )
        if not idx_err:
            idx_df = _normalize_id_cols(idx_df)
            if idx_df is not None and not idx_df.empty and "close" in idx_df.columns:
                for r in idx_df[["trade_date", "close"]].itertuples(index=False):
                    if r.close is not None:
                        baseline_close_by_date[str(r.trade_date)] = float(r.close)

    # 3. daily_basic on T.
    db_basic_t = tushare.call(
        "daily_basic", trade_date=anomaly_date, force_sync=force_sync
    )
    db_basic_lookup: dict[str, dict[str, Any]] = {}
    if db_basic_t is not None and not db_basic_t.empty:
        for r in db_basic_t.itertuples(index=False):
            db_basic_lookup[str(r.ts_code)] = {
                "turnover_rate": getattr(r, "turnover_rate", None),
                "volume_ratio": getattr(r, "volume_ratio", None),
                "pe": getattr(r, "pe", None),
                "pb": getattr(r, "pb", None),
                "circ_mv": getattr(r, "circ_mv", None),
                "total_mv": getattr(r, "total_mv", None),
            }

    # 4. Moneyflow window (T-lookback .. T).
    mf_start = _shift_calendar_days(anomaly_date, -(moneyflow_lookback + 7))
    mf_df, _ = _try_optional(
        tushare,
        "moneyflow",
        params={"start_date": mf_start, "end_date": anomaly_date},
        force_sync=force_sync,
    )
    mf_df = _normalize_id_cols(mf_df)
    if mf_df is not None and not mf_df.empty:
        mf_df = mf_df[mf_df["ts_code"].isin(candidate_codes)]

    # 5. limit_list_d 60d window.
    lu_start = history_dates[0] if history_dates else anomaly_date
    lu_df, _ = _try_optional(
        tushare,
        "limit_list_d",
        params={
            "start_date": lu_start,
            "end_date": anomaly_date,
            "limit_type": "U",
        },
        force_sync=force_sync,
    )
    lu_by_code: dict[str, list[str]] = {}
    if lu_df is not None and not lu_df.empty:
        for r in lu_df.itertuples(index=False):
            lu_by_code.setdefault(str(r.ts_code), []).append(str(r.trade_date))

    daily_by_code = _index_daily_by_code(daily_df)
    mf_by_code = _index_moneyflow_by_code(mf_df)

    # 6. Per-stock candidate feature dict (pure historical replay).
    candidate_rows: list[dict[str, Any]] = []
    for hit in anomaly_rows:
        code = hit["ts_code"]
        history = daily_by_code.get(code, [])
        if not history:
            continue
        rec = build_candidate_features(
            ts_code=code,
            trade_date=anomaly_date,
            history=history,
            daily_basic=db_basic_lookup.get(code, {}),
            moneyflow_5d=mf_by_code.get(code, [])[-moneyflow_lookback:],
            limit_up_dates=sorted(lu_by_code.get(code, [])),
            name=hit.get("name"),
            industry=hit.get("industry"),
            tracked_since=anomaly_date,  # historical samples → tracked_days = 0
            last_screened=anomaly_date,
            anomaly_pct_chg=hit.get("pct_chg"),
            anomaly_body_ratio=hit.get("body_ratio"),
            anomaly_turnover_rate=hit.get("turnover_rate"),
            anomaly_vol_ratio_5d=hit.get("vol_ratio_5d"),
            baseline_index_code=baseline_index_code,
            baseline_close_by_date=baseline_close_by_date,
        )
        candidate_rows.append(rec)

    if not candidate_rows:
        return _empty_day()

    feature_matrix = build_feature_frame(candidate_rows=candidate_rows)
    if feature_matrix.empty:
        return _empty_day()

    # 7. Label JOIN.
    label_series = fetch_labels_for_date(
        db,
        anomaly_date=anomaly_date,
        source=label_source,
        threshold_pct=label_threshold_pct,
    )
    realized = _fetch_realized_meta(
        db,
        anomaly_date=anomaly_date,
        label_source=label_source,
    )

    return _assemble_day_bundle(
        feature_matrix=feature_matrix,
        labels=label_series,
        anomaly_date=anomaly_date,
        realized_meta=realized,
    )


def _fetch_realized_meta(
    db: Database,
    *,
    anomaly_date: str,
    label_source: str,
) -> dict[str, dict[str, Any]]:
    """``{ts_code: {'max_ret_5d', 'data_status'}}`` for the anomaly_date.

    ``max_ret_5d`` is always returned (most informative single column for
    audit), regardless of which column powers the label."""
    rows = db.fetchall(
        "SELECT ts_code, max_ret_5d, data_status FROM va_realized_returns "
        "WHERE anomaly_date = ?",
        (anomaly_date,),
    )
    out: dict[str, dict[str, Any]] = {}
    for ts_code, max_ret_5d, data_status in rows:
        out[str(ts_code)] = {
            "max_ret_5d": (
                None if max_ret_5d is None else float(max_ret_5d)
            ),
            "data_status": str(data_status) if data_status is not None else None,
        }
    return out


def _assemble_day_bundle(
    *,
    feature_matrix: pd.DataFrame,
    labels: pd.Series,
    anomaly_date: str,
    realized_meta: dict[str, dict[str, Any]],
) -> _DayBundle:
    """Align labels + realized meta with the feature matrix index."""
    fm = feature_matrix.copy()
    # Align labels (Int64) on ts_code; missing label → <NA> so trainer can drop.
    aligned_label = (
        labels.reindex(fm.index).astype("Int64").rename("label")
    )
    meta_rows: list[dict[str, Any]] = []
    for ts_code in fm.index:
        rec = realized_meta.get(ts_code, {})
        meta_rows.append(
            {
                "ts_code": ts_code,
                "anomaly_date": anomaly_date,
                "max_ret_5d": rec.get("max_ret_5d"),
                "data_status": rec.get("data_status"),
            }
        )
    sample_meta = pd.DataFrame(meta_rows, columns=ckpt.META_COLUMNS)
    return _DayBundle(
        feature_matrix=fm,
        labels=aligned_label,
        sample_meta=sample_meta,
    )


# ---------------------------------------------------------------------------
# Window-level orchestration (with checkpoint resume)
# ---------------------------------------------------------------------------


def collect_training_window(
    *,
    tushare: TushareClient,
    db: Database,
    calendar: TradeCalendar,
    start_date: str,
    end_date: str,
    label_source: str = "max_ret_5d",
    label_threshold_pct: float = 5.0,
    daily_lookback: int = DEFAULT_DAILY_LOOKBACK,
    moneyflow_lookback: int = DEFAULT_MONEYFLOW_LOOKBACK,
    baseline_index_code: str = DEFAULT_BASELINE_INDEX_CODE,
    main_board_only: bool = True,
    force_sync: bool = False,
    checkpoint_resume: bool = True,
    plugin_version: str | None = None,
    on_day: Callable[[str, int, int], None] | None = None,
) -> VaLgbDataset:
    """Replay [start_date, end_date] into a fully-assembled VaLgbDataset.

    Phase-1 checkpoint
    ------------------
    On every call we open / create a checkpoint at
    ``<plugin_data_dir>/checkpoints/<digest>/``. ``<digest>`` is BLAKE2b-64
    of the fingerprint (window + label config + lookbacks + schema_version +
    main_board_only + baseline_index_code). Days that already have a shard
    on disk are skipped; freshly processed days are saved before the next
    one starts. Train success deletes the checkpoint; failures leave shards
    behind so a rerun resumes from where it crashed.

    Parameters
    ----------
    on_day
        Optional progress callback ``(anomaly_date, day_n_samples, cumulative)``.
    """
    fingerprint = ckpt.CheckpointFingerprint(
        start_date=start_date,
        end_date=end_date,
        schema_version=SCHEMA_VERSION,
        label_threshold_pct=float(label_threshold_pct),
        label_source=label_source,
        daily_lookback=int(daily_lookback),
        moneyflow_lookback=int(moneyflow_lookback),
        main_board_only=bool(main_board_only),
        baseline_index_code=baseline_index_code,
    )
    digest = fingerprint.digest()
    if checkpoint_resume:
        state = ckpt.open_or_create(fingerprint, plugin_version=plugin_version)
        already_done = ckpt.completed_dates(digest)
        if already_done - set(state.completed_dates):
            state.completed_dates = sorted(
                set(state.completed_dates) | already_done
            )
            ckpt.save_state(state)
    else:
        ckpt.delete_checkpoint(digest)
        state = ckpt.open_or_create(fingerprint, plugin_version=plugin_version)
        already_done = set()

    anomaly_dates = enumerate_anomaly_dates(
        db, start_date=start_date, end_date=end_date
    )
    cumulative = 0
    for T in anomaly_dates:
        if T in already_done:
            if on_day is not None:
                on_day(T, -1, cumulative)
            continue
        bundle = collect_day_samples(
            tushare=tushare,
            db=db,
            calendar=calendar,
            anomaly_date=T,
            label_source=label_source,
            label_threshold_pct=label_threshold_pct,
            daily_lookback=daily_lookback,
            moneyflow_lookback=moneyflow_lookback,
            baseline_index_code=baseline_index_code,
            force_sync=force_sync,
        )
        shard_df = ckpt.day_bundle_to_shard(
            feature_matrix=bundle.feature_matrix,
            labels=bundle.labels,
            sample_meta=bundle.sample_meta,
        )
        ckpt.save_day_shard(digest, T, shard_df)
        ckpt.record_day_done(digest, T)
        cumulative += int(len(shard_df))
        if on_day is not None:
            on_day(T, int(len(shard_df)), cumulative)

    return ckpt.assemble_full_dataset(
        digest,
        label_threshold_pct=label_threshold_pct,
        label_source=label_source,
        daily_lookback=daily_lookback,
        moneyflow_lookback=moneyflow_lookback,
        anomaly_dates=anomaly_dates,
    )


__all__ = [
    "DEFAULT_BASELINE_INDEX_CODE",
    "DEFAULT_DAILY_LOOKBACK",
    "DEFAULT_MONEYFLOW_LOOKBACK",
    "VaLgbDataset",
    "collect_day_samples",
    "collect_training_window",
    "enumerate_anomaly_dates",
]
