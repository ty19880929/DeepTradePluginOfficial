"""训练数据收集器 — 把多个交易日的特征 + 标签拼成可训练矩阵。

设计文档 §6.1：本模块复用 :mod:`limit_up_board.data` 的数据装配 helper
（``apply_market_filter`` / ``build_lhb_rollup`` / ``build_cyq_lookup`` /
``index_by_code``）+ :func:`lgb.features.build_feature_frame` +
:func:`lgb.labels.compute_label_for_t1`，**不进入 R1/R2 LLM 阶段**。

API 单一入口：:func:`collect_training_window`。返回 :class:`LgbDataset`，
可被 :func:`trainer.train_lightgbm` 直接消费。

PR-1.2 范围
-----------
* :class:`LgbDataset`
* :func:`collect_training_window`
* :func:`collect_day_samples`

PR-1.3 才会引入 lightgbm 调用；本模块只构造数据，不依赖 LightGBM。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from deeptrade.core.tushare_client import TushareClient

from ..calendar import TradeCalendar
from ..data import (
    SectorStrength,
    apply_market_filter,
    build_cyq_lookup,
    build_lhb_rollup,
    exclude_st,
    index_by_code,
    main_board_filter,
    resolve_sector_strength,
    summarize_limit_step,
    try_optional,
)
from .features import FEATURE_NAMES, SCHEMA_VERSION, build_feature_frame
from .labels import (
    DEFAULT_LABEL_THRESHOLD_PCT,
    compute_label_for_t1,
    compute_max_upside_pct,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------


@dataclass
class LgbDataset:
    """训练矩阵 + 标签 + 样本元信息。

    Attributes
    ----------
    feature_matrix
        ``(n_samples × n_features)``；列 = :data:`features.FEATURE_NAMES`，
        index = 整型 0..n-1。
    labels
        nullable ``Int64`` 列；``<NA>`` 表示 T+1 数据缺失，调用方应在
        训练前过滤这些行。
    sample_index
        每行的 ``ts_code / trade_date / next_trade_date / pct_chg_t1``。
    split_groups
        按 ``trade_date`` 整数化的分组键，供 ``sklearn.model_selection.GroupKFold``
        使用——避免同一交易日的样本同时出现在训练 / 验证集（信息泄漏）。
    schema_version
        :data:`features.SCHEMA_VERSION` 的快照；与模型 ``meta.json`` 对比，
        旧 schema 自动拒绝加载。
    """

    feature_matrix: pd.DataFrame
    labels: pd.Series
    sample_index: pd.DataFrame
    split_groups: pd.Series
    schema_version: int = SCHEMA_VERSION
    daily_lookback: int = 30
    moneyflow_lookback: int = 5
    label_threshold_pct: float = DEFAULT_LABEL_THRESHOLD_PCT
    trade_dates: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(len(self.feature_matrix))

    @property
    def n_positive(self) -> int:
        return int(self.labels.fillna(-1).eq(1).sum())

    @property
    def n_labeled(self) -> int:
        return int(self.labels.notna().sum())

    def filter_labeled(self) -> LgbDataset:
        """Return a new dataset containing only samples whose label is not NA."""
        mask = self.labels.notna().to_numpy()
        return LgbDataset(
            feature_matrix=self.feature_matrix.loc[mask].reset_index(drop=True),
            labels=self.labels.loc[mask].reset_index(drop=True).astype("Int64"),
            sample_index=self.sample_index.loc[mask].reset_index(drop=True),
            split_groups=self.split_groups.loc[mask].reset_index(drop=True),
            schema_version=self.schema_version,
            daily_lookback=self.daily_lookback,
            moneyflow_lookback=self.moneyflow_lookback,
            label_threshold_pct=self.label_threshold_pct,
            trade_dates=list(self.trade_dates),
        )


# ---------------------------------------------------------------------------
# Per-day collection
# ---------------------------------------------------------------------------


@dataclass
class _DayBundle:
    """中间结构：单个交易日的特征矩阵 + 标签 + 样本元信息。"""

    feature_matrix: pd.DataFrame   # index=ts_code, columns=FEATURE_NAMES
    labels: pd.Series              # Int64, index=ts_code
    sample_meta: pd.DataFrame      # ts_code / trade_date / next_trade_date / pct_chg_t1


def _empty_day(trade_date: str) -> _DayBundle:
    return _DayBundle(
        feature_matrix=pd.DataFrame(columns=FEATURE_NAMES),
        labels=pd.Series([], dtype="Int64", name="label"),
        sample_meta=pd.DataFrame(
            columns=["ts_code", "trade_date", "next_trade_date", "pct_chg_t1"]
        ),
    )


def collect_day_samples(
    *,
    tushare: TushareClient,
    trade_date: str,
    next_trade_date: str | None,
    main_pool: pd.DataFrame,
    max_float_mv_yi: float,
    max_close_yuan: float,
    label_threshold_pct: float,
    daily_lookback: int = 30,
    moneyflow_lookback: int = 5,
    prev_trade_date: str | None = None,
    force_sync: bool = False,
) -> _DayBundle:
    """收集单个交易日的训练样本。

    流程（无 R1/R2 LLM 阶段）：
        1. ``limit_list_d(T, limit='U')`` → 候选集
        2. main_board 过滤 + market filter（流通市值/股价）
        3. ST 排除（``stock_st(T)`` REQUIRED）
        4. 支撑数据 (top_list/top_inst/cyq_perf/THS/cpt/limit_step) 拉取
        5. daily / daily_basic / moneyflow 窗口
        6. 调用 :func:`build_feature_frame` 生成特征
        7. ``daily(T+1)`` 中取 pre_close & high 标签化
    """
    # 1. 涨停标的
    limit_list_d = tushare.call(
        "limit_list_d",
        trade_date=trade_date,
        params={"limit_type": "U"},
        force_sync=force_sync,
    )
    if "limit" in limit_list_d.columns:
        limit_list_d = limit_list_d[limit_list_d["limit"] == "U"]
    if limit_list_d.empty:
        return _empty_day(trade_date)

    candidates_df = limit_list_d.merge(
        main_pool[["ts_code", "market", "exchange", "industry", "list_date"]].rename(
            columns={"industry": "industry_basic"}
        ),
        on="ts_code",
        how="inner",
    )
    if candidates_df.empty:
        return _empty_day(trade_date)

    candidates_df, _filter_summary = apply_market_filter(
        candidates_df,
        max_float_mv_yi=max_float_mv_yi,
        max_close_yuan=max_close_yuan,
    )
    if candidates_df.empty:
        return _empty_day(trade_date)

    # 2. ST exclusion
    st_df = tushare.call("stock_st", trade_date=trade_date, force_sync=force_sync)
    st_codes = set(st_df["ts_code"].astype(str)) if not st_df.empty else set()
    candidates_df = exclude_st(candidates_df, st_codes)
    if candidates_df.empty:
        return _empty_day(trade_date)

    candidate_codes = set(candidates_df["ts_code"].astype(str))

    # 3. 支撑数据（同 collect_round1，但任何一个失败都不中断训练——optional 包装）
    top_list_df, _ = try_optional(
        tushare, "top_list", trade_date=trade_date, force_sync=force_sync
    )
    top_inst_df, _ = try_optional(
        tushare, "top_inst", trade_date=trade_date, force_sync=force_sync
    )
    cyq_df, _ = try_optional(
        tushare, "cyq_perf", trade_date=trade_date, force_sync=force_sync
    )
    ths_df, _ = try_optional(
        tushare,
        "limit_list_ths",
        trade_date=trade_date,
        params={"limit_type": "U"},
        force_sync=force_sync,
    )
    cpt_df, _ = try_optional(
        tushare, "limit_cpt_list", trade_date=trade_date, force_sync=force_sync
    )
    step_df, _ = try_optional(
        tushare, "limit_step", trade_date=trade_date, force_sync=force_sync
    )

    sector = resolve_sector_strength(
        candidates=candidates_df,
        limit_cpt_list=cpt_df if cpt_df is not None else None,
        limit_list_ths=ths_df if ths_df is not None else None,
    )

    # 4. market summary（包含 yesterday context，仅当 prev_trade_date 给出时填）
    today_step = summarize_limit_step(step_df) if step_df is not None else {}
    market_summary: dict[str, Any] = {
        "limit_up_count": int(len(candidates_df)),
        "limit_step_distribution": today_step,
    }
    if prev_trade_date is not None:
        from ..data import _collect_yesterday_context  # local import to keep module-init light

        yctx, _yctx_err = _collect_yesterday_context(
            tushare,
            trade_date=trade_date,
            prev_trade_date=prev_trade_date,
            today_step=today_step,
            force_sync=force_sync,
        )
        market_summary.update(yctx)

    # 5. 历史窗口（daily / daily_basic / moneyflow）
    daily_start = _shift_yyyymmdd(trade_date, -(daily_lookback * 2))
    mf_start = _shift_yyyymmdd(trade_date, -(moneyflow_lookback + 5))
    daily_df = tushare.call(
        "daily",
        params={"start_date": daily_start, "end_date": trade_date},
        force_sync=force_sync,
    )
    daily_basic_df = tushare.call(
        "daily_basic",
        params={"start_date": daily_start, "end_date": trade_date},
        force_sync=force_sync,
    )
    moneyflow_df = tushare.call(
        "moneyflow",
        params={"start_date": mf_start, "end_date": trade_date},
        force_sync=force_sync,
    )
    if daily_df is not None and not daily_df.empty and "ts_code" in daily_df.columns:
        daily_df = daily_df[daily_df["ts_code"].astype(str).isin(candidate_codes)]
    if (
        daily_basic_df is not None
        and not daily_basic_df.empty
        and "ts_code" in daily_basic_df.columns
    ):
        daily_basic_df = daily_basic_df[
            daily_basic_df["ts_code"].astype(str).isin(candidate_codes)
        ]
    if (
        moneyflow_df is not None
        and not moneyflow_df.empty
        and "ts_code" in moneyflow_df.columns
    ):
        moneyflow_df = moneyflow_df[
            moneyflow_df["ts_code"].astype(str).isin(candidate_codes)
        ]

    daily_by_code = index_by_code(daily_df)
    daily_basic_by_code = index_by_code(daily_basic_df)
    moneyflow_by_code = index_by_code(moneyflow_df)

    lhb_rollup = build_lhb_rollup(top_list_df, top_inst_df)
    cyq_lookup = build_cyq_lookup(cyq_df)

    # 6. features
    feature_df = build_feature_frame(
        candidates_df=candidates_df,
        daily_by_code=daily_by_code,
        daily_basic_by_code=daily_basic_by_code,
        moneyflow_by_code=moneyflow_by_code,
        cyq_by_code=cyq_lookup,
        lhb_rollup=lhb_rollup,
        sector_strength=sector,
        market_summary=market_summary,
        trade_date=trade_date,
        daily_lookback=daily_lookback,
        moneyflow_lookback=moneyflow_lookback,
    )

    # 7. labels — T+1 daily
    labels_dict: dict[str, int | None] = {}
    pct_chg_t1: dict[str, float | None] = {}
    if next_trade_date is not None:
        t1_daily = tushare.call(
            "daily", trade_date=next_trade_date, force_sync=force_sync
        )
        if t1_daily is not None and not t1_daily.empty:
            t1_lookup = (
                t1_daily.set_index("ts_code").to_dict(orient="index")
                if "ts_code" in t1_daily.columns
                else {}
            )
            for ts in feature_df.index:
                row = t1_lookup.get(ts)
                labels_dict[str(ts)] = compute_label_for_t1(
                    row, threshold_pct=label_threshold_pct
                )
                pct_chg_t1[str(ts)] = compute_max_upside_pct(row)
    for ts in feature_df.index:
        labels_dict.setdefault(str(ts), None)
        pct_chg_t1.setdefault(str(ts), None)

    label_series = pd.Series(
        [labels_dict[ts] for ts in feature_df.index],
        index=feature_df.index,
        dtype="Int64",
        name="label",
    )
    pct_chg_t1_series = pd.Series(
        [pct_chg_t1[ts] for ts in feature_df.index],
        index=feature_df.index,
        dtype="Float64",
        name="pct_chg_t1",
    )

    sample_meta = pd.DataFrame(
        {
            "ts_code": list(feature_df.index),
            "trade_date": [trade_date] * len(feature_df),
            "next_trade_date": [next_trade_date] * len(feature_df),
            "pct_chg_t1": pct_chg_t1_series.to_numpy(),
        }
    )

    return _DayBundle(
        feature_matrix=feature_df,
        labels=label_series,
        sample_meta=sample_meta,
    )


def _shift_yyyymmdd(yyyymmdd: str, days: int) -> str:
    from datetime import datetime, timedelta

    d = datetime.strptime(yyyymmdd, "%Y%m%d") + timedelta(days=days)
    return d.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Window orchestrator
# ---------------------------------------------------------------------------


def _enumerate_trade_dates(
    calendar: TradeCalendar, start_date: str, end_date: str
) -> list[str]:
    """All open trading days T with ``start_date <= T <= end_date``."""
    if start_date > end_date:
        return []
    df = calendar._df  # noqa: SLF001 — TradeCalendar 没暴露区间 API；直接读私有
    mask = (df["cal_date"] >= start_date) & (df["cal_date"] <= end_date)
    rows = df[mask & (df["is_open"] == 1)]
    return [str(d) for d in rows["cal_date"].tolist()]


def collect_training_window(
    *,
    tushare: TushareClient,
    calendar: TradeCalendar,
    start_date: str,
    end_date: str,
    max_float_mv_yi: float = 100.0,
    max_close_yuan: float = 15.0,
    label_threshold_pct: float = DEFAULT_LABEL_THRESHOLD_PCT,
    main_board_pool: pd.DataFrame | None = None,
    daily_lookback: int = 30,
    moneyflow_lookback: int = 5,
    force_sync: bool = False,
    on_day: Callable[[str, int, int], None] | None = None,
) -> LgbDataset:
    """对 ``[start_date, end_date]`` 内的每个交易日 T 收集训练样本。

    Parameters
    ----------
    on_day
        进度回调 ``(trade_date, day_n_samples, cum_n_samples)``——
        train CLI 用它打印 ``[YYYYMMDD] +N samples (cum. K)``。

    Returns
    -------
    LgbDataset
        :data:`feature_matrix` 用 0..n-1 整型 index，``ts_code`` 移到
        ``sample_index['ts_code']``。``labels`` 保留 ``<NA>``（T+1 数据
        缺失），训练前调用 :meth:`LgbDataset.filter_labeled` 过滤。
    """
    if start_date > end_date:
        raise ValueError(f"start_date {start_date!r} > end_date {end_date!r}")

    if main_board_pool is None:
        stock_basic = tushare.call("stock_basic", force_sync=force_sync)
        main_pool = main_board_filter(stock_basic)
    else:
        main_pool = main_board_pool

    trade_dates = _enumerate_trade_dates(calendar, start_date, end_date)

    feature_frames: list[pd.DataFrame] = []
    label_series_list: list[pd.Series] = []
    sample_meta_frames: list[pd.DataFrame] = []
    cum = 0
    for T in trade_dates:
        try:
            next_open = calendar.next_open(T)
        except ValueError:
            next_open = None  # T+1 unknown → 该日全部样本 label = <NA>
        try:
            prev_open = calendar.pretrade_date(T)
        except ValueError:
            prev_open = None

        day_bundle = collect_day_samples(
            tushare=tushare,
            trade_date=T,
            next_trade_date=next_open,
            main_pool=main_pool,
            max_float_mv_yi=max_float_mv_yi,
            max_close_yuan=max_close_yuan,
            label_threshold_pct=label_threshold_pct,
            daily_lookback=daily_lookback,
            moneyflow_lookback=moneyflow_lookback,
            prev_trade_date=prev_open,
            force_sync=force_sync,
        )
        n_today = int(len(day_bundle.feature_matrix))
        if n_today > 0:
            # ts_code from index → 列；为 concat 重置 index
            feature_frames.append(day_bundle.feature_matrix.reset_index(drop=True))
            label_series_list.append(day_bundle.labels.reset_index(drop=True))
            sample_meta_frames.append(day_bundle.sample_meta.reset_index(drop=True))
        cum += n_today
        if on_day is not None:
            on_day(T, n_today, cum)
        else:
            logger.info("[%s] +%d samples (cum. %d)", T, n_today, cum)

    if not feature_frames:
        empty_feat = pd.DataFrame(columns=FEATURE_NAMES)
        empty_labels = pd.Series([], dtype="Int64", name="label")
        empty_meta = pd.DataFrame(
            columns=["ts_code", "trade_date", "next_trade_date", "pct_chg_t1"]
        )
        empty_groups = pd.Series([], dtype="Int64", name="split_group")
        return LgbDataset(
            feature_matrix=empty_feat,
            labels=empty_labels,
            sample_index=empty_meta,
            split_groups=empty_groups,
            label_threshold_pct=label_threshold_pct,
            trade_dates=trade_dates,
        )

    feature_matrix = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_series_list, ignore_index=True)
    sample_index = pd.concat(sample_meta_frames, ignore_index=True)
    split_groups = (
        sample_index["trade_date"].astype(int).astype("Int64").rename("split_group")
    )

    # 列顺序锁死 = FEATURE_NAMES（设计 §3.3 schema 单一来源）。
    feature_matrix = feature_matrix.reindex(columns=FEATURE_NAMES)

    return LgbDataset(
        feature_matrix=feature_matrix,
        labels=labels.astype("Int64").rename("label"),
        sample_index=sample_index.reset_index(drop=True),
        split_groups=split_groups.reset_index(drop=True),
        daily_lookback=daily_lookback,
        moneyflow_lookback=moneyflow_lookback,
        label_threshold_pct=label_threshold_pct,
        trade_dates=trade_dates,
    )
