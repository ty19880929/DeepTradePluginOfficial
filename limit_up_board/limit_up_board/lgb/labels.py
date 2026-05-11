"""T+1 标签构造（仅训练用）。

设计文档 §5。本模块负责把 ``(ts_code, T)`` 训练样本与其下一交易日 ``T+1``
的 daily 行结合起来，产出二分类标签：

``label = 1  iff  (T+1.high - T+1.pre_close) / T+1.pre_close >= label_threshold_pct``

默认阈值 ``9.7``（10cm 主板涨停容差），可经 ``LubConfig.lgb_label_threshold_pct``
覆盖。

设计要点
--------

* 标签语义比 "T+1 是否真实涨停" 更通用：包含 "高开 ≥ 9.8% 但未封死" 的次日溢价
  机会，与 LLM 评估"次日溢价空间"语义对齐。
* 不依赖 ``limit_list_d(T+1)``——只看 daily 一张表，数据完整性更高。
* 缺失 ``pre_close`` 或 ``high`` → 返回 ``None``，由 ``dataset`` 模块决定
  是否丢弃该样本（推荐丢弃，避免后续训练阶段产生隐式 0）。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

DEFAULT_LABEL_THRESHOLD_PCT: float = 9.7


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def compute_label_for_t1(
    daily_t1_row: dict[str, Any] | None,
    *,
    threshold_pct: float = DEFAULT_LABEL_THRESHOLD_PCT,
) -> int | None:
    """Return 1 / 0 / None for one (ts_code, T) sample given its T+1 daily row.

    ``daily_t1_row`` 必须包含 ``pre_close`` 与 ``high``。任一缺失 → 返回 ``None``。
    阈值默认 9.7（即 ``(high - pre_close) / pre_close * 100 >= 9.7`` 判 1）。
    """
    if daily_t1_row is None:
        return None
    pre_close = _to_float(daily_t1_row.get("pre_close"))
    high = _to_float(daily_t1_row.get("high"))
    if pre_close is None or high is None or pre_close <= 0:
        return None
    pct = (high - pre_close) / pre_close * 100.0
    return 1 if pct >= threshold_pct else 0


def compute_max_upside_pct(daily_t1_row: dict[str, Any] | None) -> float | None:
    """辅助：返回 T+1 真实最大溢价 ``(high - pre_close) / pre_close * 100``。

    用于 evaluate-lgb 命令计算 Top-K 真实命中率；本身不参与训练标签。
    """
    if daily_t1_row is None:
        return None
    pre_close = _to_float(daily_t1_row.get("pre_close"))
    high = _to_float(daily_t1_row.get("high"))
    if pre_close is None or high is None or pre_close <= 0:
        return None
    return (high - pre_close) / pre_close * 100.0


def label_dataframe(
    samples_df: pd.DataFrame,
    daily_t1_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    threshold_pct: float = DEFAULT_LABEL_THRESHOLD_PCT,
) -> pd.Series:
    """Batch label构造，给 dataset 模块调用。

    Parameters
    ----------
    samples_df
        含 ``ts_code`` 与 ``next_trade_date`` 两列的训练样本表。
    daily_t1_lookup
        ``{(ts_code, next_trade_date): daily_row_dict}``——从 lub_daily 拉取。
        缺失键意味着 T+1 当天该股停牌 / 数据未覆盖，对应样本会被标为 ``None``。

    Returns
    -------
    pd.Series
        与 ``samples_df`` index 对齐的 nullable ``Int64`` 列（0 / 1 / <NA>）。
    """
    required_cols = {"ts_code", "next_trade_date"}
    missing = required_cols - set(samples_df.columns)
    if missing:
        raise ValueError(f"samples_df missing required columns: {missing}")

    labels: list[int | None] = []
    for ts, t1 in zip(
        samples_df["ts_code"].astype(str),
        samples_df["next_trade_date"].astype(str),
        strict=False,
    ):
        labels.append(
            compute_label_for_t1(
                daily_t1_lookup.get((ts, t1)),
                threshold_pct=threshold_pct,
            )
        )
    return pd.Series(labels, index=samples_df.index, dtype="Int64", name="label")
