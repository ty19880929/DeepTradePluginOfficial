"""LightGBM 训练 + 推理共用特征工程（设计 §4）。

唯一 entrypoint: :func:`build_feature_frame`。训练管线（``lgb.dataset``）和
推理路径（``data.collect_analyze_bundle``）都通过这里产生同一份特征矩阵，
杜绝 train/serve skew。

关键约束（设计 §4.1）：
* **不重新拉数据**——只对 ``data._build_candidate_row`` 已经产出的字段做
  重命名 / 派生 / 比率 / 类型转换；缺失字段一律返回 NaN，由 LightGBM 原生
  missing-value handling 接收。
* ``FEATURE_NAMES`` 是列名 + 顺序的单一来源。任何模型加载时的
  ``feature_name()`` 与本表不一致即拒绝加载（见 :class:`LgbModelSchemaMismatch`
  in scorer，PR-2.1）。
* ``SCHEMA_VERSION`` 在任何特征清单变动时 bump，旧模型自动被 scorer 拒载。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# Sentinel for "this field is part of the contract but no underlying signal
# is available yet" — value is intentionally NaN rather than 0 so the booster
# can route on missingness.
_NA = float("nan")


# ---------------------------------------------------------------------------
# Feature list (single source of truth) — design §4.2
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # ---- 异动核心 -----------------------------------------------------
    "f_vol_anomaly_pct_chg",
    "f_vol_anomaly_body_ratio",
    "f_vol_anomaly_turnover_rate",
    "f_vol_anomaly_vol_ratio_5d",
    # ---- 量能 ---------------------------------------------------------
    "f_vol_ratio_5d_t",
    "f_vol_ratio_60d_t",
    "f_vol_amount_yi_t",
    "f_vol_max_vol_60d_ratio",
    # ---- 形态 / 均线 --------------------------------------------------
    "f_mom_close_to_ma5_pct",
    "f_mom_close_to_ma10_pct",
    "f_mom_close_to_ma20_pct",
    "f_mom_close_to_ma60_pct",
    "f_mom_above_ma20",
    "f_mom_above_ma60",
    # ---- 形态 / 区间 --------------------------------------------------
    "f_mom_range_pct_60d",
    "f_mom_pct_chg_60d",
    # ---- 阻力位 -------------------------------------------------------
    "f_mom_dist_to_120d_high_pct",
    "f_mom_dist_to_250d_high_pct",
    "f_mom_is_above_120d_high",
    "f_mom_is_above_250d_high",
    "f_mom_pos_in_120d_range",
    # ---- VCP / 波动率 -------------------------------------------------
    "f_vcp_atr_10d_pct",
    "f_vcp_atr_10d_quantile_60d",
    "f_vcp_bbw_20d",
    "f_vcp_bbw_compression_ratio",
    # ---- 洗盘 ---------------------------------------------------------
    "f_wash_base_days",
    "f_wash_base_drawdown_pct",
    "f_wash_base_vol_shrink_ratio",
    "f_wash_base_avg_turnover",
    "f_wash_days_since_last_limit_up",
    "f_wash_prior_limit_up_count_60d",
    # ---- 资金流 -------------------------------------------------------
    "f_mf_cum_net_5d_yi",
    "f_mf_cum_elg_lg_5d_yi",
    "f_mf_trend_rising",
    "f_mf_rows_used",
    # ---- 换手 ---------------------------------------------------------
    "f_chip_turnover_rate_t",
    "f_chip_volume_ratio_t",
    "f_chip_pe_t",
    "f_chip_pb_t",
    # ---- 相对强度 -----------------------------------------------------
    "f_alpha_5d_pct",
    "f_alpha_20d_pct",
    "f_alpha_60d_pct",
    "f_alpha_leading",
    # ---- 板块 ---------------------------------------------------------
    "f_sec_strength_source_rank",
    "f_sec_today_industry_up_count",
    # ---- 市场 ---------------------------------------------------------
    "f_mkt_total_limit_up",
    "f_mkt_yesterday_failure_rate",
    # ---- 静态属性 -----------------------------------------------------
    "f_st_circ_mv_yi",
    "f_st_total_mv_yi",
    "f_st_last_close_yuan",
    "f_st_tracked_days",
    "f_st_listed_days",
]


# Ratio-style fields that are clipped to [-CLIP, CLIP] to guard against ÷-by-near-0.
_CLIP_BOUND = 500.0
_CLIP_FIELDS = {
    "f_mom_close_to_ma5_pct",
    "f_mom_close_to_ma10_pct",
    "f_mom_close_to_ma20_pct",
    "f_mom_close_to_ma60_pct",
    "f_mom_range_pct_60d",
    "f_mom_pct_chg_60d",
    "f_mom_dist_to_120d_high_pct",
    "f_mom_dist_to_250d_high_pct",
    "f_alpha_5d_pct",
    "f_alpha_20d_pct",
    "f_alpha_60d_pct",
    "f_vcp_bbw_compression_ratio",
    "f_vol_ratio_5d_t",
    "f_vol_ratio_60d_t",
    "f_vol_max_vol_60d_ratio",
    "f_wash_base_vol_shrink_ratio",
    "f_wash_base_drawdown_pct",
    "f_vcp_atr_10d_pct",
    "f_vcp_bbw_20d",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LgbFeatureSchemaError(ValueError):
    """Raised when a feature frame's columns disagree with FEATURE_NAMES."""


def assert_columns(df: pd.DataFrame, *, expected: list[str] | None = None) -> None:
    """Raise :class:`LgbFeatureSchemaError` if columns / order differ.

    Trainer + scorer both call this at the seam so a schema mismatch is loud.
    """
    cols = expected if expected is not None else FEATURE_NAMES
    if list(df.columns) != list(cols):
        missing = [c for c in cols if c not in df.columns]
        extra = [c for c in df.columns if c not in cols]
        raise LgbFeatureSchemaError(
            f"feature columns mismatch: missing={missing!r} extra={extra!r}"
        )


def build_feature_frame(
    *,
    candidate_rows: list[dict[str, Any]],
    market_summary: dict[str, Any] | None = None,
    sector_strength_data: dict[str, Any] | None = None,
    sector_strength_source: str | None = None,
) -> pd.DataFrame:
    """Build the (n_candidates × n_features) feature matrix.

    Parameters
    ----------
    candidate_rows
        Dictionaries produced by ``data._build_candidate_row``. Each must
        carry ``ts_code``; other fields are best-effort (missing → NaN).
    market_summary
        ``AnalyzeBundle.market_summary``; currently consulted for optional
        ``limit_up_count`` / ``yesterday_failure_rate`` fields (NaN when
        absent).
    sector_strength_data
        ``AnalyzeBundle.sector_strength_data``; used to size
        ``f_sec_today_industry_up_count`` per ts_code (NaN when absent).
    sector_strength_source
        One of ``limit_cpt_list`` / ``lu_desc_aggregation`` /
        ``industry_fallback``; maps to ``f_sec_strength_source_rank`` ∈
        {1, 2, 3}.

    Returns
    -------
    pd.DataFrame
        Index = ``ts_code``; columns = ``FEATURE_NAMES`` (exact order).
    """
    if not candidate_rows:
        return pd.DataFrame(columns=FEATURE_NAMES)

    market_summary = market_summary or {}
    sector_strength_data = sector_strength_data or {}

    source_rank = _sector_source_rank(sector_strength_source)
    mkt_total_limit_up = _coerce_float(market_summary.get("limit_up_count"))
    yest_failure = market_summary.get("yesterday_failure_rate")
    if isinstance(yest_failure, dict):
        mkt_yesterday_failure_rate = _coerce_float(yest_failure.get("rate_pct"))
    else:
        mkt_yesterday_failure_rate = _coerce_float(yest_failure)

    industry_up_by_code = _industry_up_count_lookup(
        candidate_rows, sector_strength_data
    )

    rows: list[dict[str, float]] = []
    index: list[str] = []
    for row in candidate_rows:
        ts_code = str(row.get("ts_code", ""))
        index.append(ts_code)
        feat = _features_for_row(
            row,
            source_rank=source_rank,
            industry_up_count=industry_up_by_code.get(ts_code, _NA),
            mkt_total_limit_up=mkt_total_limit_up,
            mkt_yesterday_failure_rate=mkt_yesterday_failure_rate,
        )
        rows.append(feat)

    df = pd.DataFrame(rows, index=pd.Index(index, name="ts_code"))
    df = df.reindex(columns=FEATURE_NAMES)
    for col in _CLIP_FIELDS:
        if col in df.columns:
            df[col] = df[col].clip(lower=-_CLIP_BOUND, upper=_CLIP_BOUND)
    df = df.astype(float, copy=False)
    return df


def feature_hash(feature_row: pd.Series | dict[str, Any]) -> str:
    """8-byte BLAKE2b digest of a single sample's feature vector (hex).

    Used by ``va_lgb_predictions.feature_hash`` for audit ↔ replay reconciliation.
    """
    if isinstance(feature_row, pd.Series):
        items = [(str(k), feature_row[k]) for k in FEATURE_NAMES if k in feature_row.index]
    else:
        items = [(k, feature_row.get(k)) for k in FEATURE_NAMES]
    payload = []
    for k, v in items:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            payload.append((k, None))
        else:
            payload.append((k, round(float(v), 6)))
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def feature_missing(feature_row: pd.Series) -> list[str]:
    """List of FEATURE_NAMES that ended up NaN for *feature_row*."""
    missing: list[str] = []
    for name in FEATURE_NAMES:
        if name not in feature_row.index:
            missing.append(name)
            continue
        v = feature_row[name]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# Per-row feature extraction
# ---------------------------------------------------------------------------


def _features_for_row(
    row: dict[str, Any],
    *,
    source_rank: float,
    industry_up_count: float,
    mkt_total_limit_up: float,
    mkt_yesterday_failure_rate: float,
) -> dict[str, float]:
    last_close = _coerce_float(row.get("last_close"))
    ma5 = _coerce_float(row.get("ma5"))
    ma10 = _coerce_float(row.get("ma10"))
    ma20 = _coerce_float(row.get("ma20"))
    ma60 = _coerce_float(row.get("ma60"))

    mf = row.get("moneyflow_5d_summary") or {}
    if not isinstance(mf, dict):
        mf = {}

    return {
        # ---- 异动核心 ------------------------------------------------------
        "f_vol_anomaly_pct_chg": _coerce_float(row.get("anomaly_pct_chg")),
        "f_vol_anomaly_body_ratio": _coerce_float(row.get("anomaly_body_ratio")),
        "f_vol_anomaly_turnover_rate": _coerce_float(row.get("anomaly_turnover_rate")),
        "f_vol_anomaly_vol_ratio_5d": _coerce_float(row.get("anomaly_vol_ratio_5d")),
        # ---- 量能 ---------------------------------------------------------
        # candidate_row 不带 history → 训练侧需要补；用 anomaly_vol_ratio_5d 作为
        # 5-day 量比代理（语义等价：异动当日量比即 vol_t / mean(vol_{t-5..t-1})）
        "f_vol_ratio_5d_t": _coerce_float(row.get("anomaly_vol_ratio_5d")),
        "f_vol_ratio_60d_t": _NA,
        "f_vol_amount_yi_t": _NA,
        "f_vol_max_vol_60d_ratio": _NA,
        # ---- 形态 / 均线 --------------------------------------------------
        "f_mom_close_to_ma5_pct": _pct_diff(last_close, ma5),
        "f_mom_close_to_ma10_pct": _pct_diff(last_close, ma10),
        "f_mom_close_to_ma20_pct": _pct_diff(last_close, ma20),
        "f_mom_close_to_ma60_pct": _pct_diff(last_close, ma60),
        "f_mom_above_ma20": _bool_to_int(row.get("above_ma20")),
        "f_mom_above_ma60": _bool_to_int(row.get("above_ma60")),
        # ---- 形态 / 区间 --------------------------------------------------
        "f_mom_range_pct_60d": _coerce_float(row.get("range_pct_60d")),
        "f_mom_pct_chg_60d": _coerce_float(row.get("pct_chg_60d")),
        # ---- 阻力位 -------------------------------------------------------
        "f_mom_dist_to_120d_high_pct": _coerce_float(row.get("dist_to_120d_high_pct")),
        "f_mom_dist_to_250d_high_pct": _coerce_float(row.get("dist_to_250d_high_pct")),
        "f_mom_is_above_120d_high": _bool_to_int(row.get("is_above_120d_high")),
        "f_mom_is_above_250d_high": _bool_to_int(row.get("is_above_250d_high")),
        "f_mom_pos_in_120d_range": _coerce_float(row.get("pos_in_120d_range")),
        # ---- VCP / 波动率 -------------------------------------------------
        "f_vcp_atr_10d_pct": _coerce_float(row.get("atr_10d_pct")),
        "f_vcp_atr_10d_quantile_60d": _coerce_float(row.get("atr_10d_quantile_in_60d")),
        "f_vcp_bbw_20d": _coerce_float(row.get("bbw_20d")),
        "f_vcp_bbw_compression_ratio": _coerce_float(row.get("bbw_compression_ratio")),
        # ---- 洗盘 ---------------------------------------------------------
        "f_wash_base_days": _coerce_float(row.get("base_days")),
        "f_wash_base_drawdown_pct": _coerce_float(row.get("base_max_drawdown_pct")),
        "f_wash_base_vol_shrink_ratio": _coerce_float(row.get("base_vol_shrink_ratio")),
        # candidate_row 当前未输出 base_avg_turnover_rate（key 已声明但不填）
        "f_wash_base_avg_turnover": _coerce_float(row.get("base_avg_turnover_rate")),
        "f_wash_days_since_last_limit_up": _coerce_float(
            row.get("days_since_last_limit_up")
        ),
        "f_wash_prior_limit_up_count_60d": _coerce_float(
            row.get("prior_limit_up_count_60d"), default_for_none=0.0
        ),
        # ---- 资金流 -------------------------------------------------------
        "f_mf_cum_net_5d_yi": _coerce_float(mf.get("cum_net_mf_yi")),
        "f_mf_cum_elg_lg_5d_yi": _coerce_float(mf.get("cum_elg_plus_lg_buy_yi")),
        "f_mf_trend_rising": _mf_trend_code(mf.get("net_mf_trend")),
        "f_mf_rows_used": _coerce_float(mf.get("rows_used"), default_for_none=0.0),
        # ---- 换手 ---------------------------------------------------------
        "f_chip_turnover_rate_t": _coerce_float(row.get("turnover_rate_t")),
        "f_chip_volume_ratio_t": _coerce_float(row.get("volume_ratio_t")),
        "f_chip_pe_t": _coerce_float(row.get("pe_t")),
        "f_chip_pb_t": _coerce_float(row.get("pb_t")),
        # ---- 相对强度 -----------------------------------------------------
        "f_alpha_5d_pct": _coerce_float(row.get("alpha_5d_pct")),
        "f_alpha_20d_pct": _coerce_float(row.get("alpha_20d_pct")),
        "f_alpha_60d_pct": _coerce_float(row.get("alpha_60d_pct")),
        "f_alpha_leading": _alpha_leading_code(row.get("rel_strength_label")),
        # ---- 板块 ---------------------------------------------------------
        "f_sec_strength_source_rank": source_rank,
        "f_sec_today_industry_up_count": industry_up_count,
        # ---- 市场 ---------------------------------------------------------
        "f_mkt_total_limit_up": mkt_total_limit_up,
        "f_mkt_yesterday_failure_rate": mkt_yesterday_failure_rate,
        # ---- 静态属性 -----------------------------------------------------
        "f_st_circ_mv_yi": _coerce_float(row.get("circ_mv_yi")),
        "f_st_total_mv_yi": _coerce_float(row.get("total_mv_yi")),
        "f_st_last_close_yuan": _coerce_float(row.get("last_close")),
        # tracked_days 在训练样本中统一被 dataset.py 显式置 0；推理时是实际值。
        # 设计 §4.3 把 tracked_days 视为唯一受控的 train-serve skew。
        "f_st_tracked_days": _coerce_float(
            row.get("tracked_days"), default_for_none=0.0
        ),
        "f_st_listed_days": _coerce_float(row.get("listed_days")),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_float(value: Any, *, default_for_none: float | None = None) -> float:
    if value is None or value == "":
        return float(default_for_none) if default_for_none is not None else _NA
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _NA
    if math.isnan(f) or math.isinf(f):
        return _NA if math.isnan(f) else _NA
    return f


def _bool_to_int(value: Any) -> float:
    if value is None:
        return _NA
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return 1.0 if float(value) else 0.0
        except (TypeError, ValueError):
            return _NA
    return _NA


def _pct_diff(numerator: float, denominator: float) -> float:
    if numerator is None or denominator is None:
        return _NA
    if math.isnan(numerator) or math.isnan(denominator):
        return _NA
    if denominator == 0:
        return _NA
    return (numerator - denominator) / denominator * 100.0


def _mf_trend_code(trend: Any) -> float:
    if trend == "rising":
        return 1.0
    if trend == "falling":
        return -1.0
    if trend == "flat":
        return 0.0
    return _NA


def _alpha_leading_code(label: Any) -> float:
    if label == "leading":
        return 1.0
    if label == "in_line":
        return 0.0
    if label == "lagging":
        return -1.0
    return _NA


def _sector_source_rank(source: str | None) -> float:
    if source == "limit_cpt_list":
        return 1.0
    if source == "lu_desc_aggregation":
        return 2.0
    if source == "industry_fallback":
        return 3.0
    return _NA


def _industry_up_count_lookup(
    candidate_rows: list[dict[str, Any]],
    sector_strength_data: dict[str, Any],
) -> dict[str, float]:
    """Map each candidate's ts_code → today's industry up-count (NaN if unknown).

    Two source shapes are supported:

    * ``limit_cpt_list`` rows ⇒ ``top_sectors[*].{name, count}`` (today's
      涨停板块).
    * ``industry_fallback`` rows ⇒ ``top_sectors[*].{sector, watchlist_count}``
      (watchlist 行业聚合 — 不是真正的涨停统计，但作为代理填入).
    """
    sectors = sector_strength_data.get("top_sectors") if sector_strength_data else None
    if not sectors:
        return {}
    by_industry: dict[str, float] = {}
    for entry in sectors:
        if not isinstance(entry, dict):
            continue
        name = (
            entry.get("name")
            or entry.get("sector")
            or entry.get("industry")
        )
        count = entry.get("count") or entry.get("watchlist_count") or entry.get("up_count")
        if name is None or count is None:
            continue
        try:
            by_industry[str(name)] = float(count)
        except (TypeError, ValueError):
            continue
    out: dict[str, float] = {}
    for row in candidate_rows:
        code = str(row.get("ts_code", ""))
        ind = row.get("industry")
        if ind is None:
            continue
        if ind in by_industry:
            out[code] = by_industry[ind]
    return out


# Re-exported for tests that want the numeric NaN sentinel from this module.
NA_FILL = _NA


__all__ = [
    "FEATURE_NAMES",
    "LgbFeatureSchemaError",
    "NA_FILL",
    "SCHEMA_VERSION",
    "assert_columns",
    "build_feature_frame",
    "feature_hash",
    "feature_missing",
]


# Silence "imported but unused" for type checkers that don't pick up the
# numpy use inside helpers (we use ``float("nan")`` directly).
_np_module = np  # noqa: F841
