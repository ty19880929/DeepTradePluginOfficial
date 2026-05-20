"""LightGBM training + serving feature contract for accumulation-probe-washout.

Single entrypoint :func:`build_feature_frame` — both the training pipeline
(``lgb.dataset`` in PR-3) and the analyse path (``LgbScorer`` in PR-4) use
this function so train/serve skew is impossible by construction.

Hard constraints (mirroring the VA-equivalent file):

* No data fetching — only re-shapes the dicts produced by
  :func:`accumulation_probe_washout.data.pack_candidate`.
* Missing fields → ``NaN``. The LightGBM booster routes NaN natively, so we
  never substitute 0 (that would conflate "feature absent" with "feature
  legitimately zero").
* ``FEATURE_NAMES`` is the single source of truth for column names AND
  order. Any addition / removal / reorder bumps ``SCHEMA_VERSION``; the
  scorer rejects models whose feature_list doesn't match.
* ratio-style columns are clipped to ``[-CLIP_BOUND, CLIP_BOUND]`` to guard
  against divide-by-near-zero blow-ups poisoning the dataset.

PR-2 introduces SCHEMA_VERSION=1. PR-3 may add limit-up history fields
(``f_limit_up_*``) and adj_factor-aware volume ratios — both will bump to
SCHEMA_VERSION=2 with explicit migration notes here.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

SCHEMA_VERSION = 1

# Sentinel used in dict literals where there is intentionally no value.
_NA = float("nan")
_CLIP_BOUND = 500.0


# ---------------------------------------------------------------------------
# FEATURE_NAMES — single source of truth (order matters)
# ---------------------------------------------------------------------------


FEATURE_NAMES: list[str] = [
    # ---- 吸筹 (accumulation) -------------------------------------------------
    "f_acc_score",
    "f_acc_days",
    "f_acc_net_mf_yi",
    "f_acc_price_change_pct",
    "f_acc_low_position_score",
    # ---- 试盘 (probe day) ---------------------------------------------------
    "f_probe_quality",
    "f_probe_vol_ratio_5d",
    "f_probe_vol_ratio_20d",
    "f_probe_vol_rank_pct_60d",
    "f_probe_amount_ratio_20d",
    "f_probe_turnover",
    "f_probe_amplitude_pct",
    "f_probe_upper_shadow_ratio",
    "f_probe_body_ratio",
    "f_probe_pct_chg",
    "f_probe_moneyflow_net_yi",
    "f_probe_days_ago",
    # ---- 洗盘 (washout) -----------------------------------------------------
    "f_wash_score",
    "f_wash_days",
    "f_wash_max_drawdown_pct",
    "f_wash_vol_shrink_ratio",
    "f_wash_low_broken",
    "f_wash_ma20_broken",
    "f_wash_ma60_broken",
    "f_wash_moneyflow_net_yi",
    "f_wash_volatility_compression",
    # ---- 启动 (launch setup) ------------------------------------------------
    "f_launch_setup_score",
    "f_launch_close_to_probe_high_pct",
    "f_launch_break_probe_high",
    "f_launch_break_washout_high",
    "f_launch_cur_vol_ratio_5d",
    "f_launch_cur_vol_ratio_20d",
    "f_launch_cur_moneyflow_net_yi",
    "f_launch_above_ma5",
    "f_launch_above_ma10",
    "f_launch_above_ma20",
    "f_launch_relative_strength_20d",
    # ---- VCP / 波动率 (carried from VA semantics) ---------------------------
    "f_vcp_atr_10d_pct",
    "f_vcp_atr_10d_quantile_60d",
    "f_vcp_bbw_20d",
    "f_vcp_bbw_compression_ratio",
    # ---- 阻力位 (long-range resistance) ------------------------------------
    "f_mom_dist_to_120d_high_pct",
    "f_mom_dist_to_250d_high_pct",
    "f_mom_is_above_120d_high",
    "f_mom_is_above_250d_high",
    "f_mom_pos_in_120d_range",
    # ---- 均线距离 (MA distances) -------------------------------------------
    "f_mom_close_to_ma5_pct",
    "f_mom_close_to_ma10_pct",
    "f_mom_close_to_ma20_pct",
    "f_mom_close_to_ma60_pct",
    # ---- 相对强度 (alpha vs baseline index) --------------------------------
    "f_alpha_5d_pct",
    "f_alpha_20d_pct",
    "f_alpha_60d_pct",
    "f_alpha_leading",
    # ---- 涨停历史 (placeholder — populated in PR-3 once limit_list_d wires up)
    "f_limit_up_prior_count_60d",
    "f_limit_up_days_since_last",
    # ---- 板块 / 市场 (sector / market) -------------------------------------
    "f_sec_strength_score",
    # ---- 量能事件 (VA T-day rating, downgraded to auxiliary) ---------------
    "f_volume_event_score",
    # ---- 静态属性 (chip / size) --------------------------------------------
    "f_st_circ_mv_yi",
    "f_st_close_yuan",
    "f_st_pct_chg",
    "f_st_turnover_rate",
    "f_st_amount_yi",
    "f_st_listed_days",
]


_CLIP_FIELDS: frozenset[str] = frozenset(
    {
        "f_acc_price_change_pct",
        "f_probe_amplitude_pct",
        "f_probe_pct_chg",
        "f_wash_max_drawdown_pct",
        "f_wash_vol_shrink_ratio",
        "f_launch_close_to_probe_high_pct",
        "f_launch_cur_vol_ratio_5d",
        "f_launch_cur_vol_ratio_20d",
        "f_launch_relative_strength_20d",
        "f_vcp_atr_10d_pct",
        "f_vcp_bbw_20d",
        "f_vcp_bbw_compression_ratio",
        "f_mom_dist_to_120d_high_pct",
        "f_mom_dist_to_250d_high_pct",
        "f_mom_close_to_ma5_pct",
        "f_mom_close_to_ma10_pct",
        "f_mom_close_to_ma20_pct",
        "f_mom_close_to_ma60_pct",
        "f_alpha_5d_pct",
        "f_alpha_20d_pct",
        "f_alpha_60d_pct",
        "f_st_pct_chg",
    }
)


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
    sector_strength_data: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the (n_candidates × n_features) feature matrix.

    Parameters
    ----------
    candidate_rows
        Dictionaries produced by ``data.pack_candidate``. Each must carry
        ``ts_code``; other fields are best-effort (missing → NaN).
    sector_strength_data
        Optional mapping ``{ts_code: {industry_up_count: int, ...}}`` —
        reserved for PR-3 where the sector pipeline becomes available.

    Returns
    -------
    pd.DataFrame
        Index = ``ts_code``; columns = ``FEATURE_NAMES`` (exact order).
    """
    if not candidate_rows:
        return pd.DataFrame(columns=FEATURE_NAMES)

    sector_strength_data = sector_strength_data or {}

    rows: list[dict[str, float]] = []
    index: list[str] = []
    for row in candidate_rows:
        ts_code = str(row.get("ts_code", ""))
        index.append(ts_code)
        rows.append(_features_for_row(row))

    df = pd.DataFrame(rows, index=pd.Index(index, name="ts_code"))
    df = df.reindex(columns=FEATURE_NAMES)
    for col in _CLIP_FIELDS:
        if col in df.columns:
            df[col] = df[col].clip(lower=-_CLIP_BOUND, upper=_CLIP_BOUND)
    df = df.astype(float)
    return df


def feature_hash(feature_row: pd.Series | dict[str, Any]) -> str:
    """8-byte BLAKE2b digest of a single sample's feature vector (hex).

    Used by ``apw_lgb_predictions.feature_hash`` (PR-4) for audit ↔ replay
    reconciliation.
    """
    if isinstance(feature_row, pd.Series):
        items = [
            (str(k), feature_row[k]) for k in FEATURE_NAMES if k in feature_row.index
        ]
    else:
        items = [(k, feature_row.get(k)) for k in FEATURE_NAMES]
    payload: list[tuple[str, Any]] = []
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
# Per-row extraction
# ---------------------------------------------------------------------------


_ALPHA_LEADING_CODES = {"LEADING": 1.0, "NEUTRAL": 0.0, "LAGGING": -1.0}


def _features_for_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        # ---- 吸筹 ------------------------------------------------------------
        "f_acc_score": _f(row.get("accumulation_score")),
        "f_acc_days": _f(row.get("accumulation_days")),
        "f_acc_net_mf_yi": _f(row.get("accumulation_net_mf_yi")),
        "f_acc_price_change_pct": _f(row.get("accumulation_price_change_pct")),
        "f_acc_low_position_score": _f(row.get("low_position_score")),
        # ---- 试盘 ------------------------------------------------------------
        "f_probe_quality": _f(row.get("probe_quality_score")),
        "f_probe_vol_ratio_5d": _f(row.get("probe_volume_ratio_5d")),
        "f_probe_vol_ratio_20d": _f(row.get("probe_volume_ratio_20d")),
        "f_probe_vol_rank_pct_60d": _f(row.get("probe_volume_rank_pct_60d")),
        "f_probe_amount_ratio_20d": _f(row.get("probe_amount_ratio_20d")),
        "f_probe_turnover": _f(row.get("probe_turnover_rate")),
        "f_probe_amplitude_pct": _f(row.get("probe_amplitude_pct")),
        "f_probe_upper_shadow_ratio": _f(row.get("probe_upper_shadow_ratio")),
        "f_probe_body_ratio": _f(row.get("probe_body_ratio")),
        "f_probe_pct_chg": _f(row.get("probe_pct_chg")),
        "f_probe_moneyflow_net_yi": _f(row.get("probe_moneyflow_net_yi")),
        "f_probe_days_ago": _f(row.get("probe_days_ago")),
        # ---- 洗盘 ------------------------------------------------------------
        "f_wash_score": _f(row.get("washout_score")),
        "f_wash_days": _f(row.get("washout_days")),
        "f_wash_max_drawdown_pct": _f(row.get("post_probe_max_drawdown_pct")),
        "f_wash_vol_shrink_ratio": _f(row.get("post_probe_volume_shrink_ratio")),
        "f_wash_low_broken": _b(row.get("post_probe_low_broken")),
        "f_wash_ma20_broken": _b(row.get("post_probe_ma20_broken")),
        "f_wash_ma60_broken": _b(row.get("post_probe_ma60_broken")),
        "f_wash_moneyflow_net_yi": _f(row.get("post_probe_moneyflow_net_yi")),
        "f_wash_volatility_compression": _f(
            row.get("washout_volatility_compression")
        ),
        # ---- 启动 ------------------------------------------------------------
        "f_launch_setup_score": _f(row.get("launch_setup_score")),
        "f_launch_close_to_probe_high_pct": _f(row.get("close_to_probe_high_pct")),
        "f_launch_break_probe_high": _b(row.get("break_probe_high")),
        "f_launch_break_washout_high": _b(row.get("break_washout_high")),
        "f_launch_cur_vol_ratio_5d": _f(row.get("current_volume_ratio_5d")),
        "f_launch_cur_vol_ratio_20d": _f(row.get("current_volume_ratio_20d")),
        "f_launch_cur_moneyflow_net_yi": _f(row.get("current_moneyflow_net_yi")),
        "f_launch_above_ma5": _b(row.get("above_ma5")),
        "f_launch_above_ma10": _b(row.get("above_ma10")),
        "f_launch_above_ma20": _b(row.get("above_ma20")),
        "f_launch_relative_strength_20d": _f(row.get("relative_strength_20d")),
        # ---- VCP -------------------------------------------------------------
        "f_vcp_atr_10d_pct": _f(row.get("atr_10d_pct")),
        "f_vcp_atr_10d_quantile_60d": _f(row.get("atr_10d_quantile_in_60d")),
        "f_vcp_bbw_20d": _f(row.get("bbw_20d")),
        "f_vcp_bbw_compression_ratio": _f(row.get("bbw_compression_ratio")),
        # ---- 阻力位 -----------------------------------------------------------
        "f_mom_dist_to_120d_high_pct": _f(row.get("dist_to_120d_high_pct")),
        "f_mom_dist_to_250d_high_pct": _f(row.get("dist_to_250d_high_pct")),
        "f_mom_is_above_120d_high": _b(row.get("is_above_120d_high")),
        "f_mom_is_above_250d_high": _b(row.get("is_above_250d_high")),
        "f_mom_pos_in_120d_range": _f(row.get("pos_in_120d_range")),
        # ---- 均线距离 ---------------------------------------------------------
        "f_mom_close_to_ma5_pct": _f(row.get("close_to_ma5_pct")),
        "f_mom_close_to_ma10_pct": _f(row.get("close_to_ma10_pct")),
        "f_mom_close_to_ma20_pct": _f(row.get("close_to_ma20_pct")),
        "f_mom_close_to_ma60_pct": _f(row.get("close_to_ma60_pct")),
        # ---- alpha -----------------------------------------------------------
        "f_alpha_5d_pct": _f(row.get("alpha_5d_pct")),
        "f_alpha_20d_pct": _f(row.get("alpha_20d_pct")),
        "f_alpha_60d_pct": _f(row.get("alpha_60d_pct")),
        "f_alpha_leading": _alpha_leading_code(row.get("alpha_leading")),
        # ---- 涨停历史 (placeholder until PR-3 wires limit_list_d) -------------
        "f_limit_up_prior_count_60d": _f(
            row.get("prior_limit_up_count_60d"), default_for_none=0.0
        ),
        "f_limit_up_days_since_last": _f(row.get("days_since_last_limit_up")),
        # ---- 板块 / 市场 -----------------------------------------------------
        "f_sec_strength_score": _f(row.get("sector_strength_score")),
        # ---- 量能事件 ---------------------------------------------------------
        "f_volume_event_score": _f(row.get("volume_event_score")),
        # ---- 静态属性 ---------------------------------------------------------
        "f_st_circ_mv_yi": _f(row.get("circ_mv_yi")),
        "f_st_close_yuan": _f(row.get("close")),
        "f_st_pct_chg": _f(row.get("pct_chg")),
        "f_st_turnover_rate": _f(row.get("turnover_rate")),
        "f_st_amount_yi": _f(row.get("amount_yi")),
        "f_st_listed_days": _f(row.get("listed_days")),
    }


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _f(value: Any, *, default_for_none: float | None = None) -> float:
    if value is None or value == "":
        return float(default_for_none) if default_for_none is not None else _NA
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _NA
    if math.isnan(f) or math.isinf(f):
        return _NA
    return f


def _b(value: Any) -> float:
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


def _alpha_leading_code(value: Any) -> float:
    if value is None:
        return _NA
    return _ALPHA_LEADING_CODES.get(str(value).upper(), _NA)
