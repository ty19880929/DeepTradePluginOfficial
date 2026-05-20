"""v0.4.0 — FEATURE_NAMES / SCHEMA_VERSION lockfile.

These tests are deliberately picky: any reorder / rename / add / remove of
features must bump SCHEMA_VERSION and update both the byte snapshot and the
expected_v1 list in this file. Catching this here prevents a silent
train-serve skew (scorer would happily score on a re-ordered matrix and
silently produce wrong predictions).
"""

from __future__ import annotations

import pytest

from accumulation_probe_washout.lgb.features import (
    FEATURE_NAMES,
    SCHEMA_VERSION,
    LgbFeatureSchemaError,
    assert_columns,
    build_feature_frame,
)


def test_schema_version_locked_at_1():
    """Bumping SCHEMA_VERSION without updating this file forces an explicit
    review of every downstream consumer (scorer / dataset / registry)."""
    assert SCHEMA_VERSION == 1


def test_feature_names_lockfile():
    """Snapshot of the v1 feature set. If you intentionally change the list,
    update this expected vector AND bump SCHEMA_VERSION."""
    expected = [
        "f_acc_score", "f_acc_days", "f_acc_net_mf_yi",
        "f_acc_price_change_pct", "f_acc_low_position_score",
        "f_probe_quality", "f_probe_vol_ratio_5d", "f_probe_vol_ratio_20d",
        "f_probe_vol_rank_pct_60d", "f_probe_amount_ratio_20d",
        "f_probe_turnover", "f_probe_amplitude_pct",
        "f_probe_upper_shadow_ratio", "f_probe_body_ratio",
        "f_probe_pct_chg", "f_probe_moneyflow_net_yi", "f_probe_days_ago",
        "f_wash_score", "f_wash_days", "f_wash_max_drawdown_pct",
        "f_wash_vol_shrink_ratio", "f_wash_low_broken",
        "f_wash_ma20_broken", "f_wash_ma60_broken",
        "f_wash_moneyflow_net_yi", "f_wash_volatility_compression",
        "f_launch_setup_score", "f_launch_close_to_probe_high_pct",
        "f_launch_break_probe_high", "f_launch_break_washout_high",
        "f_launch_cur_vol_ratio_5d", "f_launch_cur_vol_ratio_20d",
        "f_launch_cur_moneyflow_net_yi",
        "f_launch_above_ma5", "f_launch_above_ma10", "f_launch_above_ma20",
        "f_launch_relative_strength_20d",
        "f_vcp_atr_10d_pct", "f_vcp_atr_10d_quantile_60d",
        "f_vcp_bbw_20d", "f_vcp_bbw_compression_ratio",
        "f_mom_dist_to_120d_high_pct", "f_mom_dist_to_250d_high_pct",
        "f_mom_is_above_120d_high", "f_mom_is_above_250d_high",
        "f_mom_pos_in_120d_range",
        "f_mom_close_to_ma5_pct", "f_mom_close_to_ma10_pct",
        "f_mom_close_to_ma20_pct", "f_mom_close_to_ma60_pct",
        "f_alpha_5d_pct", "f_alpha_20d_pct", "f_alpha_60d_pct",
        "f_alpha_leading",
        "f_limit_up_prior_count_60d", "f_limit_up_days_since_last",
        "f_sec_strength_score",
        "f_volume_event_score",
        "f_st_circ_mv_yi", "f_st_close_yuan", "f_st_pct_chg",
        "f_st_turnover_rate", "f_st_amount_yi", "f_st_listed_days",
    ]
    assert FEATURE_NAMES == expected


def test_feature_names_uniqueness():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_build_feature_frame_empty_input_returns_empty_frame_with_columns():
    df = build_feature_frame(candidate_rows=[])
    assert list(df.columns) == FEATURE_NAMES
    assert len(df) == 0


def test_assert_columns_accepts_canonical_frame():
    df = build_feature_frame(candidate_rows=[{"ts_code": "600000.SH"}])
    assert_columns(df)  # does not raise


def test_assert_columns_rejects_reorder():
    import pandas as pd
    df = build_feature_frame(candidate_rows=[{"ts_code": "600000.SH"}])
    reordered = df[list(reversed(FEATURE_NAMES))]
    with pytest.raises(LgbFeatureSchemaError):
        assert_columns(reordered)
