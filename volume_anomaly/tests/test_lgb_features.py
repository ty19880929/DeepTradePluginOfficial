"""Tests for ``volume_anomaly.lgb.features`` — design §4 / iteration PR-1.1."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from volume_anomaly.lgb.features import (
    FEATURE_NAMES,
    SCHEMA_VERSION,
    LgbFeatureSchemaError,
    assert_columns,
    build_feature_frame,
    feature_hash,
    feature_missing,
)


def _candidate_row(ts_code: str = "600000.SH", **overrides):
    base = {
        "ts_code": ts_code,
        "name": "TestCo",
        "industry": "银行",
        "tracked_since": "20260101",
        "tracked_days": 3,
        "anomaly_day": "20260102",
        "anomaly_pct_chg": 4.5,
        "anomaly_body_ratio": 0.72,
        "anomaly_turnover_rate": 8.4,
        "anomaly_vol_ratio_5d": 2.1,
        "last_close": 12.30,
        "ma5": 12.10,
        "ma10": 12.00,
        "ma20": 11.50,
        "ma60": 10.80,
        "above_ma20": True,
        "above_ma60": True,
        "high_60d": 12.80,
        "low_60d": 10.20,
        "range_pct_60d": 25.49,
        "pct_chg_60d": 9.40,
        "atr_10d_pct": 2.1,
        "atr_10d_quantile_in_60d": 0.42,
        "bbw_20d": 0.085,
        "bbw_compression_ratio": 0.78,
        "alpha_5d_pct": 1.3,
        "alpha_20d_pct": 6.0,
        "alpha_60d_pct": 3.8,
        "rel_strength_label": "leading",
        "high_120d": 12.90,
        "high_250d": 13.50,
        "low_120d": 10.00,
        "dist_to_120d_high_pct": -4.65,
        "dist_to_250d_high_pct": -8.89,
        "is_above_120d_high": False,
        "is_above_250d_high": False,
        "pos_in_120d_range": 0.793,
        "base_days": 12,
        "base_max_drawdown_pct": 5.1,
        "base_avg_vol": 120000.0,
        "base_vol_shrink_ratio": 0.92,
        "days_since_last_limit_up": 14,
        "prior_limit_up_count_60d": 1,
        "turnover_rate_t": 7.2,
        "volume_ratio_t": 1.8,
        "pe_t": 18.5,
        "pb_t": 1.6,
        "circ_mv_yi": 85.0,
        "total_mv_yi": 120.0,
        "moneyflow_5d_summary": {
            "cum_net_mf_yi": 0.42,
            "cum_elg_plus_lg_buy_yi": 1.85,
            "net_mf_trend": "rising",
            "rows_used": 5,
        },
    }
    base.update(overrides)
    return base


def test_feature_names_length_matches_schema_version_contract():
    # Each declared feature must be a non-empty string and unique.
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert all(isinstance(n, str) and n for n in FEATURE_NAMES)
    # Schema version 1 anchors the v0.7.0 wire contract.
    assert SCHEMA_VERSION == 1
    # Sanity: must be in the ~50-feature band the design promised.
    assert 40 <= len(FEATURE_NAMES) <= 60


def test_build_feature_frame_returns_expected_shape_and_dtype():
    rows = [_candidate_row("600000.SH"), _candidate_row("000001.SZ")]
    df = build_feature_frame(
        candidate_rows=rows,
        market_summary={"limit_up_count": 73},
        sector_strength_data={
            "top_sectors": [
                {"name": "银行", "count": 4},
                {"name": "煤炭", "count": 2},
            ],
        },
        sector_strength_source="limit_cpt_list",
    )
    assert list(df.columns) == FEATURE_NAMES
    assert df.shape == (2, len(FEATURE_NAMES))
    assert df.index.tolist() == ["600000.SH", "000001.SZ"]
    assert df.dtypes.unique().tolist() == [pd.api.types.pandas_dtype("float64")]
    # The "industry up count" lookup must resolve via the 'name' key.
    assert df.loc["600000.SH", "f_sec_today_industry_up_count"] == 4.0
    # Sector-source rank = 1 (limit_cpt_list).
    assert df.loc["600000.SH", "f_sec_strength_source_rank"] == 1.0
    # Market summary rolls through unchanged.
    assert df.loc["600000.SH", "f_mkt_total_limit_up"] == 73.0


def test_build_feature_frame_empty_input_returns_empty_frame_with_schema():
    df = build_feature_frame(candidate_rows=[])
    assert list(df.columns) == FEATURE_NAMES
    assert df.empty


def test_build_feature_frame_missing_fields_yield_nan_not_zero():
    row = _candidate_row()
    # Strip all factor fields; only ts_code remains.
    minimal = {"ts_code": row["ts_code"]}
    df = build_feature_frame(candidate_rows=[minimal])
    # Every non-default-0 numeric should be NaN.
    nan_required = [
        "f_vol_anomaly_pct_chg",
        "f_mom_close_to_ma5_pct",
        "f_alpha_5d_pct",
        "f_mom_above_ma20",
        "f_alpha_leading",
        "f_mf_trend_rising",
    ]
    for col in nan_required:
        v = df.loc[row["ts_code"], col]
        assert math.isnan(v), f"{col} should be NaN when source field is missing; got {v!r}"
    # Counters explicitly default to 0 (business-meaning of "no data").
    assert df.loc[row["ts_code"], "f_mf_rows_used"] == 0.0
    assert df.loc[row["ts_code"], "f_wash_prior_limit_up_count_60d"] == 0.0
    assert df.loc[row["ts_code"], "f_st_tracked_days"] == 0.0


def test_close_to_ma_pct_matches_manual_computation():
    row = _candidate_row(last_close=11.0, ma5=10.0, ma10=12.0, ma20=10.0, ma60=10.0)
    df = build_feature_frame(candidate_rows=[row])
    assert df.loc[row["ts_code"], "f_mom_close_to_ma5_pct"] == pytest.approx(10.0)
    assert df.loc[row["ts_code"], "f_mom_close_to_ma10_pct"] == pytest.approx(
        (11.0 - 12.0) / 12.0 * 100.0
    )


def test_close_to_ma_pct_handles_zero_denominator():
    row = _candidate_row(last_close=11.0, ma5=0.0)
    df = build_feature_frame(candidate_rows=[row])
    assert math.isnan(df.loc[row["ts_code"], "f_mom_close_to_ma5_pct"])


def test_bool_fields_encode_truefalse_to_one_zero():
    row = _candidate_row(above_ma20=False, above_ma60=True)
    df = build_feature_frame(candidate_rows=[row])
    assert df.loc[row["ts_code"], "f_mom_above_ma20"] == 0.0
    assert df.loc[row["ts_code"], "f_mom_above_ma60"] == 1.0


def test_alpha_leading_encoding():
    cases = {
        "leading": 1.0,
        "in_line": 0.0,
        "lagging": -1.0,
    }
    for label, want in cases.items():
        row = _candidate_row(rel_strength_label=label)
        df = build_feature_frame(candidate_rows=[row])
        assert df.loc[row["ts_code"], "f_alpha_leading"] == want
    # None / unknown → NaN.
    row = _candidate_row(rel_strength_label=None)
    df = build_feature_frame(candidate_rows=[row])
    assert math.isnan(df.loc[row["ts_code"], "f_alpha_leading"])


def test_mf_trend_encoding_and_rows_used():
    for trend, want in {"rising": 1.0, "flat": 0.0, "falling": -1.0}.items():
        row = _candidate_row()
        row["moneyflow_5d_summary"] = {"net_mf_trend": trend, "rows_used": 5}
        df = build_feature_frame(candidate_rows=[row])
        assert df.loc[row["ts_code"], "f_mf_trend_rising"] == want
        assert df.loc[row["ts_code"], "f_mf_rows_used"] == 5.0
    # Missing moneyflow → trend NaN, rows_used 0.
    row = _candidate_row()
    row["moneyflow_5d_summary"] = {"rows_used": 0}
    df = build_feature_frame(candidate_rows=[row])
    assert math.isnan(df.loc[row["ts_code"], "f_mf_trend_rising"])
    assert df.loc[row["ts_code"], "f_mf_rows_used"] == 0.0


def test_sector_source_rank_matrix():
    for src, want in {
        "limit_cpt_list": 1.0,
        "lu_desc_aggregation": 2.0,
        "industry_fallback": 3.0,
    }.items():
        row = _candidate_row()
        df = build_feature_frame(
            candidate_rows=[row],
            sector_strength_source=src,
        )
        assert df.loc[row["ts_code"], "f_sec_strength_source_rank"] == want
    # None / unknown → NaN.
    row = _candidate_row()
    df = build_feature_frame(candidate_rows=[row])
    assert math.isnan(df.loc[row["ts_code"], "f_sec_strength_source_rank"])


def test_industry_up_count_resolves_for_fallback_payload():
    rows = [_candidate_row("600000.SH", industry="银行")]
    df = build_feature_frame(
        candidate_rows=rows,
        sector_strength_data={
            "top_sectors": [{"sector": "银行", "watchlist_count": 3}],
        },
        sector_strength_source="industry_fallback",
    )
    assert df.loc["600000.SH", "f_sec_today_industry_up_count"] == 3.0


def test_clip_protects_against_extreme_ratios():
    # Manually engineer an extreme close/ma5 differential.
    row = _candidate_row(last_close=1e8, ma5=1.0)
    df = build_feature_frame(candidate_rows=[row])
    # The raw ratio would be 1e10 %; must be clipped.
    assert df.loc[row["ts_code"], "f_mom_close_to_ma5_pct"] == 500.0


def test_assert_columns_accepts_matching_and_rejects_otherwise():
    df = pd.DataFrame(columns=FEATURE_NAMES)
    assert_columns(df)  # no raise
    bad = df.drop(columns=[FEATURE_NAMES[0]])
    with pytest.raises(LgbFeatureSchemaError):
        assert_columns(bad)


def test_feature_hash_is_stable_for_identical_row():
    row = _candidate_row()
    df = build_feature_frame(candidate_rows=[row])
    h1 = feature_hash(df.loc[row["ts_code"]])
    h2 = feature_hash(df.loc[row["ts_code"]])
    assert h1 == h2
    assert len(h1) == 16  # 8 bytes hex


def test_feature_hash_changes_when_value_changes():
    row_a = _candidate_row(anomaly_pct_chg=4.5)
    row_b = _candidate_row(anomaly_pct_chg=4.6)
    df_a = build_feature_frame(candidate_rows=[row_a])
    df_b = build_feature_frame(candidate_rows=[row_b])
    assert feature_hash(df_a.iloc[0]) != feature_hash(df_b.iloc[0])


def test_build_feature_frame_is_idempotent_under_repeated_calls():
    rows = [_candidate_row("600000.SH"), _candidate_row("000001.SZ")]
    df1 = build_feature_frame(candidate_rows=rows)
    df2 = build_feature_frame(candidate_rows=rows)
    pd.testing.assert_frame_equal(df1, df2)


def test_feature_missing_lists_nan_columns():
    row = _candidate_row()
    df = build_feature_frame(candidate_rows=[row])
    miss = feature_missing(df.iloc[0])
    # Several intentionally-unfilled features must show up.
    assert "f_vol_ratio_60d_t" in miss
    assert "f_vol_amount_yi_t" in miss
    assert "f_mkt_yesterday_failure_rate" in miss
    # And populated ones must not.
    assert "f_vol_anomaly_pct_chg" not in miss


def test_inf_and_nan_inputs_become_nan_outputs():
    row = _candidate_row(anomaly_pct_chg=float("inf"))
    df = build_feature_frame(candidate_rows=[row])
    assert math.isnan(df.loc[row["ts_code"], "f_vol_anomaly_pct_chg"])
