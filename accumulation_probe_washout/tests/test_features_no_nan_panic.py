"""v0.4.0 — build_feature_frame must never raise on missing fields.

The LightGBM booster uses native NaN routing, so the contract is "absent
field → NaN", not "absent field → 0 or KeyError". Tested against a
deliberately sparse candidate dict.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from accumulation_probe_washout.lgb.features import (
    FEATURE_NAMES,
    build_feature_frame,
    feature_hash,
    feature_missing,
)


def test_minimum_candidate_yields_all_nan_features():
    df = build_feature_frame(candidate_rows=[{"ts_code": "600000.SH"}])
    assert len(df) == 1
    # Every column must be NaN (no field populated beyond ts_code).
    row = df.iloc[0]
    nan_count = int(row.isna().sum())
    # f_limit_up_prior_count_60d defaults to 0 when None — by design, so we
    # tolerate exactly that one non-NaN.
    assert nan_count == len(FEATURE_NAMES) - 1
    assert row["f_limit_up_prior_count_60d"] == 0.0


def test_partial_candidate_populates_known_fields_only():
    cand = {
        "ts_code": "600000.SH",
        "accumulation_score": 72.0,
        "probe_quality_score": 88.0,
        "atr_10d_pct": 2.5,
        "alpha_leading": "LEADING",
    }
    df = build_feature_frame(candidate_rows=[cand])
    row = df.iloc[0]
    assert row["f_acc_score"] == 72.0
    assert row["f_probe_quality"] == 88.0
    assert row["f_vcp_atr_10d_pct"] == 2.5
    assert row["f_alpha_leading"] == 1.0  # LEADING → 1.0
    # Fields we didn't populate must be NaN.
    assert math.isnan(row["f_launch_setup_score"])


def test_alpha_leading_codes():
    df = build_feature_frame(
        candidate_rows=[
            {"ts_code": "A", "alpha_leading": "LEADING"},
            {"ts_code": "B", "alpha_leading": "NEUTRAL"},
            {"ts_code": "C", "alpha_leading": "LAGGING"},
            {"ts_code": "D", "alpha_leading": "junk"},  # unknown → NaN
        ]
    )
    assert df.loc["A", "f_alpha_leading"] == 1.0
    assert df.loc["B", "f_alpha_leading"] == 0.0
    assert df.loc["C", "f_alpha_leading"] == -1.0
    assert math.isnan(df.loc["D", "f_alpha_leading"])


def test_clip_fields_bounded():
    cand = {
        "ts_code": "X",
        "accumulation_price_change_pct": 10_000.0,
        "alpha_60d_pct": -10_000.0,
    }
    df = build_feature_frame(candidate_rows=[cand])
    assert df.loc["X", "f_acc_price_change_pct"] == 500.0
    assert df.loc["X", "f_alpha_60d_pct"] == -500.0


def test_inf_and_nan_inputs_become_nan():
    cand = {
        "ts_code": "X",
        "accumulation_score": float("inf"),
        "probe_quality_score": float("nan"),
        "launch_setup_score": "not a number",
    }
    df = build_feature_frame(candidate_rows=[cand])
    row = df.iloc[0]
    assert math.isnan(row["f_acc_score"])
    assert math.isnan(row["f_probe_quality"])
    assert math.isnan(row["f_launch_setup_score"])


def test_feature_hash_stable_under_field_reorder():
    """Hash depends on FEATURE_NAMES order — not on the dict insertion order."""
    cand_a = {"ts_code": "X", "accumulation_score": 70, "probe_quality_score": 80}
    cand_b = {"ts_code": "X", "probe_quality_score": 80, "accumulation_score": 70}
    df_a = build_feature_frame(candidate_rows=[cand_a])
    df_b = build_feature_frame(candidate_rows=[cand_b])
    assert feature_hash(df_a.iloc[0]) == feature_hash(df_b.iloc[0])


def test_feature_missing_lists_nan_columns():
    df = build_feature_frame(candidate_rows=[{"ts_code": "X"}])
    miss = feature_missing(df.iloc[0])
    # Every feature except f_limit_up_prior_count_60d (default 0) is missing.
    assert "f_limit_up_prior_count_60d" not in miss
    assert "f_acc_score" in miss
    assert "f_volume_event_score" in miss


def test_multi_row_frame_index_is_ts_code():
    df = build_feature_frame(
        candidate_rows=[
            {"ts_code": "600000.SH", "accumulation_score": 70},
            {"ts_code": "600001.SH", "accumulation_score": 60},
        ]
    )
    assert list(df.index) == ["600000.SH", "600001.SH"]
    assert df.index.name == "ts_code"
    assert df.loc["600000.SH", "f_acc_score"] == 70.0
    assert df.loc["600001.SH", "f_acc_score"] == 60.0
