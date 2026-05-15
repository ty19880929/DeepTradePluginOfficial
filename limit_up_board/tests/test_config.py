"""Tests for ``limit_up_board.config.validate_config`` (P3-3).

One test per validation rule. ``LubConfig()`` defaults are valid by
construction; each test breaches one field and asserts ValueError contains
the offending field name (so CLI users get an actionable message).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from limit_up_board.config import LubConfig, validate_config


def test_defaults_pass() -> None:
    """Sanity: the dataclass defaults are themselves valid."""
    validate_config(LubConfig())


def test_min_float_mv_yi_must_be_non_negative() -> None:
    cfg = replace(LubConfig(), min_float_mv_yi=-1.0)
    with pytest.raises(ValueError, match="min_float_mv_yi"):
        validate_config(cfg)


def test_min_must_be_strictly_less_than_max() -> None:
    cfg = replace(LubConfig(), min_float_mv_yi=100.0, max_float_mv_yi=100.0)
    with pytest.raises(ValueError, match="max_float_mv_yi"):
        validate_config(cfg)


def test_max_close_yuan_must_be_positive() -> None:
    cfg = replace(LubConfig(), max_close_yuan=0.0)
    with pytest.raises(ValueError, match="max_close_yuan"):
        validate_config(cfg)
    cfg = replace(LubConfig(), max_close_yuan=-5.0)
    with pytest.raises(ValueError, match="max_close_yuan"):
        validate_config(cfg)


def test_lgb_label_threshold_must_be_within_bounds() -> None:
    # Lower bound exclusive (0)
    cfg = replace(LubConfig(), lgb_label_threshold_pct=0.0)
    with pytest.raises(ValueError, match="lgb_label_threshold_pct"):
        validate_config(cfg)
    # Upper bound exclusive (20)
    cfg = replace(LubConfig(), lgb_label_threshold_pct=20.0)
    with pytest.raises(ValueError, match="lgb_label_threshold_pct"):
        validate_config(cfg)
    # Beyond upper bound
    cfg = replace(LubConfig(), lgb_label_threshold_pct=25.0)
    with pytest.raises(ValueError, match="lgb_label_threshold_pct"):
        validate_config(cfg)


def test_lgb_min_score_floor_accepts_none() -> None:
    """None means 'do not surface a floor to the prompt' — valid."""
    validate_config(replace(LubConfig(), lgb_min_score_floor=None))


def test_lgb_min_score_floor_range() -> None:
    cfg = replace(LubConfig(), lgb_min_score_floor=-1.0)
    with pytest.raises(ValueError, match="lgb_min_score_floor"):
        validate_config(cfg)
    cfg = replace(LubConfig(), lgb_min_score_floor=101.0)
    with pytest.raises(ValueError, match="lgb_min_score_floor"):
        validate_config(cfg)


def test_lgb_train_lookback_days_minimum() -> None:
    cfg = replace(LubConfig(), lgb_train_lookback_days=29)
    with pytest.raises(ValueError, match="lgb_train_lookback_days"):
        validate_config(cfg)


def test_lgb_train_min_samples_minimum() -> None:
    cfg = replace(LubConfig(), lgb_train_min_samples=99)
    with pytest.raises(ValueError, match="lgb_train_min_samples"):
        validate_config(cfg)


def test_lgb_max_models_to_keep_minimum() -> None:
    cfg = replace(LubConfig(), lgb_max_models_to_keep=0)
    with pytest.raises(ValueError, match="lgb_max_models_to_keep"):
        validate_config(cfg)
