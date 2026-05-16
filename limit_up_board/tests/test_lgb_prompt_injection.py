"""PR-2.3 — 强势初筛 / 连板预测 system prompts contain the LGB §8.1 / §8.2 paragraph.

Asserts:
* Default ``build_screening_system()`` mentions ``lgb_score`` + the numeric floor.
* ``build_screening_system(lgb_min_score_floor=None)`` drops the numeric line but
  keeps the rest of the LGB guidance.
* Custom floor (e.g. 42.5) is correctly interpolated.
* Same set for 连板预测.
* The user prompt itself doesn't need to change — ``screening_user_prompt`` /
  ``prediction_user_prompt`` just dump the candidates dict; we verify that an
  ``lgb_score`` key on a candidate naturally lands in the rendered prompt.
"""

from __future__ import annotations

import json

import pytest

from limit_up_board.prompts import (
    PREDICTION_SYSTEM,
    SCREENING_SYSTEM,
    build_prediction_system,
    build_screening_system,
    prediction_user_prompt,
    screening_user_prompt,
)


# ---------------------------------------------------------------------------
# Constants reflect the LubConfig default (30.0)
# ---------------------------------------------------------------------------


def test_screening_system_default_contains_lgb_block_and_default_floor() -> None:
    assert "lgb_score" in SCREENING_SYSTEM
    assert "lgb_decile" in SCREENING_SYSTEM
    assert "lgb_score < 30" in SCREENING_SYSTEM
    # Discipline checks the model still hard rules apply
    assert "硬性纪律" in SCREENING_SYSTEM


def test_prediction_system_default_contains_lgb_block_and_default_floor() -> None:
    assert "lgb_score" in PREDICTION_SYSTEM
    assert "lgb_decile" in PREDICTION_SYSTEM
    assert "lgb_score < 30" in PREDICTION_SYSTEM
    assert "硬性纪律" in PREDICTION_SYSTEM


# ---------------------------------------------------------------------------
# Builder: custom floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("floor", [10.0, 25, 42.5, 50.0, 70.5])
def test_build_screening_system_custom_floor(floor: float) -> None:
    out = build_screening_system(lgb_min_score_floor=floor)
    # The custom value should be present and no other floor.
    assert f"lgb_score < {floor:g}" in out


@pytest.mark.parametrize("floor", [10.0, 25, 42.5, 50.0, 70.5])
def test_build_prediction_system_custom_floor(floor: float) -> None:
    out = build_prediction_system(lgb_min_score_floor=floor)
    assert f"lgb_score < {floor:g}" in out


# ---------------------------------------------------------------------------
# Builder: floor=None
# ---------------------------------------------------------------------------


def test_build_screening_system_no_floor_drops_threshold() -> None:
    out = build_screening_system(lgb_min_score_floor=None)
    assert "lgb_score < " not in out
    # The rest of the LGB block survives
    assert "lgb_score" in out
    assert "lgb_decile" in out


def test_build_prediction_system_no_floor_drops_threshold() -> None:
    out = build_prediction_system(lgb_min_score_floor=None)
    assert "lgb_score < " not in out
    assert "lgb_score" in out
    assert "lgb_decile" in out


# ---------------------------------------------------------------------------
# User prompts: candidate dict with lgb_score naturally shows up
# ---------------------------------------------------------------------------


def test_screening_user_prompt_includes_candidate_lgb_score() -> None:
    candidates = [
        {
            "candidate_id": "600519.SH",
            "ts_code": "600519.SH",
            "name": "茅台",
            "lgb_score": 73.4,
            "lgb_decile": 8,
            "lgb_feature_missing": [],
        }
    ]
    text = screening_user_prompt(
        trade_date="20260530",
        batch_no=1,
        batch_total=1,
        candidates=candidates,
        market_summary={"limit_up_count": 1},
        sector_strength_source="limit_cpt_list",
        sector_strength_data={"top_sectors": []},
        data_unavailable=[],
    )
    # JSON dump preserves the fields verbatim.
    assert '"lgb_score": 73.4' in text
    assert '"lgb_decile": 8' in text


def test_prediction_user_prompt_includes_candidate_lgb_score() -> None:
    candidates = [
        {
            "candidate_id": "600519.SH",
            "ts_code": "600519.SH",
            "name": "茅台",
            "lgb_score": 41.0,
            "lgb_decile": 4,
            "lgb_feature_missing": ["f_lhb_inst_count"],
        }
    ]
    text = prediction_user_prompt(
        trade_date="20260530",
        next_trade_date="20260531",
        candidates=candidates,
        market_context={"limit_up_count": 1},
        sector_strength_source="limit_cpt_list",
        sector_strength_data={"top_sectors": []},
        data_unavailable=[],
    )
    assert '"lgb_score": 41' in text  # JSON drops trailing .0 sometimes; loose match
    assert '"lgb_decile": 4' in text


def test_user_prompts_omit_lgb_when_field_absent() -> None:
    """When LGB disabled and not injected, prompt JSON dump simply lacks the key."""
    candidates = [
        {
            "candidate_id": "600519.SH",
            "ts_code": "600519.SH",
            "name": "茅台",
        }
    ]
    text = screening_user_prompt(
        trade_date="20260530",
        batch_no=1,
        batch_total=1,
        candidates=candidates,
        market_summary={},
        sector_strength_source="limit_cpt_list",
        sector_strength_data={},
        data_unavailable=[],
    )
    assert "lgb_score" not in text
    assert json.dumps(candidates, ensure_ascii=False, indent=2) in text
