"""Pydantic schema validation — T3.1."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from accumulation_probe_washout.schemas import (
    APWDimensionScores,
    APWEvidenceItem,
    APWTrendCandidate,
    APWTrendResponse,
    EvidenceFieldError,
    INPUT_FIELD_WHITELIST,
    check_response_against_inputs,
)


def _ok_candidate(**overrides) -> dict:
    base = {
        "candidate_id": "20260515_600000.SH",
        "ts_code": "600000.SH",
        "name": "示例股份",
        "rank": 1,
        "launch_score": 75.0,
        "confidence": "medium",
        "prediction": "launch_ready",
        "main_pattern": "probe_washout_breakout",
        "phase": "launch_ready",
        "dimension_scores": {
            "accumulation": 70, "probe": 80, "washout": 75,
            "launch_timing": 80, "capital_confirmation": 60, "risk": 30,
        },
        "rationale": "测试理由",
        "key_evidence": [
            {"field": "probe_quality_score", "value": 84, "unit": "score",
             "interpretation": "高质量试盘日"}
        ],
        "next_session_watch": ["突破试盘高点"],
        "invalidation_triggers": ["跌破试盘日 low"],
    }
    base.update(overrides)
    return base


def _ok_response(candidates=None) -> dict:
    return {
        "stage": "accumulation_probe_washout_analysis",
        "trade_date": "20260515",
        "next_trade_date": "20260516",
        "batch_no": 1,
        "batch_total": 1,
        "market_context_summary": "市场窄幅震荡",
        "risk_disclaimer": "仅作辅助判断，不构成交易建议",
        "candidates": candidates or [_ok_candidate()],
    }


class TestSchemaValidation:
    def test_valid_minimal_response(self) -> None:
        APWTrendResponse.model_validate(_ok_response())

    def test_extra_field_rejected(self) -> None:
        cand = _ok_candidate(extra_key=1)
        with pytest.raises(ValidationError):
            APWTrendResponse.model_validate(_ok_response([cand]))

    def test_rank_must_be_dense(self) -> None:
        c1 = _ok_candidate(candidate_id="20260515_600000.SH", rank=1)
        c2 = _ok_candidate(candidate_id="20260515_600001.SH", ts_code="600001.SH", rank=3)
        with pytest.raises(ValidationError):
            APWTrendResponse.model_validate(_ok_response([c1, c2]))

    def test_prediction_enum_enforced(self) -> None:
        with pytest.raises(ValidationError):
            APWTrendResponse.model_validate(_ok_response([_ok_candidate(prediction="invalid")]))

    def test_dimension_scores_clamped(self) -> None:
        with pytest.raises(ValidationError):
            APWTrendResponse.model_validate(_ok_response([
                _ok_candidate(dimension_scores={
                    "accumulation": 150, "probe": 80, "washout": 75,
                    "launch_timing": 80, "capital_confirmation": 60, "risk": 30,
                })
            ]))


class TestCallerValidators:
    def test_id_set_mismatch_raises(self) -> None:
        resp = APWTrendResponse.model_validate(_ok_response())
        with pytest.raises(ValueError):
            check_response_against_inputs(
                resp, input_ids={"some_other_id"}
            )

    def test_evidence_field_outside_whitelist_raises(self) -> None:
        cand = _ok_candidate(key_evidence=[{
            "field": "not_a_real_field", "value": 1, "unit": "u",
            "interpretation": "x"
        }])
        resp = APWTrendResponse.model_validate(_ok_response([cand]))
        with pytest.raises(EvidenceFieldError):
            check_response_against_inputs(resp, {cand["candidate_id"]})

    def test_evidence_field_from_whitelist_ok(self) -> None:
        resp = APWTrendResponse.model_validate(_ok_response())
        check_response_against_inputs(
            resp, {resp.candidates[0].candidate_id}
        )

    def test_whitelist_contains_expected_fields(self) -> None:
        # Sanity check — every field name listed in the design spec §8 must be
        # callable as evidence.
        for f in ("accumulation_score", "probe_quality_score", "washout_score",
                  "launch_setup_score", "post_probe_low_broken", "above_ma5"):
            assert f in INPUT_FIELD_WHITELIST
