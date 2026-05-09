"""volume-anomaly v0.6.0 — VADimensionScores + 整体 schema 校验。"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from volume_anomaly.prompts_examples import (
    VA_TREND_FEWSHOT,
)
from volume_anomaly.schemas import (
    VADimensionScores,
    VATrendCandidate,
)


def _candidate_payload(**overrides) -> dict:
    base = {
        "candidate_id": "000001.SZ",
        "ts_code": "000001.SZ",
        "name": "测试",
        "rank": 1,
        "launch_score": 65.0,
        "confidence": "high",
        "prediction": "imminent_launch",
        "pattern": "breakout",
        "washout_quality": "sufficient",
        "rationale": "测试",
        "dimension_scores": {
            "washout": 70,
            "pattern": 70,
            "capital": 70,
            "sector": 70,
            "historical": 60,
            "risk": 25,
        },
        "key_evidence": [
            {"field": "base_days", "value": 24, "unit": "日", "interpretation": "整理周期较长"}
        ],
        "next_session_watch": ["开盘"],
        "invalidation_triggers": ["跌破 MA10"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# VADimensionScores boundary validation
# ---------------------------------------------------------------------------


class TestDimensionScoresValidation:
    def test_valid_payload(self) -> None:
        m = VADimensionScores(
            washout=80, pattern=70, capital=60, sector=70, historical=50, risk=20
        )
        assert m.washout == 80
        assert m.risk == 20

    def test_score_above_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VADimensionScores(
                washout=101, pattern=70, capital=60, sector=70, historical=50, risk=20
            )

    def test_score_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VADimensionScores(
                washout=-1, pattern=70, capital=60, sector=70, historical=50, risk=20
            )

    def test_missing_dimension_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VADimensionScores(washout=70, pattern=70)  # type: ignore[call-arg]

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VADimensionScores(
                washout=70, pattern=70, capital=70, sector=70,
                historical=50, risk=20, liquidity=80,  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# VATrendCandidate — dimension_scores is required (F8)
# ---------------------------------------------------------------------------


class TestVATrendCandidateRequiresDimensionScores:
    def test_full_payload_passes(self) -> None:
        c = VATrendCandidate.model_validate(_candidate_payload())
        assert c.dimension_scores.washout == 70

    def test_missing_dimension_scores_rejected(self) -> None:
        payload = _candidate_payload()
        del payload["dimension_scores"]
        with pytest.raises(ValidationError):
            VATrendCandidate.model_validate(payload)

    def test_partial_dimension_scores_rejected(self) -> None:
        payload = _candidate_payload(
            dimension_scores={"washout": 70, "pattern": 70, "capital": 70}
        )
        with pytest.raises(ValidationError):
            VATrendCandidate.model_validate(payload)


# ---------------------------------------------------------------------------
# Few-shot examples include valid dimension_scores blocks
# ---------------------------------------------------------------------------


class TestFewShotDimensionScores:
    def test_examples_contain_dimension_scores(self) -> None:
        # Should appear in BOTH example A and example B
        assert VA_TREND_FEWSHOT.count('"dimension_scores":') == 2

    def test_dimension_score_values_in_range(self) -> None:
        # Pull out all integer field values inside dimension_scores blocks via
        # a forgiving regex; assert they fall within [0, 100].
        # Pattern: matches each `"<dim>": NN` for the six dim names.
        dim_names = ["washout", "pattern", "capital", "sector", "historical", "risk"]
        for name in dim_names:
            for m in re.finditer(rf'"{name}":\s*(\d+)', VA_TREND_FEWSHOT):
                v = int(m.group(1))
                assert 0 <= v <= 100, f"out-of-range {name}={v} in few-shot"
