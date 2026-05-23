"""P1-2：``ContinuationCandidate._apply_empty_array_policy`` 按
``empty_array_policy`` ContextVar 在三种行为间切换。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from limit_up_board.schemas import (
    ContinuationCandidate,
    EvidenceItem,
    RevisedContinuationCandidate,
    apply_empty_array_policy,
)


def _evidence() -> list[dict]:
    return [
        {"field": "fd_amount_yi", "value": 1.0, "unit": "亿", "interpretation": "封单 1 亿"}
    ]


def _base_payload(**override) -> dict:
    data = {
        "candidate_id": "C1",
        "ts_code": "000001.SZ",
        "name": "平安银行",
        "rank": 1,
        "continuation_score": 60.0,
        "confidence": "medium",
        "prediction": "watchlist",
        "rationale": "stub",
        "key_evidence": _evidence(),
        "next_day_watch_points": ["盯紧 9:30 跳空"],
        "failure_triggers": ["跌破 5 日均线"],
    }
    data.update(override)
    return data


# ---------------------------------------------------------------------------
# Default policy is "repair"
# ---------------------------------------------------------------------------


def test_default_policy_is_repair_and_rejects_empty_watch_points() -> None:
    payload = _base_payload(next_day_watch_points=[])
    with pytest.raises(ValidationError, match="empty array"):
        ContinuationCandidate.model_validate(payload)


def test_default_policy_is_repair_and_rejects_empty_failure_triggers() -> None:
    payload = _base_payload(failure_triggers=[])
    with pytest.raises(ValidationError, match="empty array"):
        ContinuationCandidate.model_validate(payload)


def test_default_policy_rejects_both_empty_in_one_message() -> None:
    payload = _base_payload(next_day_watch_points=[], failure_triggers=[])
    with pytest.raises(ValidationError) as exc_info:
        ContinuationCandidate.model_validate(payload)
    msg = str(exc_info.value)
    assert "next_day_watch_points" in msg
    assert "failure_triggers" in msg


# ---------------------------------------------------------------------------
# "degraded" — 占位符 + degraded_fields 标注
# ---------------------------------------------------------------------------


def test_degraded_policy_fills_placeholder_and_records_field_name() -> None:
    payload = _base_payload(next_day_watch_points=[])
    with apply_empty_array_policy("degraded"):
        obj = ContinuationCandidate.model_validate(payload)
    assert obj.next_day_watch_points == ["（LLM 未给出观察点，人工复核）"]
    assert obj.failure_triggers == ["跌破 5 日均线"]
    assert obj.degraded_fields == ["next_day_watch_points"]


def test_degraded_policy_records_both_when_both_empty() -> None:
    payload = _base_payload(next_day_watch_points=[], failure_triggers=[])
    with apply_empty_array_policy("degraded"):
        obj = ContinuationCandidate.model_validate(payload)
    assert obj.next_day_watch_points == ["（LLM 未给出观察点，人工复核）"]
    assert obj.failure_triggers == ["（LLM 未给出失败触发条件，人工复核）"]
    assert sorted(obj.degraded_fields) == ["failure_triggers", "next_day_watch_points"]


# ---------------------------------------------------------------------------
# "fallback" — 维持 v0.12.3 及之前行为：仅占位符、不标注 degraded_fields
# ---------------------------------------------------------------------------


def test_fallback_policy_fills_placeholder_without_recording() -> None:
    payload = _base_payload(next_day_watch_points=[], failure_triggers=[])
    with apply_empty_array_policy("fallback"):
        obj = ContinuationCandidate.model_validate(payload)
    assert obj.next_day_watch_points == ["（LLM 未给出观察点，人工复核）"]
    assert obj.failure_triggers == ["（LLM 未给出失败触发条件，人工复核）"]
    assert obj.degraded_fields == []


# ---------------------------------------------------------------------------
# Policy is scoped to the context manager (ContextVar reset on exit)
# ---------------------------------------------------------------------------


def test_policy_resets_after_context_exits() -> None:
    payload = _base_payload(next_day_watch_points=[])
    # Inside degraded ctx → no raise
    with apply_empty_array_policy("degraded"):
        ContinuationCandidate.model_validate(payload)
    # After exit → back to default "repair"
    with pytest.raises(ValidationError):
        ContinuationCandidate.model_validate(payload)


# ---------------------------------------------------------------------------
# Non-empty input is untouched by all three policies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["repair", "degraded", "fallback"])
def test_non_empty_arrays_pass_through_all_policies(policy: str) -> None:
    payload = _base_payload()
    with apply_empty_array_policy(policy):  # type: ignore[arg-type]
        obj = ContinuationCandidate.model_validate(payload)
    assert obj.next_day_watch_points == ["盯紧 9:30 跳空"]
    assert obj.failure_triggers == ["跌破 5 日均线"]
    assert obj.degraded_fields == []


# ---------------------------------------------------------------------------
# RevisedContinuationCandidate inherits the same policy semantics
# ---------------------------------------------------------------------------


def test_revised_candidate_inherits_policy() -> None:
    payload = _base_payload(next_day_watch_points=[])
    payload["revision_note"] = "downgrade due to fading mood"
    with pytest.raises(ValidationError):
        RevisedContinuationCandidate.model_validate(payload)
    with apply_empty_array_policy("degraded"):
        obj = RevisedContinuationCandidate.model_validate(payload)
    assert obj.degraded_fields == ["next_day_watch_points"]
    assert obj.revision_note == "downgrade due to fading mood"


# ---------------------------------------------------------------------------
# EvidenceItem still allows list[str] (legacy) — P2-1 changes that later
# ---------------------------------------------------------------------------


def test_evidence_item_constructable() -> None:
    """Sanity: empty-array policy didn't break the EvidenceItem schema."""
    ev = EvidenceItem(field="up_stat", value="3连板", unit="次", interpretation="x")
    assert ev.field == "up_stat"
