import pytest
from pydantic import ValidationError
from limit_up_board.schemas import EvidenceItem, EvidenceItemStrict


def test_evidence_item_legacy_allows_list():
    """v0.12.4 (P2-1): the LEGACY EvidenceItem keeps allowing list[str] values so
    historical lub_stage_results rows still deserialize cleanly under
    report rebuild / winrate review."""
    data = {
        "field": "lhb_famous_seats",
        "value": ["Seat A", "Seat B"],
        "unit": "无",
        "interpretation": "Famous seats detected"
    }
    item = EvidenceItem(**data)
    assert item.value == ["Seat A", "Seat B"]
    assert item.field == "lhb_famous_seats"


def test_evidence_item_accepts_scalar_values():
    """EvidenceItem (legacy) still accepts scalars too."""
    for val in ["string", 123, 45.6, None]:
        item = EvidenceItem(
            field="test_field",
            value=val,
            unit="unit",
            interpretation="interp"
        )
        assert item.value == val


def test_evidence_item_rejects_other_types():
    """EvidenceItem (legacy) rejects dict / set / etc."""
    with pytest.raises(ValidationError):
        EvidenceItem(
            field="test_field",
            value={"key": "value"},
            unit="unit",
            interpretation="interp"
        )


# ---------------------------------------------------------------------------
# v0.12.4 (P2-1) — EvidenceItemStrict rejects list values to align with the
# prompt's "严禁数组" hard constraint. StrongCandidate / ContinuationCandidate
# now use this strict schema.
# ---------------------------------------------------------------------------


def test_evidence_item_strict_rejects_list():
    with pytest.raises(ValidationError):
        EvidenceItemStrict(
            field="lhb_famous_seats",
            value=["Seat A", "Seat B"],
            unit="无",
            interpretation="x",
        )


def test_evidence_item_strict_accepts_scalar_values():
    for val in ["text", 7, 3.14, None]:
        item = EvidenceItemStrict(
            field="f",
            value=val,
            unit="unit",
            interpretation="x",
        )
        assert item.value == val


def test_evidence_item_strict_rejects_dict():
    with pytest.raises(ValidationError):
        EvidenceItemStrict(
            field="f",
            value={"a": 1},  # type: ignore[arg-type]
            unit="unit",
            interpretation="x",
        )
