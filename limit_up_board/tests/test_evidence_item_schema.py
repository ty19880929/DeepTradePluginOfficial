import pytest
from pydantic import ValidationError
from limit_up_board.schemas import EvidenceItem

def test_evidence_item_accepts_list_value():
    """Verify that EvidenceItem.value now accepts a list of strings."""
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
    """Verify that EvidenceItem.value still accepts scalars."""
    for val in ["string", 123, 45.6, None]:
        item = EvidenceItem(
            field="test_field",
            value=val,
            unit="unit",
            interpretation="interp"
        )
        assert item.value == val

def test_evidence_item_rejects_other_types():
    """Verify that EvidenceItem.value still rejects unsupported types (like dict)."""
    with pytest.raises(ValidationError):
        EvidenceItem(
            field="test_field",
            value={"key": "value"},
            unit="unit",
            interpretation="interp"
        )
