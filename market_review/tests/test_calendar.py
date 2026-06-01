"""TradeCalendar — pure DataFrame round-trip checks."""

from __future__ import annotations

import pandas as pd
import pytest

from market_review.calendar import TradeCalendar


def _make_cal(dates_open: dict[str, int]) -> pd.DataFrame:
    """Build a minimal trade_cal frame from {cal_date: 0/1}."""
    rows = [
        {"exchange": "SSE", "cal_date": d, "is_open": is_open, "pretrade_date": None}
        for d, is_open in sorted(dates_open.items())
    ]
    return pd.DataFrame(rows)


def test_constructor_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        TradeCalendar(pd.DataFrame({"cal_date": ["20260101"]}))


def test_is_open_and_pretrade_next_open() -> None:
    cal = TradeCalendar(
        _make_cal({
            "20260101": 0,  # holiday
            "20260102": 1,  # Mon
            "20260103": 1,  # Tue
            "20260104": 0,  # Wed holiday
            "20260105": 1,  # Thu
        })
    )
    assert cal.is_open("20260102") is True
    assert cal.is_open("20260101") is False
    assert cal.is_open("20990101") is False  # unknown → False

    assert cal.pretrade_date("20260105") == "20260103"
    assert cal.next_open("20260103") == "20260105"

    with pytest.raises(ValueError, match="no prior open"):
        cal.pretrade_date("20260101")
    with pytest.raises(ValueError, match="no future open"):
        cal.next_open("20260105")


def test_latest_closed_on_or_before_snaps_backward() -> None:
    cal = TradeCalendar(_make_cal({"20260102": 1, "20260103": 1, "20260104": 0}))
    assert cal.latest_closed_on_or_before("20260103") == "20260103"
    assert cal.latest_closed_on_or_before("20260104") == "20260103"


def test_range_ascending_and_filters() -> None:
    cal = TradeCalendar(
        _make_cal({
            "20260101": 0,
            "20260102": 1,
            "20260103": 1,
            "20260104": 0,
            "20260105": 1,
        })
    )
    assert cal.range("20260101", "20260105") == ["20260102", "20260103", "20260105"]
    assert cal.range("20260103", "20260103") == ["20260103"]
    # reversed range → []
    assert cal.range("20260105", "20260101") == []


def test_constructor_normalizes_int_cal_date() -> None:
    """Tushare occasionally returns cal_date as int64; calendar must stringify."""
    df = pd.DataFrame(
        [
            {"exchange": "SSE", "cal_date": 20260102, "is_open": 1, "pretrade_date": None},
            {"exchange": "SSE", "cal_date": 20260103, "is_open": "1", "pretrade_date": None},
        ]
    )
    cal = TradeCalendar(df)
    assert cal.is_open("20260102")
    assert cal.is_open("20260103")
