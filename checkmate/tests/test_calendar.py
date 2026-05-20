"""Tests for checkmate.calendar — TradeCalendar primitives + parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from checkmate.calendar import TradeCalendar, load_trade_calendar


# Synthetic 2-week window: 2024-01-01 (Mon, NY closed but A-share holiday so
# we mark is_open=0 for that week's first two days), then 03/04/05 open, etc.
# The exact open/closed pattern doesn't need to mirror reality — the helper
# only cares about boolean is_open and ordering by cal_date.
_FRAME = pd.DataFrame({
    "cal_date": [
        "20240101", "20240102", "20240103", "20240104", "20240105",
        "20240108", "20240109", "20240110", "20240111", "20240112",
    ],
    "is_open": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
})


def test_is_trade_day() -> None:
    cal = TradeCalendar(_FRAME)
    assert cal.is_trade_day("20240102") is True
    assert cal.is_trade_day("20240101") is False
    assert cal.is_trade_day("20240112") is False
    # Out-of-window dates degrade to False, not raise.
    assert cal.is_trade_day("20990101") is False


def test_next_session_skips_closed_days() -> None:
    cal = TradeCalendar(_FRAME)
    assert cal.next_session("20240101") == "20240102"
    # If the input itself is a trade day, next_session returns the *next* one.
    assert cal.next_session("20240102") == "20240103"
    # Crossing a non-open weekend block (Fri 05 → Mon 08 here).
    assert cal.next_session("20240105") == "20240108"


def test_prev_session_skips_closed_days() -> None:
    cal = TradeCalendar(_FRAME)
    assert cal.prev_session("20240108") == "20240105"
    # Strict-before semantics: passing a trade day returns the previous one.
    assert cal.prev_session("20240103") == "20240102"


def test_prev_session_raises_when_no_prior_day() -> None:
    cal = TradeCalendar(_FRAME)
    with pytest.raises(ValueError):
        cal.prev_session("20240101")  # first row is closed, no opens before it


def test_sessions_in_range_inclusive_endpoints() -> None:
    cal = TradeCalendar(_FRAME)
    got = cal.sessions_in_range("20240103", "20240108")
    assert got == ["20240103", "20240104", "20240105", "20240108"]
    # Empty when start > end.
    assert cal.sessions_in_range("20240108", "20240103") == []


def test_n_sessions_before() -> None:
    cal = TradeCalendar(_FRAME)
    assert cal.n_sessions_before("20240110", 1) == "20240109"
    assert cal.n_sessions_before("20240110", 3) == "20240105"
    with pytest.raises(ValueError):
        cal.n_sessions_before("20240110", 99)


def test_normalisation_accepts_dashed_form() -> None:
    cal = TradeCalendar(_FRAME)
    assert cal.is_trade_day("2024-01-02") is True
    assert cal.next_session("2024-01-01") == "20240102"


# ---------------------------------------------------------------------------
# load_trade_calendar — parquet cache behaviour
# ---------------------------------------------------------------------------


class _StubTushare:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.calls: list[tuple[str, dict]] = []

    def call(self, api_name: str, **kwargs: object) -> pd.DataFrame:
        self.calls.append((api_name, dict(kwargs)))
        return self._frame.copy()


def test_load_trade_calendar_writes_then_hits_cache(tmp_path: Path) -> None:
    cache = tmp_path / "trade_cal.parquet"
    stub = _StubTushare(_FRAME)

    cal1 = load_trade_calendar(stub, cache_path=cache)
    assert cache.is_file()
    assert cal1.is_trade_day("20240102") is True
    assert len(stub.calls) == 1

    # Second load: cache hit — no new tushare call.
    cal2 = load_trade_calendar(stub, cache_path=cache)
    assert cal2.is_trade_day("20240102") is True
    assert len(stub.calls) == 1


def test_load_trade_calendar_refresh_bypasses_cache(tmp_path: Path) -> None:
    cache = tmp_path / "trade_cal.parquet"
    stub = _StubTushare(_FRAME)
    load_trade_calendar(stub, cache_path=cache)
    load_trade_calendar(stub, cache_path=cache, refresh=True)
    assert len(stub.calls) == 2


def test_load_trade_calendar_raises_on_empty_response(tmp_path: Path) -> None:
    stub = _StubTushare(pd.DataFrame(columns=["cal_date", "is_open"]))
    with pytest.raises(RuntimeError, match="trade_cal"):
        load_trade_calendar(stub, cache_path=tmp_path / "tc.parquet")
