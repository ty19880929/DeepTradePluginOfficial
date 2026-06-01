"""resolve_window — single-day / range / snap-backward / cap enforcement."""

from __future__ import annotations

import pandas as pd
import pytest

from market_review.calendar import TradeCalendar
from market_review.windows import (
    HARD_MAX_WINDOW_DAYS,
    Window,
    WindowSpecError,
    resolve_window,
)


def _cal_from(*open_days: str, holidays: tuple[str, ...] = ()) -> TradeCalendar:
    """Build a calendar where ``open_days`` are open + ``holidays`` are closed."""
    rows: list[dict] = []
    for d in open_days:
        rows.append({"exchange": "SSE", "cal_date": d, "is_open": 1, "pretrade_date": None})
    for d in holidays:
        rows.append({"exchange": "SSE", "cal_date": d, "is_open": 0, "pretrade_date": None})
    return TradeCalendar(pd.DataFrame(rows))


def test_day_mode_implicit_uses_latest_trade_date() -> None:
    cal = _cal_from("20260526", "20260527", "20260528")
    w = resolve_window(cal, latest_trade_date="20260528")
    assert w.mode == "day"
    assert w.trade_dates == ("20260528",)
    assert w.anchor == "20260528"
    assert w.snapped_from is None
    assert w.n_days == 1


def test_day_mode_implicit_requires_latest_when_no_args() -> None:
    cal = _cal_from("20260528")
    with pytest.raises(WindowSpecError, match="latest_trade_date"):
        resolve_window(cal)


def test_day_mode_implicit_snaps_when_probe_falls_on_holiday() -> None:
    cal = _cal_from("20260527", holidays=("20260528",))
    w = resolve_window(cal, latest_trade_date="20260528")
    assert w.trade_dates == ("20260527",)
    assert w.snapped_from == ("20260528", "20260528")


def test_day_mode_explicit_snaps_non_trading_input() -> None:
    cal = _cal_from("20260527", "20260530", holidays=("20260528", "20260529"))
    w = resolve_window(cal, trade_date="20260529")
    assert w.mode == "day"
    assert w.anchor == "20260527"
    assert w.snapped_from == ("20260529", "20260529")


def test_day_mode_explicit_keeps_open_date() -> None:
    cal = _cal_from("20260527")
    w = resolve_window(cal, trade_date="20260527")
    assert w.anchor == "20260527"
    assert w.snapped_from is None


def test_range_mode_collects_open_days_in_window() -> None:
    cal = _cal_from(
        "20260520", "20260521", "20260522", "20260525", "20260526", "20260527",
    )
    w = resolve_window(cal, start="20260520", end="20260527")
    assert w.mode == "range"
    assert w.start == "20260520"
    assert w.end == "20260527"
    assert w.anchor == "20260527"
    assert w.trade_dates == (
        "20260520", "20260521", "20260522", "20260525", "20260526", "20260527",
    )
    assert w.n_days == 6


def test_range_mode_snaps_both_endpoints() -> None:
    cal = _cal_from(
        "20260518", "20260520", "20260521",
        holidays=("20260519",),
    )
    # start=20260519 (holiday) → snap back to 20260518.
    # end=20260522 (not in calendar at all) → latest_closed_on_or_before walks
    # back to 20260521 (the most recent open day).
    w = resolve_window(cal, start="20260519", end="20260522")
    assert w.start == "20260518"
    assert w.end == "20260521"
    assert w.snapped_from == ("20260519", "20260522")
    assert w.trade_dates == ("20260518", "20260520", "20260521")


def test_range_mode_rejects_start_after_end() -> None:
    cal = _cal_from("20260520", "20260521")
    with pytest.raises(WindowSpecError, match="不能晚于"):
        resolve_window(cal, start="20260521", end="20260520")


def test_range_mode_rejects_window_over_max() -> None:
    open_days = [f"202605{d:02d}" for d in range(1, 32) if d not in (4, 5, 11, 12, 18, 19, 25, 26)]
    cal = _cal_from(*open_days)
    # 21 trading days — comfortably more than max_window_days=5
    with pytest.raises(WindowSpecError, match="超过 max_window_days"):
        resolve_window(cal, start=open_days[0], end=open_days[-1], max_window_days=5)


def test_resolver_rejects_invalid_max_window_days() -> None:
    cal = _cal_from("20260527")
    with pytest.raises(WindowSpecError, match="max_window_days"):
        resolve_window(cal, trade_date="20260527", max_window_days=0)
    with pytest.raises(WindowSpecError, match="硬上限"):
        resolve_window(cal, trade_date="20260527", max_window_days=HARD_MAX_WINDOW_DAYS + 1)


def test_resolver_rejects_mutex_violation() -> None:
    cal = _cal_from("20260527")
    with pytest.raises(WindowSpecError, match="互斥"):
        resolve_window(cal, trade_date="20260527", start="20260520")
    with pytest.raises(WindowSpecError, match="互斥"):
        resolve_window(cal, start="20260520")  # missing end


def test_resolver_rejects_bad_date_format() -> None:
    cal = _cal_from("20260527")
    with pytest.raises(WindowSpecError, match="YYYYMMDD"):
        resolve_window(cal, trade_date="2026-05-27")


def test_range_with_no_open_days_in_span_raises() -> None:
    # Span is entirely before the earliest open day in the calendar — snapping
    # backward has nothing to find, so latest_closed_on_or_before raises.
    cal = _cal_from("20260601", holidays=("20260101", "20260102"))
    with pytest.raises(ValueError):
        resolve_window(cal, start="20260101", end="20260102")


def test_window_is_frozen() -> None:
    w = Window(
        mode="day", start="20260527", end="20260527",
        trade_dates=("20260527",), anchor="20260527",
    )
    with pytest.raises((AttributeError, TypeError)):
        w.start = "20260528"  # type: ignore[misc]
