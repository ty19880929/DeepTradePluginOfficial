"""Unit tests for Step 0 trade-date resolution after the v0.6.10 rewrite.

The old ``resolve_trade_date`` derived T from ``datetime.now()`` and
``trade_cal``, which silently anchored to the wrong day whenever the machine's
clock or timezone was off. T is now sourced from market data via
:func:`fetch_latest_trade_date` (``index_daily`` probe); ``resolve_trade_date``
itself is a pure transform over a caller-supplied probe value + the user
override.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from limit_up_board.calendar import TradeCalendar
from limit_up_board.data import fetch_latest_trade_date, resolve_trade_date


def _calendar_with_open_days(open_days: list[str]) -> TradeCalendar:
    """Build a TradeCalendar where exactly ``open_days`` are open."""
    all_days = sorted(set(open_days) | {"20260101", "20271231"})
    rows = []
    prev_open: str | None = None
    for d in all_days:
        is_open = d in open_days
        rows.append({"cal_date": d, "is_open": 1 if is_open else 0, "pretrade_date": prev_open})
        if is_open:
            prev_open = d
    return TradeCalendar(pd.DataFrame(rows))


class TestResolveTradeDate:
    def test_user_specified_wins_over_probe(self) -> None:
        cal = _calendar_with_open_days(["20260601", "20260602", "20260603"])
        T, T1 = resolve_trade_date(
            cal,
            latest_trade_date="20260603",
            user_specified="20260601",
        )
        assert T == "20260601"
        assert T1 == "20260602"

    def test_probe_used_when_no_user_override(self) -> None:
        cal = _calendar_with_open_days(["20260601", "20260602", "20260603"])
        T, T1 = resolve_trade_date(cal, latest_trade_date="20260602")
        assert T == "20260602"
        assert T1 == "20260603"

    def test_wrong_local_clock_does_not_affect_T(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Even with a wildly-wrong fake `datetime.now`, T comes from the probe."""
        # Pretend the machine thinks it's 2030 — the function still must use
        # the value the caller (i.e. fetch_latest_trade_date) supplies.
        class _FakeDt(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return datetime(2030, 1, 1, 12, 0, 0)

        monkeypatch.setattr("limit_up_board.data.datetime", _FakeDt)
        cal = _calendar_with_open_days(["20260601", "20260602"])
        T, T1 = resolve_trade_date(cal, latest_trade_date="20260601")
        assert T == "20260601"
        assert T1 == "20260602"

    def test_missing_both_inputs_raises(self) -> None:
        cal = _calendar_with_open_days(["20260601", "20260602"])
        with pytest.raises(ValueError, match="user_specified or latest_trade_date"):
            resolve_trade_date(cal, latest_trade_date=None, user_specified=None)


class TestFetchLatestTradeDate:
    def test_returns_max_trade_date_from_response(self) -> None:
        tushare = MagicMock()
        tushare.call.return_value = pd.DataFrame(
            {
                "ts_code": ["000001.SH"] * 3,
                "trade_date": ["20260530", "20260602", "20260601"],
                "close": [3000.0, 3010.0, 3005.0],
            }
        )
        out = fetch_latest_trade_date(tushare)
        assert out == "20260602"
        # The probe MUST force-sync to bypass tushare's trade_day_immutable cache.
        kwargs = tushare.call.call_args.kwargs
        assert kwargs["force_sync"] is True
        assert kwargs["params"]["ts_code"] == "000001.SH"
        assert "start_date" in kwargs["params"] and "end_date" in kwargs["params"]

    def test_empty_response_raises(self) -> None:
        tushare = MagicMock()
        tushare.call.return_value = pd.DataFrame(columns=["ts_code", "trade_date"])
        with pytest.raises(RuntimeError, match="index_daily"):
            fetch_latest_trade_date(tushare)

    def test_none_response_raises(self) -> None:
        tushare = MagicMock()
        tushare.call.return_value = None
        with pytest.raises(RuntimeError, match="index_daily"):
            fetch_latest_trade_date(tushare)

    def test_wrong_local_clock_still_returns_probe_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe's local-time window is only a quota-friendly bound — it
        must not override the trade_date Tushare actually returned."""
        class _FakeDt(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return datetime(2099, 1, 1, 0, 0, 0)

        monkeypatch.setattr("limit_up_board.data.datetime", _FakeDt)
        tushare = MagicMock()
        tushare.call.return_value = pd.DataFrame(
            {"ts_code": ["000001.SH"], "trade_date": ["20260530"], "close": [3000.0]}
        )
        assert fetch_latest_trade_date(tushare) == "20260530"
