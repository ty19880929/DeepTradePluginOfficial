"""Determinism guard for ``_fetch_history_window`` (v0.18).

Root cause of the "same trade_date → different LLM result" report: a transient
whole-market empty on a single open trade-date was silently dropped (``continue``)
and never cached as ``ok``, so the next run re-fetched it and might get data —
shifting the trailing ``closes[-N:]`` window membership and drifting ma5/ma10/ma20
(and the LGB features/score derived from them, and thus the entire prompt).

These tests pin the new contract:
  * an empty day is retried once with ``force_sync=True`` (repopulate the immutable
    cache) when ``retry_empty_days=True``;
  * a day still empty after the retry is reported via ``missing_days`` (loud, not
    silent) rather than vanishing;
  * a day with whole-market data but no *candidate* rows is NOT a missing day.
"""

from __future__ import annotations

import pandas as pd

from limit_up_board.data import _fetch_history_window


class _FakeTushare:
    """Records every ``call`` so we can assert force_sync retries fired.

    ``trade_cal`` returns a calendar covering the requested window; per-API
    per-date frames are looked up in ``self.frames[(api, date)]`` (default empty).
    ``empty_until`` maps (api, date) → number of times to return empty before
    yielding the real frame, simulating a transient blip that a force_sync retry
    overcomes.
    """

    def __init__(self, open_days, frames, empty_until=None):
        self._open_days = open_days
        self.frames = frames
        self.empty_until = dict(empty_until or {})
        self.calls: list[tuple[str, str | None, bool]] = []

    def call(self, api_name, *, trade_date=None, params=None, fields=None, force_sync=False):
        self.calls.append((api_name, trade_date, force_sync))
        if api_name == "trade_cal":
            return pd.DataFrame(
                {"cal_date": self._open_days, "is_open": [1] * len(self._open_days)}
            )
        key = (api_name, trade_date)
        remaining = self.empty_until.get(key, 0)
        if remaining > 0:
            self.empty_until[key] = remaining - 1
            return pd.DataFrame()
        return self.frames.get(key, pd.DataFrame())


def _daily_row(ts_code, date, close):
    return pd.DataFrame({"ts_code": [ts_code], "trade_date": [date], "close": [close]})


def test_transient_empty_day_is_retried_and_recovered() -> None:
    days = ["20260520", "20260521", "20260522"]
    codes = {"000001.SZ"}
    frames = {("daily", d): _daily_row("000001.SZ", d, 10.0 + i) for i, d in enumerate(days)}
    # 20260521 returns empty exactly once, then real data on the force_sync retry.
    fake = _FakeTushare(days, frames, empty_until={("daily", "20260521"): 1})

    df, missing = _fetch_history_window(
        fake, "daily", "20260520", "20260522", codes, retry_empty_days=True
    )

    assert missing == []  # recovered via retry → no gap surfaced
    assert sorted(df["trade_date"].tolist()) == days  # all three bars present
    # a force_sync retry must have fired for the flaky day
    assert ("daily", "20260521", True) in fake.calls


def test_persistent_empty_day_is_reported_not_dropped() -> None:
    days = ["20260520", "20260521", "20260522"]
    codes = {"000001.SZ"}
    frames = {
        ("daily", "20260520"): _daily_row("000001.SZ", "20260520", 10.0),
        ("daily", "20260522"): _daily_row("000001.SZ", "20260522", 12.0),
        # 20260521 never has data
    }
    fake = _FakeTushare(days, frames, empty_until={("daily", "20260521"): 99})

    df, missing = _fetch_history_window(
        fake, "daily", "20260520", "20260522", codes, retry_empty_days=True
    )

    assert missing == ["20260521"]  # loud, not silent
    assert sorted(df["trade_date"].tolist()) == ["20260520", "20260522"]


def test_no_retry_when_flag_off() -> None:
    days = ["20260520", "20260521"]
    codes = {"000001.SZ"}
    frames = {("daily", "20260520"): _daily_row("000001.SZ", "20260520", 10.0)}
    fake = _FakeTushare(days, frames, empty_until={("daily", "20260521"): 99})

    df, missing = _fetch_history_window(
        fake, "daily", "20260520", "20260521", codes, retry_empty_days=False
    )

    assert missing == ["20260521"]
    # no force_sync retry fired (retry disabled)
    assert not any(c == ("daily", "20260521", True) for c in fake.calls)


def test_day_with_no_candidate_rows_is_not_missing() -> None:
    """A day whose whole-market frame is non-empty but contains none of our
    candidates is a legitimate no-op, not a window gap."""
    days = ["20260520", "20260521"]
    codes = {"000001.SZ"}
    frames = {
        ("daily", "20260520"): _daily_row("000001.SZ", "20260520", 10.0),
        # 20260521 has data, but for a different stock only
        ("daily", "20260521"): _daily_row("600519.SH", "20260521", 1800.0),
    }
    fake = _FakeTushare(days, frames)

    df, missing = _fetch_history_window(
        fake, "daily", "20260520", "20260521", codes, retry_empty_days=True
    )

    assert missing == []  # whole-market frame was non-empty → not a gap
    assert df["trade_date"].tolist() == ["20260520"]
