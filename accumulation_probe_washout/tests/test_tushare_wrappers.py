"""Tushare thin wrappers — T2.9 (degradation on optional API failure)."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.data import (
    fetch_daily_basic_on,
    fetch_index_daily,
    fetch_moneyflow,
)


class _FakeTushare:
    def __init__(self, raise_on: set[str] | None = None, data: dict | None = None) -> None:
        self.raise_on = raise_on or set()
        self.data = data or {}

    def call(
        self,
        api: str,
        *,
        trade_date=None,
        params=None,
        fields=None,
        force_sync=False,
    ):
        if api in self.raise_on:
            raise RuntimeError(f"{api} unavailable")
        return self.data.get(api, pd.DataFrame())


class TestOptionalDegradation:
    def test_moneyflow_failure_returns_empty_outcome(self) -> None:
        fake = _FakeTushare(raise_on={"moneyflow"})
        outcome = fetch_moneyflow(fake, ts_codes=["600000.SH"], start="20260101", end="20260131")
        assert outcome.df.empty
        assert "moneyflow" in outcome.missing

    def test_moneyflow_success_returns_data(self) -> None:
        df = pd.DataFrame(
            [{"ts_code": "600000.SH", "trade_date": "20260101", "net_mf_amount": 5000.0}]
        )
        fake = _FakeTushare(data={"moneyflow": df})
        outcome = fetch_moneyflow(fake, ts_codes=["600000.SH"], start="20260101", end="20260131")
        assert not outcome.df.empty
        assert outcome.missing == []

    def test_index_daily_failure_degrades(self) -> None:
        fake = _FakeTushare(raise_on={"index_daily"})
        outcome = fetch_index_daily(fake, index_code="000300.SH", start="20260101", end="20260131")
        assert outcome.df.empty
        assert "index_daily" in outcome.missing


class _BatchRecordingTushare:
    """Records the ``ts_code`` list passed in each moneyflow call so the test
    can assert no single batch exceeds the 1000-entry Tushare cap."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def call(
        self,
        api: str,
        *,
        trade_date=None,
        params=None,
        fields=None,
        force_sync=False,
    ):
        if api != "moneyflow":
            return pd.DataFrame()
        codes = (params or {}).get("ts_code", "").split(",")
        codes = [c for c in codes if c]
        self.calls.append(codes)
        # Emulate real moneyflow shape — one row per ts_code in this batch.
        rows = [
            {"ts_code": c, "trade_date": "20260101", "net_mf_amount": 1000.0}
            for c in codes
        ]
        return pd.DataFrame(rows)


class TestMoneyflowBatching:
    def test_batches_respect_tushare_limit(self) -> None:
        """Tushare caps ``moneyflow`` ts_code lists at 1000. Passing ~3000
        codes must split into multiple calls, each ≤ batch size, and the
        merged frame must contain every code exactly once.
        """
        codes = [f"60{i:04d}.SH" for i in range(2996)]
        fake = _BatchRecordingTushare()
        outcome = fetch_moneyflow(
            fake, ts_codes=codes, start="20260101", end="20260131"
        )

        # Every batch is ≤ 1000 (Tushare hard cap)
        assert fake.calls, "moneyflow must be called at least once"
        for batch in fake.calls:
            assert 0 < len(batch) <= 1000, (
                f"batch of {len(batch)} exceeds Tushare 1000-code list cap"
            )

        # All codes flowed through, no duplicates
        observed = [c for batch in fake.calls for c in batch]
        assert sorted(observed) == sorted(codes)
        assert len(outcome.df) == len(codes)
        assert outcome.missing == []

    def test_batch_failure_degrades_only_that_batch(self) -> None:
        """A single failing batch shouldn't wipe the whole result."""

        class _PartialFail:
            def __init__(self) -> None:
                self.n = 0

            def call(self, api: str, *, params=None, **kwargs):
                self.n += 1
                if self.n == 2:
                    raise RuntimeError("transient")
                codes = (params or {}).get("ts_code", "").split(",")
                return pd.DataFrame(
                    [{"ts_code": c, "trade_date": "20260101", "net_mf_amount": 1.0}
                     for c in codes if c]
                )

        codes = [f"60{i:04d}.SH" for i in range(1600)]
        outcome = fetch_moneyflow(
            _PartialFail(), ts_codes=codes, start="20260101", end="20260131"
        )
        # batch1 (800) succeeded, batch2 (800) failed.
        assert not outcome.df.empty
        assert "moneyflow" in outcome.missing


class TestDailyBasicOnTradeDate:
    def test_fetch_daily_basic_on_passes_trade_date_kwarg(self) -> None:
        """The fix replaces multi-code ts_code list with a single
        ``trade_date`` snapshot — verify the kwarg shape."""

        captured: dict = {}

        class _Fake:
            def call(self, api, *, trade_date=None, params=None, **kwargs):
                captured["api"] = api
                captured["trade_date"] = trade_date
                captured["params"] = params
                return pd.DataFrame(
                    [{"ts_code": "600000.SH", "trade_date": trade_date,
                      "turnover_rate": 1.0, "circ_mv": 200000.0}]
                )

        df = fetch_daily_basic_on(_Fake(), trade_date="20260520")
        assert captured["api"] == "daily_basic"
        assert captured["trade_date"] == "20260520"
        # ts_code must NOT appear in params — that's the shape that returns 0 rows.
        assert captured["params"] is None
        assert not df.empty
        assert df.iloc[0]["trade_date"] == "20260520"

    def test_fetch_daily_basic_on_handles_none_response(self) -> None:
        class _Fake:
            def call(self, api, **kwargs):
                return None

        df = fetch_daily_basic_on(_Fake(), trade_date="20260520")
        assert df.empty
