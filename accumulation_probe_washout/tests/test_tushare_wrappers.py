"""Tushare thin wrappers — T2.9 (degradation on optional API failure)."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.data import fetch_moneyflow, fetch_index_daily


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
