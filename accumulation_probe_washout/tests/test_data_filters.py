"""filter_main_board + ST/suspend filtering — T2.3."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from accumulation_probe_washout.calendar import TradeCalendar
from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.data import (
    filter_main_board,
    filter_st_and_suspend,
    resolve_trade_date,
)


def _make_basic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Main board SH — keep
            {"ts_code": "600000.SH", "name": "浦发银行", "market": "主板",
             "exchange": "SSE", "list_status": "L", "list_date": "20100101"},
            # Main board SZ — keep
            {"ts_code": "000001.SZ", "name": "平安银行", "market": "主板",
             "exchange": "SZSE", "list_status": "L", "list_date": "20100101"},
            # STAR board — drop
            {"ts_code": "688001.SH", "name": "华兴源创", "market": "主板",
             "exchange": "SSE", "list_status": "L", "list_date": "20100101"},
            # ChiNext — drop
            {"ts_code": "300001.SZ", "name": "特锐德", "market": "创业板",
             "exchange": "SZSE", "list_status": "L", "list_date": "20100101"},
            # BSE — drop
            {"ts_code": "830799.BJ", "name": "北证某", "market": "北交所",
             "exchange": "BSE", "list_status": "L", "list_date": "20100101"},
            # Shenzhen B-share (200xxx) — drop
            {"ts_code": "200012.SZ", "name": "南玻B", "market": "主板",
             "exchange": "SZSE", "list_status": "L", "list_date": "20100101"},
            # Shanghai B-share (900xxx) — drop
            {"ts_code": "900901.SH", "name": "外高B股", "market": "主板",
             "exchange": "SSE", "list_status": "L", "list_date": "20100101"},
            # Main board but delisted — drop
            {"ts_code": "600999.SH", "name": "delisted", "market": "主板",
             "exchange": "SSE", "list_status": "D", "list_date": "20100101"},
            # New stock — drop when listed_days_min=120
            {"ts_code": "600101.SH", "name": "new stock", "market": "主板",
             "exchange": "SSE", "list_status": "L", "list_date": "20260301"},
        ]
    )


class TestMainBoardFilter:
    def test_keeps_only_main_board(self) -> None:
        cfg = ApwConfig()
        df = _make_basic()
        out = filter_main_board(df, cfg)
        codes = set(out["ts_code"])
        # 600999 is delisted, 600101 is new — without trade_date the latter stays
        assert "600000.SH" in codes
        assert "000001.SZ" in codes
        assert "688001.SH" not in codes
        assert "300001.SZ" not in codes
        assert "830799.BJ" not in codes
        assert "600999.SH" not in codes
        # B-shares (200xxx.SZ / 900xxx.SH) must not survive the filter (P3-1).
        assert "200012.SZ" not in codes
        assert "900901.SH" not in codes

    def test_excludes_b_share_prefixes(self) -> None:
        """Regression guard for P3-1 — B-share prefixes must be dropped."""
        cfg = ApwConfig()
        df = _make_basic()
        out = filter_main_board(df, cfg)
        codes = set(out["ts_code"])
        assert "200012.SZ" not in codes, "Shenzhen B-share 200xxx leaked through"
        assert "900901.SH" not in codes, "Shanghai B-share 900xxx leaked through"

    def test_listed_days_filter(self) -> None:
        cfg = ApwConfig()  # listed_days_min = 120
        df = _make_basic()
        out = filter_main_board(df, cfg, trade_date="20260515")
        codes = set(out["ts_code"])
        # listed on 20260301 → ~75 calendar days as of 20260515 → drop
        assert "600101.SH" not in codes
        assert "600000.SH" in codes


def _calendar_through(last_date: str) -> TradeCalendar:
    """Build a tiny TradeCalendar that ends exactly at ``last_date`` (open day).

    Used to reproduce P1-1: when the loaded trade_cal doesn't extend past T,
    ``next_open(T)`` would raise. ``resolve_trade_date`` must degrade gracefully
    instead of crashing the screen/analyze pipeline.
    """
    from datetime import timedelta as _td

    end = datetime.strptime(last_date, "%Y%m%d")
    opens: list[str] = []
    cur = end
    while len(opens) < 30:
        if cur.weekday() < 5:  # Mon–Fri only
            opens.append(cur.strftime("%Y%m%d"))
        cur = cur - _td(days=1)
    opens.reverse()
    rows = []
    prev: str | None = None
    for d in opens:
        rows.append({"cal_date": d, "is_open": 1, "pretrade_date": prev})
        prev = d
    return TradeCalendar(pd.DataFrame(rows))


class TestResolveTradeDate:
    """P1-1 — resolve_trade_date must not crash when calendar ends at T."""

    def test_user_specified_T_at_calendar_tail_falls_back(self) -> None:
        cal = _calendar_through("20260515")  # last open day == T
        T, next_T = resolve_trade_date(cal, user_specified="20260515")
        assert T == "20260515"
        # Fallback synthesizes T+1 calendar day instead of raising.
        assert next_T == "20260516"

    def test_probe_used_when_no_user_override(self) -> None:
        cal = _calendar_through("20260515")
        T, next_T = resolve_trade_date(cal, latest_trade_date="20260515")
        assert T == "20260515"
        assert next_T == "20260516"

    def test_next_open_used_when_calendar_extends_forward(self) -> None:
        # Calendar with future open days — fallback path NOT triggered.
        rows = [
            {"cal_date": "20260515", "is_open": 1, "pretrade_date": "20260514"},
            {"cal_date": "20260516", "is_open": 0, "pretrade_date": "20260515"},
            {"cal_date": "20260517", "is_open": 0, "pretrade_date": "20260515"},
            {"cal_date": "20260518", "is_open": 1, "pretrade_date": "20260515"},
        ]
        cal = TradeCalendar(pd.DataFrame(rows))
        T, next_T = resolve_trade_date(cal, user_specified="20260515")
        assert T == "20260515"
        assert next_T == "20260518"  # real next open, not the synthetic fallback


class TestStAndSuspend:
    def test_drops_st_and_suspended(self) -> None:
        df = pd.DataFrame(
            [
                {"ts_code": "600000.SH"},
                {"ts_code": "600001.SH"},
                {"ts_code": "600002.SH"},
                {"ts_code": "600003.SH"},
            ]
        )
        out = filter_st_and_suspend(df, st_codes={"600001.SH"}, suspended_codes={"600002.SH"})
        codes = set(out["ts_code"])
        assert codes == {"600000.SH", "600003.SH"}
