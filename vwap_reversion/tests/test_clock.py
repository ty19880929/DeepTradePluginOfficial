"""时区 + 启动时机决策单测（设计 §4，P0 验收项）.

纯 stdlib —— 不依赖 deeptrade，可在任何环境直接跑。
核心场景：本机时区 ≠ 中国时区时，所有判断仍按 Asia/Shanghai。
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from vwap_reversion.clock import (
    AFTERNOON_CLOSE,
    MORNING_OPEN,
    MarketClock,
    SessionPhase,
    StartupAction,
    classify_phase,
    decide_startup,
    in_continuous_session,
    next_trading_day,
    parse_hhmm,
)

SH = ZoneInfo("Asia/Shanghai")


def clock_at(sh_dt: datetime) -> MarketClock:
    """构造一个「现在」固定为上海墙钟 *sh_dt* 的 MarketClock。"""
    epoch = sh_dt.replace(tzinfo=SH).timestamp()
    return MarketClock("Asia/Shanghai", now_fn=lambda: epoch)


def trading_days(*dates: str):
    """YYYYMMDD 集合 → is_trading_day 谓词。"""
    s = {date(int(d[:4]), int(d[4:6]), int(d[6:8])) for d in dates}
    return lambda d: d in s


# 2026-06-03 是周三；按周内日历造一个假交易周（周一~周五）。
WEEK = trading_days("20260601", "20260602", "20260603", "20260604", "20260605")


# ---------------------------------------------------------------------------
# classify_phase / session helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "phase"),
    [
        (time(0, 0), SessionPhase.PRE_OPEN),
        (time(9, 29, 59), SessionPhase.PRE_OPEN),
        (time(9, 30), SessionPhase.MORNING),        # 边界：09:30 整属盘中
        (time(11, 29, 59), SessionPhase.MORNING),
        (time(11, 30), SessionPhase.LUNCH),          # 边界：11:30 整属午休
        (time(12, 59, 59), SessionPhase.LUNCH),
        (time(13, 0), SessionPhase.AFTERNOON),
        (time(14, 59, 59), SessionPhase.AFTERNOON),
        (time(15, 0), SessionPhase.AFTER_CLOSE),     # 边界：15:00 整属收盘后
        (time(23, 59), SessionPhase.AFTER_CLOSE),
    ],
)
def test_classify_phase(t: time, phase: SessionPhase) -> None:
    assert classify_phase(t) is phase


def test_in_continuous_session() -> None:
    assert in_continuous_session(time(10, 0))
    assert in_continuous_session(time(14, 0))
    assert not in_continuous_session(time(12, 0))
    assert not in_continuous_session(time(9, 0))
    assert not in_continuous_session(time(15, 30))


# ---------------------------------------------------------------------------
# MarketClock — 本机时区无关性
# ---------------------------------------------------------------------------


def test_today_follows_market_tz_not_local() -> None:
    # 上海 2026-06-03 01:00 == UTC 2026-06-02 17:00。
    # 若误用本机/UTC 日期会得到 06-02；市场时区必须给 06-03。
    clk = clock_at(datetime(2026, 6, 3, 1, 0))
    assert clk.today_str() == "20260603"
    assert clk.now().tzinfo is not None
    assert clk.now().hour == 1


def test_wall_and_epoch_roundtrip() -> None:
    clk = clock_at(datetime(2026, 6, 3, 9, 0))
    open_dt = clk.wall(date(2026, 6, 3), MORNING_OPEN)
    assert open_dt.hour == 9 and open_dt.minute == 30
    # 上海 09:30 == UTC 01:30（无 DST）。
    assert open_dt.astimezone(timezone.utc).hour == 1
    assert clk.seconds_until(open_dt) == pytest.approx(30 * 60)


def test_parse_hhmm() -> None:
    assert parse_hhmm("14:55") == time(14, 55)
    assert parse_hhmm(" 9:05 ") == time(9, 5)
    with pytest.raises(ValueError):
        parse_hhmm("1455")


def test_next_trading_day_skips_gap_and_caps_lookahead() -> None:
    assert next_trading_day(date(2026, 6, 5), WEEK) is None  # 周五之后没了
    assert next_trading_day(date(2026, 6, 3), WEEK) == date(2026, 6, 4)
    assert next_trading_day(date(2026, 5, 30), WEEK) == date(2026, 6, 1)
    assert next_trading_day(date(2026, 6, 5), lambda d: False) is None  # 30 天封顶


# ---------------------------------------------------------------------------
# decide_startup（§4.3 决策表逐行覆盖）
# ---------------------------------------------------------------------------


def test_startup_before_open_standby_until_0930() -> None:
    clk = clock_at(datetime(2026, 6, 3, 8, 50))
    d = decide_startup(clk, is_trading_day=WEEK)
    assert d.action is StartupAction.STANDBY
    assert d.trade_date == "20260603"
    assert d.standby_until is not None
    assert d.standby_until.timetz().replace(tzinfo=None) == MORNING_OPEN
    assert d.standby_until_epoch == pytest.approx(clk.now_epoch() + 40 * 60)


def test_startup_before_open_in_foreign_local_tz_is_still_standby() -> None:
    # 等价场景：本机在 UTC（2026-06-03 00:50 UTC == 上海 08:50）。
    # MarketClock 只吃 epoch，决策必须与上一测试完全一致。
    epoch = datetime(2026, 6, 3, 0, 50, tzinfo=timezone.utc).timestamp()
    clk = MarketClock("Asia/Shanghai", now_fn=lambda: epoch)
    d = decide_startup(clk, is_trading_day=WEEK)
    assert d.action is StartupAction.STANDBY
    assert d.trade_date == "20260603"


@pytest.mark.parametrize(
    "hhmm",
    [time(9, 30), time(10, 15), time(11, 30), time(12, 30), time(13, 0), time(14, 59)],
)
def test_startup_intraday_and_lunch_run_now(hhmm: time) -> None:
    clk = clock_at(datetime(2026, 6, 3, hhmm.hour, hhmm.minute))
    d = decide_startup(clk, is_trading_day=WEEK)
    assert d.action is StartupAction.RUN_NOW
    assert d.trade_date == "20260603"


def test_startup_after_close_exits_with_next_day_hint() -> None:
    clk = clock_at(datetime(2026, 6, 3, 15, 0))
    d = decide_startup(clk, is_trading_day=WEEK)
    assert d.action is StartupAction.EXIT
    assert d.trade_date == "20260604"  # 提示下一交易日


def test_startup_after_close_standby_across_days() -> None:
    clk = clock_at(datetime(2026, 6, 3, 15, 30))
    d = decide_startup(clk, is_trading_day=WEEK, standby_across_days=True)
    assert d.action is StartupAction.STANDBY
    assert d.trade_date == "20260604"
    assert d.standby_until == datetime(2026, 6, 4, 9, 30, tzinfo=SH)


def test_startup_non_trading_day_exits() -> None:
    clk = clock_at(datetime(2026, 6, 6, 10, 0))  # 周六盘中时刻也不行
    d = decide_startup(clk, is_trading_day=WEEK)
    assert d.action is StartupAction.EXIT
    assert d.trade_date is None  # WEEK 里 06-06 之后无交易日


def test_startup_non_trading_day_standby_across_days() -> None:
    days = trading_days("20260608")  # 下周一
    clk = clock_at(datetime(2026, 6, 6, 10, 0))
    d = decide_startup(clk, is_trading_day=days, standby_across_days=True)
    assert d.action is StartupAction.STANDBY
    assert d.trade_date == "20260608"
    assert d.standby_until == datetime(2026, 6, 8, 9, 30, tzinfo=SH)


def test_eod_constants_sane() -> None:
    # eod_flat_time 校验依赖的窗口常量不被误改。
    assert MORNING_OPEN == time(9, 30)
    assert AFTERNOON_CLOSE == time(15, 0)
