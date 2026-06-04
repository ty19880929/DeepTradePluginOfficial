"""MarketClock — 市场时区 + 会话窗口 + 启动时机决策（设计 §4）.

本机时区可能不是中国时区，因此所有「交易日 / 开盘 / 收盘 / 午休 / EOD」判断
一律以市场时区（默认 ``Asia/Shanghai``）计算，绝不读本机本地时区：

* "今天"   = ``now(UTC).astimezone(market_tz).date()``，而非 ``date.today()``。
* 时间戳   = 全部存 epoch 秒（UTC 基准，无歧义）；展示时再转市场时区。
* sleep    = 目标市场墙钟时刻先转 epoch，再与 ``time.time()`` 求差。

纯 stdlib 模块（zoneinfo + datetime），不依赖 deeptrade —— 这样
``validate_static`` 可以安全 import，单测也无需框架环境。

启动时机决策（§4.3）见 :func:`decide_startup`：开盘前 → STANDBY（待机到
09:30 自动开跑）；盘中/午休 → RUN_NOW；收盘后/非交易日 → EXIT（除非
``standby_across_days=True`` 则待机到下一交易日开盘）。
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Callable
from zoneinfo import ZoneInfo

DEFAULT_MARKET_TZ = "Asia/Shanghai"

# A 股连续竞价会话窗口（市场墙钟）。集合竞价(09:15–09:25)的量计入 09:30 首快照。
MORNING_OPEN = time(9, 30)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
AFTERNOON_CLOSE = time(15, 0)


class SessionPhase(Enum):
    """某一市场墙钟时刻落在当日会话结构的哪一段（不含交易日判断）。"""

    PRE_OPEN = "pre_open"          # 00:00 – 09:30
    MORNING = "morning"            # 09:30 – 11:30
    LUNCH = "lunch"                # 11:30 – 13:00
    AFTERNOON = "afternoon"        # 13:00 – 15:00
    AFTER_CLOSE = "after_close"    # 15:00 – 24:00


class StartupAction(Enum):
    RUN_NOW = "run_now"            # 盘中（含午休）：立即进入运行
    STANDBY = "standby"            # 开盘前：待机到 standby_until 自动开跑
    EXIT = "exit"                  # 今日已无会话且不跨夜守候：退出并提示


@dataclass(frozen=True)
class StartupDecision:
    action: StartupAction
    # STANDBY 时的目标开盘时刻（market-tz aware datetime）；其余为 None。
    standby_until: datetime | None
    # 决策针对的交易日（YYYYMMDD）；EXIT 时为建议的下一交易日（可能为 None）。
    trade_date: str | None
    reason: str

    @property
    def standby_until_epoch(self) -> float | None:
        return None if self.standby_until is None else self.standby_until.timestamp()


def classify_phase(t: time) -> SessionPhase:
    """把市场墙钟时刻归类到会话段。边界遵循左闭右开：09:30 整即 MORNING。"""
    if t < MORNING_OPEN:
        return SessionPhase.PRE_OPEN
    if t < MORNING_CLOSE:
        return SessionPhase.MORNING
    if t < AFTERNOON_OPEN:
        return SessionPhase.LUNCH
    if t < AFTERNOON_CLOSE:
        return SessionPhase.AFTERNOON
    return SessionPhase.AFTER_CLOSE


def in_continuous_session(t: time) -> bool:
    return classify_phase(t) in (SessionPhase.MORNING, SessionPhase.AFTERNOON)


class MarketClock:
    """市场时区时钟。所有插件代码经由它取「现在」，禁止散落 ``datetime.now()``。

    ``now_fn`` 注入点仅供测试（返回 epoch 秒）；生产默认 ``time.time``。
    """

    def __init__(
        self,
        market_timezone: str = DEFAULT_MARKET_TZ,
        *,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.tz = ZoneInfo(market_timezone)
        self._now_fn = now_fn or _time.time

    # ---- 基础读数 -------------------------------------------------------

    def now_epoch(self) -> float:
        return self._now_fn()

    def now(self) -> datetime:
        """市场时区的 aware datetime。"""
        return datetime.fromtimestamp(self._now_fn(), tz=timezone.utc).astimezone(self.tz)

    def today(self) -> date:
        return self.now().date()

    def today_str(self) -> str:
        return self.today().strftime("%Y%m%d")

    def phase(self) -> SessionPhase:
        return classify_phase(self.now().time())

    # ---- 墙钟 ↔ epoch ---------------------------------------------------

    def wall(self, d: date, t: time) -> datetime:
        """市场时区某日某墙钟时刻的 aware datetime。"""
        return datetime.combine(d, t, tzinfo=self.tz)

    def epoch_of(self, d: date, t: time) -> float:
        return self.wall(d, t).timestamp()

    def seconds_until(self, target: datetime) -> float:
        """now → target 的秒数（target 须为 aware）。负数表示已过。"""
        return target.timestamp() - self._now_fn()


def parse_hhmm(s: str) -> time:
    """'14:55' → time(14, 55)。供 eod_flat_time 等配置解析。"""
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"无法解析 HH:MM 时刻: {s!r}")
    return time(int(parts[0]), int(parts[1]))


def next_trading_day(
    after: date,
    is_trading_day: Callable[[date], bool],
    *,
    max_lookahead_days: int = 30,
) -> date | None:
    """*after* 之后（不含当日）的下一个交易日；30 天内找不到返回 None
    （正常日历不会发生；防 trade_cal 缓存缺失时死循环）。"""
    d = after
    for _ in range(max_lookahead_days):
        d = d + timedelta(days=1)
        if is_trading_day(d):
            return d
    return None


def decide_startup(
    clock: MarketClock,
    *,
    is_trading_day: Callable[[date], bool],
    standby_across_days: bool = False,
) -> StartupDecision:
    """启动时机决策（设计 §4.3）。

    ============================  =====================================
    启动时点（市场时区）           行为
    ============================  =====================================
    交易日，未到 09:30             STANDBY → 待机到今日 09:30
    交易日，09:30–15:00（含午休）  RUN_NOW（午休由 daemon 自行 sleep）
    交易日，已过 15:00             EXIT；standby_across_days → 下一交易日
    非交易日                       EXIT；standby_across_days → 下一交易日
    ============================  =====================================
    """
    now = clock.now()
    today = now.date()
    phase = classify_phase(now.time())

    if is_trading_day(today):
        if phase is SessionPhase.PRE_OPEN:
            target = clock.wall(today, MORNING_OPEN)
            return StartupDecision(
                action=StartupAction.STANDBY,
                standby_until=target,
                trade_date=today.strftime("%Y%m%d"),
                reason=f"未到开盘，待机到 {target.strftime('%H:%M')}（市场时区）",
            )
        if phase in (SessionPhase.MORNING, SessionPhase.LUNCH, SessionPhase.AFTERNOON):
            return StartupDecision(
                action=StartupAction.RUN_NOW,
                standby_until=None,
                trade_date=today.strftime("%Y%m%d"),
                reason="盘中启动，立即运行",
            )
        # AFTER_CLOSE — 今日会话已结束。
        return _no_session_today(clock, today, is_trading_day, standby_across_days, "今日已收盘")

    return _no_session_today(clock, today, is_trading_day, standby_across_days, "今日非交易日")


def _no_session_today(
    clock: MarketClock,
    today: date,
    is_trading_day: Callable[[date], bool],
    standby_across_days: bool,
    why: str,
) -> StartupDecision:
    nxt = next_trading_day(today, is_trading_day)
    nxt_str = nxt.strftime("%Y%m%d") if nxt is not None else None
    if standby_across_days and nxt is not None:
        target = clock.wall(nxt, MORNING_OPEN)
        return StartupDecision(
            action=StartupAction.STANDBY,
            standby_until=target,
            trade_date=nxt_str,
            reason=f"{why}，跨日待机到 {nxt_str} 09:30（standby_across_days=true）",
        )
    return StartupDecision(
        action=StartupAction.EXIT,
        standby_until=None,
        trade_date=nxt_str,
        reason=(
            f"{why}，退出（下一交易日 {nxt_str or '未知'}；"
            "如需守候到下一开盘请设 standby_across_days=true）"
        ),
    )
