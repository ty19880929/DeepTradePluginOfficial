"""Window — resolved time-window for a single market-review run (design §3.2).

Three usage modes, all flowing through :func:`resolve_window`:

- **day**   — single trade_date (latest closed, or user-pinned)
- **range** — ``[start, end]`` closed interval of trade_dates, capped at
  ``MrConfig.max_window_days`` (default 60, hard ceiling 252).

Non-trading-day inputs are snapped backward to the most recent open day
(design §3.2 "非交易日自动向前 snap"); the caller surfaces a warning event
when the snap actually changed the date.

This module is pure (no DB / Tushare). The caller is expected to have
already populated ``mr_trade_cal`` via :func:`market_review.data.sync_calendar`
and constructed a :class:`TradeCalendar` from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .calendar import TradeCalendar

WindowMode = Literal["day", "range"]

# Design §3.2 — hard ceiling on the range mode. ``MrConfig.max_window_days``
# (default 60) is the user-visible cap; this is the safety net so a runaway
# config cannot produce 1000-day reviews.
HARD_MAX_WINDOW_DAYS = 252

_DATE_RE = re.compile(r"^\d{8}$")


class WindowSpecError(ValueError):
    """Raised when CLI inputs to :func:`resolve_window` are inconsistent.

    The CLI catches and renders this as a typer.BadParameter; tests assert
    on the message contents.
    """


@dataclass(frozen=True)
class Window:
    """Immutable description of the resolved trade-date span.

    ``trade_dates`` is the explicit enumeration of open days in the window —
    metrics modules iterate this rather than walking the calendar themselves.
    ``anchor`` is the date used for report titles and file naming: single-day
    runs use ``T``; range runs use ``T1`` (the last day).

    ``snapped_from`` carries the *original* user input when the resolver had
    to back-snap a non-trading day; ``None`` otherwise. Pipelines emit a
    one-shot WARN event when this is set.
    """

    mode: WindowMode
    start: str  # YYYYMMDD
    end: str    # YYYYMMDD
    trade_dates: tuple[str, ...]
    anchor: str
    snapped_from: tuple[str, str] | None = None  # (original_start, original_end)

    @property
    def n_days(self) -> int:
        return len(self.trade_dates)


def _validate_date(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise WindowSpecError(
            f"{label} 必须是 YYYYMMDD 格式字符串（8 位数字），收到：{value!r}"
        )
    return value


def resolve_window(
    calendar: TradeCalendar,
    *,
    trade_date: str | None = None,
    start: str | None = None,
    end: str | None = None,
    latest_trade_date: str | None = None,
    max_window_days: int = 60,
) -> Window:
    """Compute a :class:`Window` from CLI args + a calendar.

    Mode selection follows design §3.2:

    - All three of ``trade_date / start / end`` ``None`` → **day** mode anchored
      at ``latest_trade_date`` (caller must probe via
      :func:`market_review.data.fetch_latest_trade_date`).
    - ``trade_date`` set, ``start`` / ``end`` both ``None`` → **day** mode.
      Non-trading inputs snap backward.
    - ``start`` AND ``end`` set, ``trade_date`` ``None`` → **range** mode.
      Both ends snap backward; ``start > end`` raises.
    - Any other combination raises :class:`WindowSpecError`.

    Raises ``WindowSpecError`` for invalid inputs and ``ValueError`` when the
    calendar yields zero open days in the requested span.
    """
    if max_window_days <= 0:
        raise WindowSpecError(f"max_window_days 必须 > 0，收到 {max_window_days}")
    if max_window_days > HARD_MAX_WINDOW_DAYS:
        raise WindowSpecError(
            f"max_window_days={max_window_days} 超出硬上限 {HARD_MAX_WINDOW_DAYS}"
        )

    have_trade_date = trade_date is not None
    have_start = start is not None
    have_end = end is not None

    # --- Day mode: implicit (no args) -----------------------------------------
    if not have_trade_date and not have_start and not have_end:
        if not latest_trade_date:
            raise WindowSpecError(
                "未指定 --trade-date / --start / --end 时必须传入 latest_trade_date "
                "(由调用方先调 fetch_latest_trade_date 探测得到)"
            )
        anchor = _validate_date(latest_trade_date, label="latest_trade_date")
        if not calendar.is_open(anchor):
            # Probe returned a non-trading date — defensive snap.
            snapped = calendar.latest_closed_on_or_before(anchor)
            return Window(
                mode="day",
                start=snapped,
                end=snapped,
                trade_dates=(snapped,),
                anchor=snapped,
                snapped_from=(anchor, anchor),
            )
        return Window(
            mode="day",
            start=anchor,
            end=anchor,
            trade_dates=(anchor,),
            anchor=anchor,
        )

    # --- Day mode: explicit --trade-date --------------------------------------
    if have_trade_date and not have_start and not have_end:
        raw = _validate_date(trade_date, label="--trade-date")  # type: ignore[arg-type]
        snapped = calendar.latest_closed_on_or_before(raw)
        snap_meta = (raw, raw) if snapped != raw else None
        return Window(
            mode="day",
            start=snapped,
            end=snapped,
            trade_dates=(snapped,),
            anchor=snapped,
            snapped_from=snap_meta,
        )

    # --- Range mode -----------------------------------------------------------
    if have_start and have_end and not have_trade_date:
        raw_start = _validate_date(start, label="--start")  # type: ignore[arg-type]
        raw_end = _validate_date(end, label="--end")  # type: ignore[arg-type]
        if raw_start > raw_end:
            raise WindowSpecError(
                f"--start ({raw_start}) 不能晚于 --end ({raw_end})"
            )
        snapped_start = calendar.latest_closed_on_or_before(raw_start)
        snapped_end = calendar.latest_closed_on_or_before(raw_end)
        if snapped_start > snapped_end:
            raise ValueError(
                f"snap 后窗口为空：原始 [{raw_start}, {raw_end}] → "
                f"[{snapped_start}, {snapped_end}]"
            )
        dates = calendar.range(snapped_start, snapped_end)
        if not dates:
            raise ValueError(
                f"日历范围 [{snapped_start}, {snapped_end}] 内没有开市日，"
                "请检查 trade_cal 是否已同步或 start/end 是否过早"
            )
        if len(dates) > max_window_days:
            raise WindowSpecError(
                f"窗口跨度 {len(dates)} 个交易日超过 max_window_days={max_window_days}；"
                "请收缩 --start/--end 或先 `settings set max_window_days <N>`"
            )
        snap_meta = (
            (raw_start, raw_end)
            if (snapped_start, snapped_end) != (raw_start, raw_end)
            else None
        )
        # day mode short-circuit: a degenerate single-day range is conceptually
        # the same as ``--trade-date``; we keep mode=range so downstream code
        # that special-cases "range" (区间字段计算) still triggers.
        return Window(
            mode="range",
            start=snapped_start,
            end=snapped_end,
            trade_dates=tuple(dates),
            anchor=snapped_end,
            snapped_from=snap_meta,
        )

    # --- Mutually-exclusive flag violations -----------------------------------
    raise WindowSpecError(
        "--trade-date 与 --start/--end 互斥；区间模式需要同时指定 --start 与 --end。"
        f" 收到：trade_date={trade_date!r}, start={start!r}, end={end!r}"
    )
