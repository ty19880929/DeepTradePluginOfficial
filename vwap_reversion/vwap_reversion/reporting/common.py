"""报告共用小件：上海时间格式化 / params 摘要 / markdown 表格。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..clock import DEFAULT_MARKET_TZ


def market_tz_of(run: dict[str, Any]) -> ZoneInfo:
    """从 run 的 params 快照取市场时区（报告展示一律用它，设计 §4）。"""
    try:
        params = json.loads(run.get("params_json") or "{}")
        return ZoneInfo(str(params.get("market_timezone", DEFAULT_MARKET_TZ)))
    except Exception:  # noqa: BLE001 — 报告永不因脏参数崩
        return ZoneInfo(DEFAULT_MARKET_TZ)


def fmt_ts(ts: Any, tz: ZoneInfo) -> str:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(tz).strftime(
            "%H:%M:%S"
        )
    except (TypeError, ValueError, OSError):
        return "—"


def fmt_num(v: Any, fmt: str = "{:,.2f}", none: str = "—") -> str:
    if v is None:
        return none
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def params_block(run: dict[str, Any]) -> str:
    try:
        params = json.loads(run.get("params_json") or "{}")
    except json.JSONDecodeError:
        return "(参数快照损坏)"
    keys = (
        "code", "market_timezone", "poll_interval_seconds", "band_mode",
        "band_k_entry", "band_k_exit", "band_k_stop", "warmup_minutes",
        "position_mode", "base_shares", "order_qty", "max_trades_per_day",
        "min_holding_seconds", "cooldown_seconds", "per_trade_stop_pct",
        "daily_loss_limit_pct", "eod_flat_time", "initial_cash", "fee_bps",
        "slippage_bps",
    )
    lines = [f"- `{k}` = {params[k]}" for k in keys if k in params]
    return "\n".join(lines) if lines else "(无)"
