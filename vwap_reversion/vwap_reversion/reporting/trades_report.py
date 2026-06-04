"""交易汇总报告（设计 §12.2）— 源：vwr_daily_summary + vwr_trades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..persistence import fetch_daily_summary, fetch_trades, get_run
from .common import fmt_num, fmt_ts, market_tz_of, md_table

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


def build_trades_report(db: Database, run_id: str) -> str:
    run = get_run(db, run_id)
    if run is None:
        raise ValueError(f"run_id 不存在: {run_id!r}")
    tz = market_tz_of(run)
    code, trade_date = str(run["code"]), str(run["trade_date"])
    summary = fetch_daily_summary(db, code, trade_date)
    trades = fetch_trades(db, run_id)

    lines = [
        f"# vwap-reversion 交易汇总报告 — {code} {trade_date}",
        "",
        f"- run_id: `{run_id}` ｜ status: **{run['status']}**",
        "",
        "## 当日汇总",
        "",
    ]
    if summary is None:
        lines.append("(无汇总行 —— run 可能被中断，未走到收盘收尾)")
    else:
        s = summary
        rows = [
            ["成交笔数", str(s["n_trades"])],
            ["胜率", fmt_num(s["win_rate"], "{:.0%}") if s["win_rate"] is not None else "—（无平仓腿）"],
            ["盈亏比 (profit factor)", fmt_num(s["profit_factor"], "{:.2f}", none="—（无亏损腿）")],
            ["毛盈亏（已平腿）", fmt_num(s["gross_pnl"]) + " 元"],
            ["净盈亏（含浮动、扣费后）", "**" + fmt_num(s["net_pnl"], "{:+,.2f}") + " 元**"],
            ["总费用 / 滑点成本", f"{fmt_num(s['total_fee'])} / {fmt_num(s['total_slippage'])} 元"],
            ["换手（Σ成交额）", fmt_num(s["turnover"]) + " 元"],
            ["日内最大回撤", fmt_num(s["max_drawdown"], "{:.2f}") + " %（峰值权益）"],
            ["平均持仓时长", fmt_num(s["avg_holding_seconds"], "{:.0f}", none="—") + " 秒"],
            ["期末权益", fmt_num(s["final_cash"]) + " 元"],
            ["buy-and-hold 基准", fmt_num(s["buy_hold_pnl"], "{:+,.2f}", none="—") + " 元"],
            ["日亏熔断", "⛔ 触发" if s["circuit_broken"] else "未触发"],
        ]
        lines.append(md_table(["指标", "值"], rows))
    lines += ["", "## 成交明细", ""]
    if not trades:
        lines.append("(当日无成交)")
    else:
        rows = [
            [
                str(t["seq"]), fmt_ts(t["ts"], tz), str(t["side"]).upper(),
                fmt_num(t["qty"], "{:,.0f}"), fmt_num(t["price"], "{:.4f}"),
                fmt_num(t["fee"]), fmt_num(t["realized_pnl"], "{:+.2f}"),
                fmt_num(t["position_after"], "{:,.0f}"),
            ]
            for t in trades
        ]
        lines.append(md_table(
            ["#", "时间", "方向", "数量", "价格", "费用", "已实现盈亏", "持仓后"], rows
        ))
    lines.append("")
    return "\n".join(lines)
