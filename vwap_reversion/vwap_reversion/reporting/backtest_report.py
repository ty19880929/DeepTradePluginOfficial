"""backtest 报告 — 源：vwr_runs.result_json（由 backtest.run_backtest 写入）."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..persistence import get_run
from .common import fmt_num, md_table, params_block

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database


def build_backtest_report(db: Database, run_id: str) -> str:
    run = get_run(db, run_id)
    if run is None:
        raise ValueError(f"run_id 不存在: {run_id!r}")
    result: dict[str, Any] = json.loads(run.get("result_json") or "{}")
    agg: dict[str, Any] = result.get("aggregate", {})
    days: list[dict[str, Any]] = result.get("days", [])

    lines = [
        f"# vwap-reversion 回放回测报告 — {run['code']} {run['trade_date']}",
        "",
        f"- run_id: `{run_id}` ｜ mode: backtest ｜ status: **{run['status']}**",
        f"- 回放=实盘：与 paper daemon 共用同一 VwapEngine + TradingSession + Paper 撮合（同 bar 成交）",
        "",
        "## 参数快照",
        "",
        params_block(run),
        "",
        "## 聚合指标",
        "",
    ]
    if not agg:
        lines.append("(无聚合结果)")
    else:
        rows = [
            ["回放交易日数", str(agg.get("n_days", 0))],
            ["总成交笔数", str(agg.get("n_trades", 0))],
            ["总净盈亏", "**" + fmt_num(agg.get("net_pnl_total"), "{:+,.2f}") + " 元**"],
            ["盈利天数 / 胜率（按日）", f"{agg.get('n_win_days', 0)} / "
             + fmt_num(agg.get("day_win_rate"), "{:.0%}", none="—")],
            ["单日最差净盈亏", fmt_num(agg.get("worst_day_pnl"), "{:+,.2f}", none="—") + " 元"],
            ["日频 Sharpe（×√252）", fmt_num(agg.get("sharpe"), "{:.2f}", none="—（不足 2 日或零波动）")],
            ["总费用 / 滑点", f"{fmt_num(agg.get('total_fee'))} / {fmt_num(agg.get('total_slippage'))} 元"],
            ["buy-and-hold 基准合计", fmt_num(agg.get("buy_hold_total"), "{:+,.2f}", none="—") + " 元"],
            ["熔断触发天数", str(agg.get("n_circuit_days", 0))],
        ]
        lines.append(md_table(["指标", "值"], rows))
    lines += ["", "## 逐日明细", ""]
    if not days:
        lines.append("(无)")
    else:
        rows = [
            [
                str(d["trade_date"]), str(d["n_trades"]),
                fmt_num(d["win_rate"], "{:.0%}", none="—"),
                fmt_num(d["net_pnl"], "{:+,.2f}"),
                fmt_num(d["max_drawdown"], "{:.2f}") + "%",
                fmt_num(d["buy_hold_pnl"], "{:+,.2f}", none="—"),
                "⛔" if d.get("circuit_broken") else "",
            ]
            for d in days
        ]
        lines.append(md_table(
            ["交易日", "笔数", "胜率", "净盈亏(元)", "最大回撤", "buy&hold(元)", "熔断"],
            rows,
        ))
    lines.append("")
    return "\n".join(lines)
