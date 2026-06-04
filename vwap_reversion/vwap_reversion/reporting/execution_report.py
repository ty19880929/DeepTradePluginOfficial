"""执行报告（设计 §12.1）— 源：vwr_runs / vwr_events / vwr_signals / vwr_bars."""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from ..persistence import fetch_events, fetch_signals, get_run, load_bars
from .common import fmt_num, fmt_ts, market_tz_of, md_table, params_block

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

_MAX_ANOMALY_ROWS = 30


def build_execution_report(db: Database, run_id: str) -> str:
    run = get_run(db, run_id)
    if run is None:
        raise ValueError(f"run_id 不存在: {run_id!r}")
    tz = market_tz_of(run)
    code, trade_date = str(run["code"]), str(run["trade_date"])
    events = fetch_events(db, run_id)
    signals = fetch_signals(db, run_id)
    bars = load_bars(db, code, trade_date)

    # ---- 采样质量（事件 payload 投影）----
    samples = [e for e in events if _kind(e) == "sample"]
    n_samples = len(samples)
    n_no_volume = sum(1 for e in samples if "vwap" not in _payload(e))
    n_fetch_err = sum(1 for e in events if _kind(e) == "fetch_error")
    n_regression = sum(1 for e in events if _kind(e) == "regression")
    resumed = any(_kind(e) == "resume" for e in events)

    # ---- VWAP/σ 收敛快照：取当日第 1 / N/4 / N/2 / 最后 根 bar ----
    conv_rows: list[list[str]] = []
    if bars:
        picks = sorted({0, len(bars) // 4, len(bars) // 2, len(bars) - 1})
        bar_metric = _bar_metrics_map(db, code, trade_date)
        for i in picks:
            b = bars[i]
            vwap, sigma = bar_metric.get(b.ts, (None, None))
            conv_rows.append([
                fmt_ts(b.ts, tz), fmt_num(b.last, "{:.4f}"),
                fmt_num(vwap, "{:.4f}"), fmt_num(sigma, "{:.5f}"),
                fmt_num(b.cum_vol, "{:,.0f}"),
            ])

    # ---- 信号统计 ----
    executed = [s for s in signals if s["suppressed_by"] is None]
    suppressed = [s for s in signals if s["suppressed_by"] is not None]
    reason_dist = Counter(s["reason"] for s in executed)
    suppress_dist = Counter(str(s["suppressed_by"]).split(":")[0] for s in suppressed)

    # ---- 异常与降级 ----
    anomalies = [e for e in events if str(e["level"]) in ("warn", "error")]

    lines = [
        f"# vwap-reversion 执行报告 — {code} {trade_date}",
        "",
        f"- run_id: `{run_id}` ｜ mode: {run['mode']} ｜ status: **{run['status']}**",
        f"- 起止（DB 时钟）: {run['started_at']} → {run['finished_at'] or '—'}",
        f"- 市场时区: {tz.key}（下文时间均按此显示）",
        "",
        "## 参数快照",
        "",
        params_block(run),
        "",
        "## 采样质量",
        "",
        f"- 采样次数: **{n_samples}**（有效 bar {len(bars)} 根，无新成交跳过 {n_no_volume} 次）",
        f"- 行情拉取失败: {n_fetch_err} 次 ｜ 累计量回退丢弃: {n_regression} 次",
        f"- 崩溃恢复重放: {'是' if resumed else '否'}",
        "",
    ]
    if conv_rows:
        lines += [
            "### VWAP / σ 收敛快照",
            "",
            md_table(["时间", "last", "VWAP", "σ", "累计量(股)"], conv_rows),
            "",
        ]
    lines += [
        "## 信号统计",
        "",
        f"- 信号总数: **{len(signals)}**（执行 {len(executed)} ｜ 被抑制 {len(suppressed)}）",
    ]
    if reason_dist:
        lines.append("- 执行信号按类型: " + "、".join(
            f"`{r}`×{n}" for r, n in reason_dist.most_common()))
    if suppress_dist:
        lines.append("- 抑制原因分布: " + "、".join(
            f"`{r}`×{n}" for r, n in suppress_dist.most_common()))
    lines.append("")
    lines += ["## 异常与降级", ""]
    if not anomalies:
        lines.append("(无 WARN/ERROR 事件)")
    else:
        rows = [
            [fmt_ts(e["ts"], tz), str(e["level"]).upper(), _trunc(e["message"], 80)]
            for e in anomalies[:_MAX_ANOMALY_ROWS]
        ]
        lines.append(md_table(["时间", "级别", "消息"], rows))
        if len(anomalies) > _MAX_ANOMALY_ROWS:
            lines.append(f"\n（共 {len(anomalies)} 条，仅展示前 {_MAX_ANOMALY_ROWS} 条）")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(event.get("payload_json") or "{}")
    except json.JSONDecodeError:
        return {}


def _kind(event: dict[str, Any]) -> str | None:
    return _payload(event).get("kind")


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _bar_metrics_map(
    db: Database, code: str, trade_date: str
) -> dict[int, tuple[float, float]]:
    rows = db.fetchall(
        "SELECT ts, vwap, sigma FROM vwr_bars WHERE code = ? AND trade_date = ?",
        (code, trade_date),
    )
    return {int(r[0]): (r[1], r[2]) for r in rows}
