"""回放回测（设计 §10.2，P3）—— 读 vwr_bars，复用实盘同一套引擎与撮合.

「回放 = 实盘」契约：与 paper daemon 共用 :class:`VwapEngine` +
:class:`TradingSession` + Paper 撮合，**同 bar 以快照价成交**（滑点已建模）。
同一日同参数下，回放产生的信号与成交应与实盘逐笔一致（有 parity 测试钉住）。

每个交易日独立结算：每天一个新的 engine + session（initial_cash 重置），
日结果累加成聚合指标。signals/trades 以 backtest run_id 落库；**逐日汇总
不写 vwr_daily_summary**（那张表归 paper 实盘所有，PK code+trade_date 会
被回放覆盖），而是连同聚合一起写进 ``vwr_runs.result_json``。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

from .clock import MORNING_OPEN, MarketClock, parse_hhmm
from .engine.vwap import VwapEngine
from .persistence import (
    create_run,
    insert_signal,
    insert_trade,
    list_bar_dates,
    load_bars,
    set_run_result,
    update_run_status,
)
from .trading import TradingSession

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

    from .config import VwrConfig
    from .schemas import IntervalBar


class BacktestError(Exception):
    """回放前置条件不满足（窗口内无已采集 bars 等）。"""


@dataclass
class BacktestOutcome:
    run_id: str
    days: list[dict[str, Any]]
    aggregate: dict[str, Any]


def run_backtest(
    db: Database, cfg: VwrConfig, *, code: str, start: str, end: str
) -> BacktestOutcome:
    if start > end:
        raise BacktestError(f"--start {start} 必须 ≤ --end {end}")
    dates = list_bar_dates(db, code, start, end)
    if not dates:
        raise BacktestError(
            f"{code} 在 {start}..{end} 内无已采集 bars —— 先用 `run` 实盘采集"
            "（或等 P4 backfill）"
        )

    run_id = create_run(
        db, mode="backtest", code=code, trade_date=f"{start}-{end}",
        status="running", params=_params_snapshot(cfg, code, start, end),
        initial_cash=cfg.initial_cash,
    )
    clock = MarketClock(cfg.market_timezone)  # 只用墙钟→epoch 换算，不读「现在」
    trade_seq = 0
    day_rows: list[dict[str, Any]] = []
    try:
        for d in dates:
            session = replay_day(cfg, code, d, load_bars(db, code, d), clock)
            # 落库：信号 + 成交（seq 跨日连续）
            for sig in session.replayed_signals:
                insert_signal(db, run_id, sig)
            for tr in session.replayed_trades:
                trade_seq += 1
                insert_trade(db, run_id, trade_seq, tr)
            row = session.summary(run_id)
            day_rows.append(row)
        aggregate = _aggregate(day_rows, cfg.initial_cash)
        result = {
            "days": [
                {k: v for k, v in r.items() if k != "run_id"} for r in day_rows
            ],
            "aggregate": aggregate,
        }
        set_run_result(db, run_id, result)
        update_run_status(
            db, run_id, "done",
            final_cash=cfg.initial_cash + aggregate["net_pnl_total"], finished=True,
        )
        return BacktestOutcome(run_id, day_rows, aggregate)
    except Exception:
        update_run_status(db, run_id, "aborted", finished=True)
        raise


def replay_day(
    cfg: VwrConfig, code: str, trade_date: str,
    bars: list[IntervalBar], clock: MarketClock,
) -> TradingSession:
    """单日回放：与 daemon._collect_loop 同序（EOD 检查 → engine.push → on_bar）。

    回放产物挂在 session 上（``replayed_signals`` / ``replayed_trades``），
    由调用方决定怎么落库 —— session 本身保持与实盘完全同一份代码。
    """
    d = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))
    warmup_end = clock.epoch_of(d, MORNING_OPEN) + cfg.warmup_minutes * 60
    eod_epoch = clock.epoch_of(d, parse_hhmm(cfg.eod_flat_time))

    engine = VwapEngine(band_k=cfg.band_k_entry)
    session = TradingSession(cfg, code=code, trade_date=trade_date)
    signals: list[Any] = []
    trades: list[Any] = []

    def _collect(out: Any) -> None:
        signals.extend(out.signals)
        trades.extend(out.trades)

    for bar in bars:
        # 与 daemon 同序：先 EOD（实盘里 EOD 由同一时刻的快照触发，价格相同）
        if not session.eod_done and bar.ts >= eod_epoch:
            _collect(session.eod_flat(bar.last, bar.ts))
        metrics = engine.push(bar)
        _collect(session.on_bar(bar, metrics, in_warmup=bar.ts < warmup_end))
    # 兜底：当日 bars 在 EOD 前就断了（实盘中断/数据缺尾）→ 按最后价强平
    if not session.eod_done and session.last_price is not None and bars:
        _collect(session.eod_flat(session.last_price, bars[-1].ts))

    session.replayed_signals = signals  # type: ignore[attr-defined]
    session.replayed_trades = trades    # type: ignore[attr-defined]
    return session


# ---------------------------------------------------------------------------


def _aggregate(day_rows: list[dict[str, Any]], initial_cash: float) -> dict[str, Any]:
    day_pnls = [float(r["net_pnl"] or 0.0) for r in day_rows]
    win_days = [p for p in day_pnls if p > 0]
    sharpe = None
    if len(day_pnls) >= 2:
        std = statistics.pstdev(day_pnls)
        if std > 0:
            # 日收益率 = 日净盈亏/初始资金；年化 √252
            mean_r = statistics.mean(day_pnls) / initial_cash
            sharpe = mean_r / (std / initial_cash) * (252 ** 0.5)
    buy_holds = [r["buy_hold_pnl"] for r in day_rows if r["buy_hold_pnl"] is not None]
    return {
        "n_days": len(day_rows),
        "n_trades": sum(int(r["n_trades"] or 0) for r in day_rows),
        "net_pnl_total": sum(day_pnls),
        "n_win_days": len(win_days),
        "day_win_rate": (len(win_days) / len(day_pnls)) if day_pnls else None,
        "worst_day_pnl": min(day_pnls) if day_pnls else None,
        "sharpe": sharpe,
        "total_fee": sum(float(r["total_fee"] or 0.0) for r in day_rows),
        "total_slippage": sum(float(r["total_slippage"] or 0.0) for r in day_rows),
        "buy_hold_total": sum(buy_holds) if buy_holds else None,
        "n_circuit_days": sum(int(r["circuit_broken"] or 0) for r in day_rows),
    }


def _params_snapshot(cfg: VwrConfig, code: str, start: str, end: str) -> dict[str, Any]:
    from dataclasses import asdict  # noqa: PLC0415

    return {"code": code, "start": start, "end": end, **asdict(cfg)}
