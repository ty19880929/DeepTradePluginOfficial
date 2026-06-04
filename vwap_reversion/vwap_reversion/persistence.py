"""vwr_* 表读写（设计 §9）.

约定：
* ``vwr_snapshots`` / ``vwr_bars`` 按 (code, trade_date, ts) 主键、不绑 run，
  用 ``INSERT OR REPLACE`` 幂等写（重启重放同 ts 不报错）。
* ``vwr_runs.status`` 状态机：standby → running → done；任何阶段中断 → aborted。
* ``vwr_trade_cal`` 是 tushare trade_cal 的插件本地缓存；查不到的日期视为
  「日历未覆盖」并抛错（宁可 fail-fast 也不把节假日当交易日跑）。
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from .schemas import IntervalBar, Signal, Snapshot, Trade

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient

    from .engine.vwap import BarMetrics


# ---------------------------------------------------------------------------
# vwr_runs
# ---------------------------------------------------------------------------


def create_run(
    db: Database,
    *,
    mode: str,
    code: str,
    trade_date: str,
    status: str,
    params: dict[str, Any],
    initial_cash: float,
) -> str:
    run_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO vwr_runs(run_id, mode, code, trade_date, status, params_json, "
        "initial_cash) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, mode, code, trade_date, status, json.dumps(params, ensure_ascii=False),
         initial_cash),
    )
    return run_id


def update_run_status(
    db: Database,
    run_id: str,
    status: str,
    *,
    final_cash: float | None = None,
    finished: bool = False,
) -> None:
    sets = ["status = ?"]
    params: list[Any] = [status]
    if final_cash is not None:
        sets.append("final_cash = ?")
        params.append(final_cash)
    if finished:
        sets.append("finished_at = CURRENT_TIMESTAMP")
    params.append(run_id)
    db.execute(f"UPDATE vwr_runs SET {', '.join(sets)} WHERE run_id = ?", tuple(params))


def set_run_report_dir(db: Database, run_id: str, report_dir: str) -> None:
    db.execute("UPDATE vwr_runs SET report_dir = ? WHERE run_id = ?", (report_dir, run_id))


def set_run_result(db: Database, run_id: str, result: dict[str, Any]) -> None:
    db.execute(
        "UPDATE vwr_runs SET result_json = ? WHERE run_id = ?",
        (json.dumps(result, ensure_ascii=False, default=str), run_id),
    )


_RUN_COLS = (
    "run_id", "mode", "code", "trade_date", "status", "params_json", "initial_cash",
    "final_cash", "report_dir", "result_json", "started_at", "finished_at",
)


def get_run(db: Database, run_id: str) -> dict[str, Any] | None:
    row = db.fetchone(
        f"SELECT {', '.join(_RUN_COLS)} FROM vwr_runs WHERE run_id = ?", (run_id,)
    )
    return None if row is None else dict(zip(_RUN_COLS, row))


def latest_run_id(db: Database, *, mode: str | None = None) -> str | None:
    clause, params = ("WHERE mode = ? ", (mode,)) if mode else ("", ())
    row = db.fetchone(
        f"SELECT run_id FROM vwr_runs {clause}ORDER BY started_at DESC LIMIT 1", params
    )
    return None if row is None else row[0]


# ---------------------------------------------------------------------------
# vwr_events
# ---------------------------------------------------------------------------


class EventWriter:
    """每个 run 一个实例：维护 seq 自增，把 StrategyEvent 落 vwr_events。"""

    def __init__(self, db: Database, run_id: str) -> None:
        self._db = db
        self._run_id = run_id
        self._seq = 0

    def write(self, ts: int, event_type: str, level: str, message: str,
              payload: dict[str, Any] | None = None) -> None:
        self._seq += 1
        self._db.execute(
            "INSERT INTO vwr_events(run_id, seq, ts, event_type, level, message, "
            "payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._run_id, self._seq, ts, event_type, level, message,
             json.dumps(payload or {}, ensure_ascii=False, default=str)),
        )


# ---------------------------------------------------------------------------
# vwr_snapshots / vwr_bars
# ---------------------------------------------------------------------------


def upsert_snapshot(db: Database, snap: Snapshot, *, source: str = "realtime") -> None:
    db.execute(
        "INSERT OR REPLACE INTO vwr_snapshots(code, trade_date, ts, last, cum_vol, "
        "cum_amount, num_trades, pre_close, open, high, low, bid_volume1, ask_volume1, "
        "trade_time, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (snap.code, snap.trade_date, snap.ts, snap.last, snap.cum_vol, snap.cum_amount,
         snap.num_trades, snap.pre_close, snap.open, snap.high, snap.low,
         snap.bid_volume1, snap.ask_volume1, snap.trade_time, source),
    )


def last_snapshot(db: Database, code: str, trade_date: str) -> Snapshot | None:
    """崩溃恢复：取当日最后一条快照作为 BarBuilder 差分基线。"""
    row = db.fetchone(
        "SELECT code, trade_date, ts, last, cum_vol, cum_amount, num_trades, pre_close, "
        "open, high, low, bid_volume1, ask_volume1, trade_time "
        "FROM vwr_snapshots WHERE code = ? AND trade_date = ? ORDER BY ts DESC LIMIT 1",
        (code, trade_date),
    )
    if row is None:
        return None
    return Snapshot(
        code=row[0], trade_date=row[1], ts=int(row[2]), last=row[3], cum_vol=row[4],
        cum_amount=row[5], num_trades=row[6], pre_close=row[7], open=row[8],
        high=row[9], low=row[10], bid_volume1=row[11], ask_volume1=row[12],
        trade_time=row[13],
    )


def upsert_bar(
    db: Database,
    bar: IntervalBar,
    metrics: BarMetrics,
    *,
    source: str = "realtime",
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO vwr_bars(code, trade_date, ts, interval_vol, "
        "interval_amount, last, cum_vol, cum_amount, vwap, sigma, band_upper, "
        "band_lower, z, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (bar.code, bar.trade_date, bar.ts, bar.interval_vol, bar.interval_amount,
         bar.last, bar.cum_vol, bar.cum_amount, metrics.vwap, metrics.sigma,
         metrics.band_upper, metrics.band_lower, metrics.z, source),
    )


def list_bar_dates(db: Database, code: str, start: str, end: str) -> list[str]:
    """[start, end] 闭区间内有已采集 bars 的交易日（升序）—— backtest 的回放范围。"""
    rows = db.fetchall(
        "SELECT DISTINCT trade_date FROM vwr_bars WHERE code = ? AND trade_date "
        "BETWEEN ? AND ? ORDER BY trade_date",
        (code, start, end),
    )
    return [r[0] for r in rows]


def load_bars(db: Database, code: str, trade_date: str) -> list[IntervalBar]:
    """按 ts 升序载入当日已落库 bars（崩溃恢复重建引擎 / 回放回测）。"""
    rows = db.fetchall(
        "SELECT code, trade_date, ts, interval_vol, interval_amount, last, cum_vol, "
        "cum_amount FROM vwr_bars WHERE code = ? AND trade_date = ? ORDER BY ts",
        (code, trade_date),
    )
    return [
        IntervalBar(
            code=r[0], trade_date=r[1], ts=int(r[2]), interval_vol=r[3],
            interval_amount=r[4], last=r[5], cum_vol=r[6], cum_amount=r[7],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# vwr_signals / vwr_trades / vwr_daily_summary（P2）
# ---------------------------------------------------------------------------


def insert_signal(db: Database, run_id: str, sig: Signal) -> None:
    # PK (run_id, ts)：每根 bar 至多一个信号；OR REPLACE 兜重放幂等。
    db.execute(
        "INSERT OR REPLACE INTO vwr_signals(run_id, ts, side, z, vwap, sigma, price, "
        "reason, suppressed_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, sig.ts, sig.side.value, sig.z, sig.vwap, sig.sigma, sig.price,
         sig.reason, sig.suppressed_by),
    )


def insert_trade(db: Database, run_id: str, seq: int, trade: Trade) -> None:
    db.execute(
        "INSERT INTO vwr_trades(run_id, seq, ts, side, qty, price, fee, slippage, "
        "realized_pnl, cash_after, position_after) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, seq, trade.ts, trade.side.value, trade.qty, trade.price, trade.fee,
         trade.slippage, trade.realized_pnl, trade.cash_after, trade.position_after),
    )


def fetch_signals(db: Database, run_id: str) -> list[dict[str, Any]]:
    cols = ("ts", "side", "z", "vwap", "sigma", "price", "reason", "suppressed_by")
    rows = db.fetchall(
        f"SELECT {', '.join(cols)} FROM vwr_signals WHERE run_id = ? ORDER BY ts",
        (run_id,),
    )
    return [dict(zip(cols, r)) for r in rows]


def fetch_trades(db: Database, run_id: str) -> list[dict[str, Any]]:
    cols = ("seq", "ts", "side", "qty", "price", "fee", "slippage", "realized_pnl",
            "cash_after", "position_after")
    rows = db.fetchall(
        f"SELECT {', '.join(cols)} FROM vwr_trades WHERE run_id = ? ORDER BY seq",
        (run_id,),
    )
    return [dict(zip(cols, r)) for r in rows]


def fetch_events(db: Database, run_id: str) -> list[dict[str, Any]]:
    cols = ("seq", "ts", "event_type", "level", "message", "payload_json")
    rows = db.fetchall(
        f"SELECT {', '.join(cols)} FROM vwr_events WHERE run_id = ? ORDER BY seq",
        (run_id,),
    )
    return [dict(zip(cols, r)) for r in rows]


_SUMMARY_COLS = (
    "code", "trade_date", "run_id", "n_trades", "n_wins", "win_rate", "profit_factor",
    "gross_pnl", "net_pnl", "total_fee", "total_slippage", "turnover", "max_drawdown",
    "avg_holding_seconds", "final_cash", "buy_hold_pnl", "circuit_broken",
)


def fetch_daily_summary(db: Database, code: str, trade_date: str) -> dict[str, Any] | None:
    row = db.fetchone(
        f"SELECT {', '.join(_SUMMARY_COLS)} FROM vwr_daily_summary "
        "WHERE code = ? AND trade_date = ?",
        (code, trade_date),
    )
    return None if row is None else dict(zip(_SUMMARY_COLS, row))


def upsert_daily_summary(db: Database, row: dict[str, Any]) -> None:
    """写当日交易汇总（PK code+trade_date；同日重跑覆盖为最新 run 的结果）。"""
    db.execute(
        f"INSERT OR REPLACE INTO vwr_daily_summary({', '.join(_SUMMARY_COLS)}) "
        f"VALUES ({', '.join('?' for _ in _SUMMARY_COLS)})",
        tuple(row.get(c) for c in _SUMMARY_COLS),
    )


# ---------------------------------------------------------------------------
# vwr_trade_cal
# ---------------------------------------------------------------------------


class TradeCalendarStore:
    """vwr_trade_cal 缓存 + is_trading_day 谓词（clock.decide_startup 的输入）。"""

    EXCHANGE = "SSE"

    def __init__(self, db: Database) -> None:
        self._db = db
        self._memo: dict[str, bool] = {}

    def ensure_synced(self, tushare: TushareClient) -> int:
        """全量同步 trade_cal（框架缓存 static 语义，开销一次性）。返回行数。"""
        df = tushare.call("trade_cal", params={"exchange": self.EXCHANGE})
        if df is None or df.empty:
            raise RuntimeError("trade_cal 返回空表，无法建立交易日历")
        with self._db.transaction():
            for _, row in df.iterrows():
                self._db.execute(
                    "INSERT OR REPLACE INTO vwr_trade_cal(exchange, cal_date, is_open, "
                    "pretrade_date) VALUES (?, ?, ?, ?)",
                    (str(row.get("exchange", self.EXCHANGE)), str(row["cal_date"]),
                     int(row["is_open"]), str(row.get("pretrade_date") or "")),
                )
        self._memo.clear()
        return len(df)

    def covers(self, yyyymmdd: str) -> bool:
        row = self._db.fetchone(
            "SELECT 1 FROM vwr_trade_cal WHERE exchange = ? AND cal_date = ?",
            (self.EXCHANGE, yyyymmdd),
        )
        return row is not None

    def is_trading_day(self, d: date) -> bool:
        key = d.strftime("%Y%m%d")
        if key in self._memo:
            return self._memo[key]
        row = self._db.fetchone(
            "SELECT is_open FROM vwr_trade_cal WHERE exchange = ? AND cal_date = ?",
            (self.EXCHANGE, key),
        )
        if row is None:
            raise RuntimeError(
                f"交易日历未覆盖 {key}（vwr_trade_cal 无此行）；请先联网跑一次 run "
                "以同步 trade_cal"
            )
        result = bool(row[0])
        self._memo[key] = result
        return result
