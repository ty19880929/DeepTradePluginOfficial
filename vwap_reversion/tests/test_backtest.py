"""backtest 回放（P3 验收项）：回放=实盘 parity + 多日聚合 + 报告。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from deeptrade.core.db import Database

from vwap_reversion.backtest import BacktestError, run_backtest
from vwap_reversion.clock import MarketClock
from vwap_reversion.config import VwrConfig
from vwap_reversion.daemon import CollectDaemon
from vwap_reversion.engine.vwap import BarMetrics
from vwap_reversion.persistence import TradeCalendarStore, get_run, upsert_bar
from vwap_reversion.reporting import build_backtest_report
from vwap_reversion.schemas import IntervalBar, Snapshot

SH = ZoneInfo("Asia/Shanghai")
MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "20260603_001_init.sql"
DAY1, DAY2 = "20260603", "20260604"

CFG_KW = dict(fee_bps=5.0, slippage_bps=10.0)


def sh_epoch(hh: int, mm: int, ss: int = 0, day: str = DAY1) -> float:
    return datetime(
        int(day[:4]), int(day[4:6]), int(day[6:8]), hh, mm, ss, tzinfo=SH
    ).timestamp()


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "bt.duckdb")
    sql = MIGRATION.read_text(encoding="utf-8")
    for stmt in sql.split(";"):
        if stmt.strip():
            database.execute(stmt)
    for d in (DAY1, DAY2):
        database.execute(
            "INSERT INTO vwr_trade_cal(exchange, cal_date, is_open, pretrade_date) "
            "VALUES ('SSE', ?, 1, '')", (d,),
        )
    yield database
    database.close()


# ---------------------------------------------------------------------------
# 工具：实盘 daemon（同 P2 集成测试款）与直接造 bar
# ---------------------------------------------------------------------------


class FakeClock(MarketClock):
    def __init__(self, start_epoch: float) -> None:
        self._epoch = start_epoch
        super().__init__("Asia/Shanghai", now_fn=lambda: self._epoch)

    def sleep(self, seconds: float) -> None:
        self._epoch += max(float(seconds), 0.001)


class ScriptedSource:
    def __init__(self, clock: FakeClock, price_fn) -> None:
        self.clock = clock
        self.price_fn = price_fn
        self._cum_vol = 0.0
        self._cum_amount = 0.0
        self._last_ts: float | None = None

    def fetch(self) -> Snapshot:
        now = self.clock.now_epoch()
        start = self._last_ts if self._last_ts is not None else sh_epoch(9, 30)
        price = self.price_fn(now)
        dvol = max(0.0, now - start) * 100.0
        self._cum_vol += dvol
        self._cum_amount += dvol * price
        self._last_ts = now
        return Snapshot(
            code="159518.SZ", trade_date=self.clock.today_str(), ts=int(now),
            last=price, cum_vol=self._cum_vol, cum_amount=self._cum_amount,
        )


class NullRenderer:
    def start(self, meta, clock) -> None: ...

    def handle(self, event) -> None: ...

    def finish(self) -> None: ...


def run_live(db: Database, price_fn):
    from vwap_reversion.runtime import VwrRuntime

    cfg = replace(VwrConfig(), poll_interval_seconds=60, **CFG_KW)
    clock = FakeClock(sh_epoch(9, 31))
    rt = VwrRuntime(db=db, config=None, clock=clock)  # type: ignore[arg-type]
    daemon = CollectDaemon(
        rt, cfg, source=ScriptedSource(clock, price_fn),
        calendar=TradeCalendarStore(db), renderer=NullRenderer(),
        sleep_fn=clock.sleep,
    )
    return daemon.execute(code="159518.SZ")


def dip_recover_price_fn(epoch: float) -> float:
    if sh_epoch(10, 30) <= epoch < sh_epoch(11, 0):
        return 0.99
    return 1.00


def seed_day_bars(db: Database, day: str, price_fn) -> None:
    """直接造 60s 间隔 bars（两段会话），与 ScriptedSource 同口径。"""
    cum_vol, cum_amount = 0.0, 0.0
    last_ts = sh_epoch(9, 30, day=day)
    dummy = BarMetrics(vwap=0, sigma=0, band_upper=0, band_lower=0, z=None)
    windows = [(sh_epoch(9, 31, day=day), sh_epoch(11, 30, day=day)),
               (sh_epoch(13, 0, day=day), sh_epoch(15, 0, day=day))]
    for lo, hi in windows:
        ts = lo
        while ts < hi:
            price = price_fn(ts)
            dvol = (ts - last_ts) * 100.0
            cum_vol += dvol
            cum_amount += dvol * price
            upsert_bar(db, IntervalBar(
                code="159518.SZ", trade_date=day, ts=int(ts), interval_vol=dvol,
                interval_amount=dvol * price, last=price, cum_vol=cum_vol,
                cum_amount=cum_amount,
            ), dummy)
            last_ts = ts
            ts += 60


def fetch_pairs(db: Database, run_id: str, table: str, cols: str):
    return db.fetchall(
        f"SELECT {cols} FROM {table} WHERE run_id = ? ORDER BY ts", (run_id,)
    )


# ---------------------------------------------------------------------------


def test_replay_equals_live_paper_run(db: Database) -> None:
    """⭐ 回放=实盘 parity：同日同参数，信号与成交逐笔一致（设计 §10.2 契约）。"""
    live = run_live(db, dip_recover_price_fn)
    assert live.status == "done"

    cfg = replace(VwrConfig(), **CFG_KW)
    bt = run_backtest(db, cfg, code="159518.SZ", start=DAY1, end=DAY1)

    live_sigs = fetch_pairs(db, live.run_id, "vwr_signals",
                            "ts, side, reason, suppressed_by, price")
    bt_sigs = fetch_pairs(db, bt.run_id, "vwr_signals",
                          "ts, side, reason, suppressed_by, price")
    assert live_sigs == bt_sigs and len(live_sigs) == 2

    live_trades = fetch_pairs(db, live.run_id, "vwr_trades",
                              "ts, side, qty, price, realized_pnl, position_after")
    bt_trades = fetch_pairs(db, bt.run_id, "vwr_trades",
                            "ts, side, qty, price, realized_pnl, position_after")
    assert live_trades == bt_trades and len(live_trades) == 2

    # 净盈亏也一致（汇总口径同源）
    live_summary = db.fetchone(
        "SELECT net_pnl FROM vwr_daily_summary WHERE trade_date = ?", (DAY1,))[0]
    assert bt.days[0]["net_pnl"] == pytest.approx(live_summary)


def test_multi_day_aggregate(db: Database) -> None:
    seed_day_bars(db, DAY1, dip_recover_price_fn)   # 盈利日（2 笔）
    seed_day_bars(db, DAY2, lambda _e: 1.00)        # 横盘日（0 笔）

    cfg = replace(VwrConfig(), **CFG_KW)
    bt = run_backtest(db, cfg, code="159518.SZ", start=DAY1, end=DAY2)

    agg = bt.aggregate
    assert agg["n_days"] == 2
    assert agg["n_trades"] == 2
    assert agg["net_pnl_total"] > 0
    assert agg["n_win_days"] == 1 and agg["day_win_rate"] == 0.5
    assert agg["worst_day_pnl"] == pytest.approx(0.0)
    assert agg["sharpe"] is not None
    assert agg["n_circuit_days"] == 0

    run = get_run(db, bt.run_id)
    assert run["mode"] == "backtest" and run["status"] == "done"
    assert run["trade_date"] == f"{DAY1}-{DAY2}"
    assert run["result_json"] and "aggregate" in run["result_json"]
    assert run["final_cash"] == pytest.approx(
        VwrConfig().initial_cash + agg["net_pnl_total"])

    # 回放绝不写 vwr_daily_summary（归 paper 实盘所有）
    assert db.fetchone("SELECT COUNT(*) FROM vwr_daily_summary")[0] == 0


def test_no_bars_raises(db: Database) -> None:
    cfg = replace(VwrConfig(), **CFG_KW)
    with pytest.raises(BacktestError, match="无已采集 bars"):
        run_backtest(db, cfg, code="159518.SZ", start=DAY1, end=DAY2)
    with pytest.raises(BacktestError, match="必须"):
        run_backtest(db, cfg, code="159518.SZ", start=DAY2, end=DAY1)


def test_backtest_report_content(db: Database) -> None:
    seed_day_bars(db, DAY1, dip_recover_price_fn)
    cfg = replace(VwrConfig(), **CFG_KW)
    bt = run_backtest(db, cfg, code="159518.SZ", start=DAY1, end=DAY1)
    md = build_backtest_report(db, bt.run_id)
    assert "回放回测报告" in md
    assert "聚合指标" in md and "逐日明细" in md
    assert DAY1 in md
    assert "总净盈亏" in md
