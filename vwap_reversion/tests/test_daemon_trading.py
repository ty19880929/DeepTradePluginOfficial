"""daemon × 交易链路 集成测试（P2 验收：日内完整模拟交易闭环）.

ScriptedSource 按上海墙钟脚本化价格路径，引擎自然算出 z 触发交易：
低吸入场 → 回归止盈 / EOD 强平，全链路落库（signals/trades/daily_summary）。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from deeptrade.core.db import Database

from vwap_reversion.clock import MarketClock
from vwap_reversion.config import VwrConfig
from vwap_reversion.daemon import CollectDaemon
from vwap_reversion.persistence import TradeCalendarStore
from vwap_reversion.schemas import Snapshot

SH = ZoneInfo("Asia/Shanghai")
MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "20260603_001_init.sql"
DAY = "20260603"


def sh_epoch(hh: int, mm: int, ss: int = 0) -> float:
    return datetime(2026, 6, 3, hh, mm, ss, tzinfo=SH).timestamp()


class FakeClock(MarketClock):
    def __init__(self, start_epoch: float) -> None:
        self._epoch = start_epoch
        super().__init__("Asia/Shanghai", now_fn=lambda: self._epoch)

    def sleep(self, seconds: float) -> None:
        self._epoch += max(float(seconds), 0.001)


class ScriptedSource:
    """价格 = price_fn(epoch)；成交量 100 股/秒 线性累积。"""

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
        dvol = max(0.0, (now - start)) * 100.0
        self._cum_vol += dvol
        self._cum_amount += dvol * price
        self._last_ts = now
        return Snapshot(
            code="159518.SZ", trade_date=self.clock.today_str(), ts=int(now),
            last=price, cum_vol=self._cum_vol, cum_amount=self._cum_amount,
        )


class CaptureRenderer:
    def __init__(self) -> None:
        self.events = []

    def start(self, meta, clock) -> None: ...

    def handle(self, event) -> None:
        self.events.append(event)

    def finish(self) -> None: ...


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "t.duckdb")
    sql = MIGRATION.read_text(encoding="utf-8")
    for stmt in sql.split(";"):
        if stmt.strip():
            database.execute(stmt)
    database.execute(
        "INSERT INTO vwr_trade_cal(exchange, cal_date, is_open, pretrade_date) "
        "VALUES ('SSE', ?, 1, '')", (DAY,),
    )
    yield database
    database.close()


def run_daemon(db: Database, start_epoch: float, price_fn, **cfg_kw):
    from vwap_reversion.runtime import VwrRuntime

    cfg = replace(
        VwrConfig(), poll_interval_seconds=60, fee_bps=5.0, slippage_bps=10.0,
        **cfg_kw,
    )
    clock = FakeClock(start_epoch)
    rt = VwrRuntime(db=db, config=None, clock=clock)  # type: ignore[arg-type]
    renderer = CaptureRenderer()
    daemon = CollectDaemon(
        rt, cfg, source=ScriptedSource(clock, price_fn),
        calendar=TradeCalendarStore(db), renderer=renderer, sleep_fn=clock.sleep,
    )
    return daemon.execute(code="159518.SZ"), renderer


def trades_of(db: Database, run_id: str):
    return db.fetchall(
        "SELECT seq, side, qty, price, realized_pnl, position_after FROM vwr_trades "
        "WHERE run_id = ? ORDER BY seq", (run_id,),
    )


# ---------------------------------------------------------------------------


def test_dip_and_recover_full_round_trip(db: Database) -> None:
    """1.00 → 10:30 跌到 0.99 → 11:00 收复：应恰好一次低吸 + 一次回归止盈。"""
    dip_lo, dip_hi = sh_epoch(10, 30), sh_epoch(11, 0)

    def price_fn(epoch: float) -> float:
        return 0.99 if dip_lo <= epoch < dip_hi else 1.00

    outcome, renderer = run_daemon(db, sh_epoch(9, 31), price_fn)
    assert outcome.status == "done"

    trades = trades_of(db, outcome.run_id)
    assert len(trades) == 2
    (s1, side1, qty1, px1, pnl1, pos1), (s2, side2, qty2, px2, pnl2, pos2) = trades
    assert (side1, qty1, pos1) == ("buy", 100, 100)
    assert px1 == pytest.approx(0.99 * 1.001)     # 低吸 @0.99 含滑点
    assert (side2, pos2) == ("sell", 0)
    assert pnl2 > 0                               # 回归止盈为正

    # 信号链：entry_long → revert_exit，均未被抑制
    sig_reasons = [r[0] for r in db.fetchall(
        "SELECT reason FROM vwr_signals WHERE run_id = ? AND suppressed_by IS NULL "
        "ORDER BY ts", (outcome.run_id,),
    )]
    assert sig_reasons == ["entry_long", "revert_exit"]

    # 入场时刻在跌段内、出场在收复后
    entry_ts = db.fetchone(
        "SELECT ts FROM vwr_signals WHERE run_id=? AND reason='entry_long'",
        (outcome.run_id,))[0]
    exit_ts = db.fetchone(
        "SELECT ts FROM vwr_signals WHERE run_id=? AND reason='revert_exit'",
        (outcome.run_id,))[0]
    assert dip_lo <= entry_ts < dip_hi <= exit_ts

    # 当日汇总落库 + run 期末权益 = 初始 + 净盈亏
    row = db.fetchone(
        "SELECT n_trades, n_wins, win_rate, net_pnl, final_cash, circuit_broken "
        "FROM vwr_daily_summary WHERE code='159518.SZ' AND trade_date=?", (DAY,),
    )
    n_trades, n_wins, win_rate, net_pnl, final_cash, circuit = row
    assert n_trades == 2 and n_wins == 1 and win_rate == 1.0 and circuit == 0
    assert net_pnl > 0
    run_final = db.fetchone(
        "SELECT final_cash FROM vwr_runs WHERE run_id=?", (outcome.run_id,))[0]
    assert run_final == pytest.approx(final_cash)
    assert run_final == pytest.approx(VwrConfig().initial_cash + net_pnl)

    # 实时面板拿到了 trade 事件（交易记录面板数据源）
    trade_events = [e for e in renderer.events if e.payload.get("kind") == "trade"]
    assert len(trade_events) == 2
    assert all("net_pnl" in e.payload for e in trade_events)


def test_eod_force_flat_when_dip_never_recovers(db: Database) -> None:
    """14:50 跌不收复 → 入场后由 14:55 EOD 强平，绝不带腿过夜。"""
    dip = sh_epoch(14, 50)

    def price_fn(epoch: float) -> float:
        return 0.99 if epoch >= dip else 1.00

    outcome, renderer = run_daemon(db, sh_epoch(14, 0), price_fn)
    assert outcome.status == "done"

    trades = trades_of(db, outcome.run_id)
    assert len(trades) == 2
    assert trades[0][1] == "buy" and trades[1][1] == "sell"
    assert trades[1][5] == 0                       # 强平后持仓归零
    assert trades[1][0] == 2

    reasons = {r[0] for r in db.fetchall(
        "SELECT reason FROM vwr_signals WHERE run_id=?", (outcome.run_id,))}
    assert "entry_long" in reasons and "eod_flat" in reasons

    # 强平发生在 14:55 ±1 个轮询间隔
    eod_ts = db.fetchone(
        "SELECT ts FROM vwr_signals WHERE run_id=? AND reason='eod_flat'",
        (outcome.run_id,))[0]
    assert sh_epoch(14, 55) <= eod_ts <= sh_epoch(14, 57)

    # EOD 之后直到 15:00 还在采样，但再无成交
    last_trade_ts = db.fetchone(
        "SELECT MAX(ts) FROM vwr_trades WHERE run_id=?", (outcome.run_id,))[0]
    last_snap_ts = db.fetchone(
        "SELECT MAX(ts) FROM vwr_snapshots WHERE trade_date=?", (DAY,))[0]
    assert last_snap_ts > last_trade_ts
    assert any(e.payload.get("kind") == "eod" for e in renderer.events)


def test_quiet_day_no_trades_summary_still_written(db: Database) -> None:
    """全天横盘（σ≈0）→ 零信号零成交，汇总行仍落库。"""
    outcome, _ = run_daemon(db, sh_epoch(14, 0), lambda _e: 1.00)
    assert outcome.status == "done"
    assert trades_of(db, outcome.run_id) == []
    row = db.fetchone(
        "SELECT n_trades, net_pnl, final_cash FROM vwr_daily_summary "
        "WHERE trade_date=?", (DAY,),
    )
    assert row[0] == 0 and row[1] == pytest.approx(0.0)
    assert row[2] == pytest.approx(VwrConfig().initial_cash)
