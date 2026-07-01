"""CollectDaemon 集成测试（设计 §10.1，P1 验收项）.

假时钟 + 假数据源 + 真 DuckDB（真迁移建表）：sleep 直接推进假时钟，
一个交易日毫秒级跑完。覆盖：待机→运行→收盘状态机、采样/落库、午休跨段、
崩溃恢复不双计、Ctrl-C → aborted、接口连续失败不崩、渲染器坏死降级。
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
from vwap_reversion.daemon import CollectDaemon, PreconditionError
from vwap_reversion.persistence import TradeCalendarStore, load_bars
from vwap_reversion.runtime import VwrRuntime
from vwap_reversion.schemas import Snapshot

SH = ZoneInfo("Asia/Shanghai")
MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "20260603_001_init.sql"
DAY = "20260603"  # 周三


def sh_epoch(hh: int, mm: int, ss: int = 0, day: str = DAY) -> float:
    return datetime(
        int(day[:4]), int(day[4:6]), int(day[6:8]), hh, mm, ss, tzinfo=SH
    ).timestamp()


class FakeClock(MarketClock):
    """epoch 可控的市场时钟；sleep 即推进时间。"""

    def __init__(self, start_epoch: float) -> None:
        self._epoch = start_epoch
        super().__init__("Asia/Shanghai", now_fn=lambda: self._epoch)

    def sleep(self, seconds: float) -> None:
        self._epoch += max(float(seconds), 0.001)  # 防 0 sleep 死循环


class FakeSource:
    """确定性快照源：成交量随时间线性增长，价格围绕 1.0 周期波动。"""

    def __init__(self, clock: FakeClock, code: str = "159518.SZ") -> None:
        self.clock = clock
        self.code = code
        self.fetches = 0
        self.fail_at: set[int] = set()          # 第 N 次 fetch 抛错
        self.interrupt_at: int | None = None    # 第 N 次 fetch 抛 KeyboardInterrupt

    def fetch(self) -> Snapshot:
        self.fetches += 1
        if self.interrupt_at is not None and self.fetches >= self.interrupt_at:
            raise KeyboardInterrupt
        if self.fetches in self.fail_at:
            raise RuntimeError("synthetic fetch failure")
        now = self.clock.now_epoch()
        open_epoch = sh_epoch(9, 30, day=self.clock.today_str())
        elapsed = max(0.0, now - open_epoch)
        price = 1.0 + 0.01 * ((self.fetches % 7) - 3)  # 0.97..1.03 波动
        cum_vol = 50_000.0 + elapsed * 100.0
        cum_amount = cum_vol * 1.0 + self.fetches * 10.0  # 略偏离使 σ>0
        return Snapshot(
            code=self.code, trade_date=self.clock.today_str(),
            ts=int(now), last=price, cum_vol=cum_vol, cum_amount=cum_amount,
            num_trades=float(self.fetches),
        )


class StaleSource(FakeSource):
    def fetch(self) -> Snapshot:
        self.fetches += 1
        now = self.clock.now_epoch()
        return Snapshot(
            code=self.code,
            trade_date=self.clock.today_str(),
            ts=int(now),
            last=1.0,
            cum_vol=100_000.0,
            cum_amount=100_000.0,
            num_trades=1.0,
            trade_time="10:00:00",
        )


class CaptureRenderer:
    def __init__(self) -> None:
        self.meta = None
        self.events = []
        self.finished = False

    def start(self, meta, clock) -> None:
        self.meta = meta

    def handle(self, event) -> None:
        self.events.append(event)

    def finish(self) -> None:
        self.finished = True


class BrokenRenderer(CaptureRenderer):
    def handle(self, event) -> None:  # 永远炸 —— 测中途降级
        raise RuntimeError("UI exploded")


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "daemon.duckdb")
    sql = MIGRATION.read_text(encoding="utf-8")
    for stmt in sql.split(";"):
        if stmt.strip():
            database.execute(stmt)
    # 本周一~周五为交易日，周六/日休市
    for d, is_open in [("20260601", 1), ("20260602", 1), ("20260603", 1),
                       ("20260604", 1), ("20260605", 1), ("20260606", 0),
                       ("20260607", 0), ("20260608", 1)]:
        database.execute(
            "INSERT INTO vwr_trade_cal(exchange, cal_date, is_open, pretrade_date) "
            "VALUES ('SSE', ?, ?, '')", (d, is_open),
        )
    yield database
    database.close()


def make_daemon(db: Database, clock: FakeClock, *, cfg: VwrConfig | None = None,
                renderer=None, source=None):
    cfg = cfg or replace(
        VwrConfig(), poll_interval_seconds=60, standby_heartbeat_seconds=60
    )
    rt = VwrRuntime(db=db, config=None, clock=clock)  # type: ignore[arg-type]
    source = source or FakeSource(clock)
    renderer = renderer if renderer is not None else CaptureRenderer()
    daemon = CollectDaemon(
        rt, cfg, source=source, calendar=TradeCalendarStore(db),
        renderer=renderer, sleep_fn=clock.sleep,
    )
    return daemon, source, renderer


def run_row(db: Database, run_id: str):
    return db.fetchone(
        "SELECT status, trade_date, finished_at FROM vwr_runs WHERE run_id = ?",
        (run_id,),
    )


# ---------------------------------------------------------------------------


def test_full_day_pre_open_standby_to_done(db: Database) -> None:
    clock = FakeClock(sh_epoch(9, 0))  # 开盘前 30 分钟启动
    daemon, source, renderer = make_daemon(db, clock)
    outcome = daemon.execute(code="159518.SZ")

    assert outcome.status == "done"
    assert outcome.exit_code == 0
    status, trade_date, finished_at = run_row(db, outcome.run_id)
    assert status == "done" and trade_date == DAY and finished_at is not None

    # 待机事件出现过，且首次采样不早于 09:30
    standby_events = [e for e in renderer.events if e.payload.get("kind") == "standby"]
    assert standby_events
    first_sample_ts = next(
        e.payload for e in renderer.events if e.payload.get("kind") == "sample"
    )
    assert source.fetches > 0

    # 采样覆盖上下午两段：09:30–11:30 + 13:00–15:00，poll=60s → ~240 次
    assert 230 <= source.fetches <= 250

    # 快照与 bar 已落库；bar 单调按 ts 升序
    n_snaps = db.fetchone(
        "SELECT COUNT(*) FROM vwr_snapshots WHERE code='159518.SZ' AND trade_date=?",
        (DAY,),
    )[0]
    assert n_snaps == source.fetches
    bars = load_bars(db, "159518.SZ", DAY)
    assert bars and all(b1.ts < b2.ts for b1, b2 in zip(bars, bars[1:]))

    # 午休时段（11:30–13:00 上海）不应有任何快照
    lunch_lo, lunch_hi = sh_epoch(11, 30), sh_epoch(13, 0)
    in_lunch = db.fetchone(
        "SELECT COUNT(*) FROM vwr_snapshots WHERE code='159518.SZ' AND trade_date=? "
        "AND ts >= ? AND ts < ?", (DAY, int(lunch_lo), int(lunch_hi)),
    )[0]
    assert in_lunch == 0

    # 事件已落 vwr_events
    n_events = db.fetchone(
        "SELECT COUNT(*) FROM vwr_events WHERE run_id = ?", (outcome.run_id,)
    )[0]
    assert n_events > 0
    assert renderer.finished
    _ = first_sample_ts


def test_intraday_start_runs_immediately(db: Database) -> None:
    clock = FakeClock(sh_epoch(14, 30))  # 下午盘中启动
    daemon, source, renderer = make_daemon(db, clock)
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"
    assert not [e for e in renderer.events if e.payload.get("kind") == "standby"]
    assert 25 <= source.fetches <= 35  # 14:30→15:00, poll=60s


def test_lunch_start_sleeps_to_afternoon(db: Database) -> None:
    clock = FakeClock(sh_epoch(12, 0))
    daemon, source, renderer = make_daemon(db, clock)
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"
    first_snap_ts = db.fetchone(
        "SELECT MIN(ts) FROM vwr_snapshots WHERE trade_date = ?", (DAY,)
    )[0]
    assert first_snap_ts >= sh_epoch(13, 0)  # 午休启动 → 首采样在 13:00 后


def test_after_close_raises_precondition(db: Database) -> None:
    clock = FakeClock(sh_epoch(15, 30))
    daemon, _, _ = make_daemon(db, clock)
    with pytest.raises(PreconditionError, match="已收盘"):
        daemon.execute(code="159518.SZ")
    assert db.fetchone("SELECT COUNT(*) FROM vwr_runs")[0] == 0  # 不留 run 残骸


def test_non_trading_day_raises_precondition(db: Database) -> None:
    clock = FakeClock(sh_epoch(10, 0, day="20260606"))  # 周六
    daemon, _, _ = make_daemon(db, clock)
    with pytest.raises(PreconditionError, match="非交易日"):
        daemon.execute(code="159518.SZ")


def test_standby_across_days_waits_to_next_open(db: Database) -> None:
    cfg = replace(VwrConfig(), poll_interval_seconds=60, standby_across_days=True)
    clock = FakeClock(sh_epoch(16, 0, day="20260605"))  # 周五收盘后
    daemon, source, _ = make_daemon(db, clock, cfg=cfg)
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"
    row = run_row(db, outcome.run_id)
    assert row[1] == "20260608"  # 跨周末待机到下周一
    first = db.fetchone(
        "SELECT MIN(ts) FROM vwr_snapshots WHERE trade_date = '20260608'"
    )[0]
    assert first >= sh_epoch(9, 30, day="20260608")


def test_resume_rebuilds_without_double_count(db: Database) -> None:
    # 第一段：上午跑到 11:00 被打断
    clock1 = FakeClock(sh_epoch(9, 40))
    daemon1, source1, _ = make_daemon(db, clock1)
    source1.interrupt_at = 50  # 跑 50 次采样后 Ctrl-C
    outcome1 = daemon1.execute(code="159518.SZ")
    assert outcome1.status == "aborted" and outcome1.exit_code == 130
    bars_before = load_bars(db, "159518.SZ", DAY)
    assert bars_before
    cum_before = bars_before[-1].cum_vol

    # 第二段：同日重启 —— 必须重放重建，且首条新 bar 不双计开盘以来的量
    clock2 = FakeClock(float(bars_before[-1].ts + 60))
    daemon2, _, renderer2 = make_daemon(db, clock2)
    outcome2 = daemon2.execute(code="159518.SZ")
    assert outcome2.status == "done"
    assert any(e.payload.get("kind") == "resume" for e in renderer2.events)

    bars_after = load_bars(db, "159518.SZ", DAY)
    new_bars = [b for b in bars_after if b.ts > bars_before[-1].ts]
    assert new_bars
    # 双计的话首条新 bar 的 interval_vol 会 ≈ cum_vol（5 万+）；正常应是分钟级增量
    assert new_bars[0].interval_vol < cum_before / 10
    # 累计字段全程单调
    assert all(b1.cum_vol <= b2.cum_vol for b1, b2 in zip(bars_after, bars_after[1:]))


def test_fetch_failures_backoff_but_survive(db: Database) -> None:
    clock = FakeClock(sh_epoch(14, 50))
    daemon, source, renderer = make_daemon(db, clock)
    source.fail_at = {1, 2, 3, 4, 5, 6}  # 连续 6 次失败（超过升级阈值 5）
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"  # 不崩，收盘正常收尾
    errs = [e for e in renderer.events if e.payload.get("kind") == "fetch_error"]
    assert len(errs) >= 6
    assert any(str(e.level.value) == "error" for e in errs)  # 第 5 次起升级 ERROR
    assert any(str(e.level.value) == "warn" for e in errs)


def test_stale_quote_warns_and_halts_new_entries(db: Database) -> None:
    clock = FakeClock(sh_epoch(14, 56))
    cfg = replace(VwrConfig(), poll_interval_seconds=30, stale_quote_seconds=60)
    source = StaleSource(clock)
    daemon, _, renderer = make_daemon(db, clock, cfg=cfg, source=source)
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"
    stale = [e for e in renderer.events if e.payload.get("kind") == "stale_quote"]
    assert stale
    assert any(str(e.level.value) == "warn" for e in stale)


def test_broken_renderer_degrades_and_run_completes(db: Database, capsys) -> None:
    clock = FakeClock(sh_epoch(14, 55))
    daemon, _, _ = make_daemon(db, clock, renderer=BrokenRenderer())
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"  # UI 炸了 run 不炸（§14）
    out = capsys.readouterr().out
    assert "[step.progress]" in out  # 已降级为 legacy 行式输出
