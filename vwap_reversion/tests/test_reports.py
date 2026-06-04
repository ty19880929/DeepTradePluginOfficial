"""收盘双报告（P3 验收项）：daemon 自动生成 + 构建器内容。"""

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
from vwap_reversion.persistence import TradeCalendarStore, get_run
from vwap_reversion.reporting import (
    build_execution_report,
    build_trades_report,
    generate_run_reports,
)
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


class CaptureRenderer:
    def __init__(self) -> None:
        self.events = []

    def start(self, meta, clock) -> None: ...

    def handle(self, event) -> None:
        self.events.append(event)

    def finish(self) -> None: ...


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "rep.duckdb")
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


def dip_recover(epoch: float) -> float:
    if sh_epoch(10, 30) <= epoch < sh_epoch(11, 0):
        return 0.99
    return 1.00


@pytest.fixture()
def finished_run(db: Database):
    from vwap_reversion.runtime import VwrRuntime

    cfg = replace(VwrConfig(), poll_interval_seconds=60, fee_bps=5.0, slippage_bps=10.0)
    clock = FakeClock(sh_epoch(9, 31))
    rt = VwrRuntime(db=db, config=None, clock=clock)  # type: ignore[arg-type]
    renderer = CaptureRenderer()
    daemon = CollectDaemon(
        rt, cfg, source=ScriptedSource(clock, dip_recover),
        calendar=TradeCalendarStore(db), renderer=renderer, sleep_fn=clock.sleep,
    )
    outcome = daemon.execute(code="159518.SZ")
    assert outcome.status == "done"
    return db, outcome, renderer


# ---------------------------------------------------------------------------


def test_daemon_generates_reports_at_close(finished_run, tmp_path: Path) -> None:
    db, outcome, renderer = finished_run
    run = get_run(db, outcome.run_id)
    assert run["report_dir"], "收盘后 report_dir 必须回写"
    report_dir = Path(run["report_dir"])
    # conftest 已把 plugin_data_dir 隔离到 tmp —— 报告绝不落真实 ~/.deeptrade
    assert str(report_dir).startswith(str(tmp_path))
    exec_md = report_dir / "execution_report.md"
    trades_md = report_dir / "trades_report.md"
    assert exec_md.is_file() and trades_md.is_file()
    assert "159518.SZ" in exec_md.read_text(encoding="utf-8")
    assert any(e.payload.get("kind") == "reports" for e in renderer.events)


def test_execution_report_content(finished_run) -> None:
    db, outcome, _ = finished_run
    md = build_execution_report(db, outcome.run_id)
    assert f"执行报告 — 159518.SZ {DAY}" in md
    assert "## 参数快照" in md and "`band_k_entry` = 2.0" in md
    assert "## 采样质量" in md and "有效 bar" in md
    assert "VWAP / σ 收敛快照" in md
    assert "## 信号统计" in md
    assert "`entry_long`×1" in md and "`revert_exit`×1" in md
    assert "## 异常与降级" in md
    assert "Asia/Shanghai" in md


def test_trades_report_content(finished_run) -> None:
    db, outcome, _ = finished_run
    md = build_trades_report(db, outcome.run_id)
    assert f"交易汇总报告 — 159518.SZ {DAY}" in md
    assert "## 当日汇总" in md
    assert "成交笔数 | 2" in md
    assert "100%" in md          # 胜率 1/1
    assert "未触发" in md         # 熔断
    assert "## 成交明细" in md
    assert "BUY" in md and "SELL" in md
    # 成交时间按上海墙钟显示（入场在 10:30–11:00 跌段内）
    assert "| 10:3" in md


def test_generate_is_idempotent_and_quiet_day_path(db: Database) -> None:
    # 无成交、被中断（无 summary 行）的 run：报告仍能生成，不抛
    from vwap_reversion.persistence import create_run

    run_id = create_run(
        db, mode="paper", code="159518.SZ", trade_date=DAY, status="aborted",
        params={"market_timezone": "Asia/Shanghai"}, initial_cash=100000.0,
    )
    out_dir = generate_run_reports(db, run_id)
    assert (out_dir / "trades_report.md").is_file()
    md = (out_dir / "trades_report.md").read_text(encoding="utf-8")
    assert "无汇总行" in md and "当日无成交" in md
    # 幂等：重复生成覆盖
    out_dir2 = generate_run_reports(db, run_id)
    assert out_dir2 == out_dir


def test_unknown_run_raises(db: Database) -> None:
    with pytest.raises(ValueError, match="run_id 不存在"):
        build_execution_report(db, "nope")
