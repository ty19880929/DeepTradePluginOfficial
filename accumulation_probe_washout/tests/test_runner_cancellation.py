"""ApwRunner cancel-classification tests.

Cases:

* A — ``KeyboardInterrupt`` raised inside the screen pipeline →
  ``RunStatus.CANCELLED`` with friendly ``terminal_error``; friendly WARN
  log events emitted; the apw_runs row is finalised as 'cancelled' (not
  stranded at 'running').
* B — SIGINT marker already set (simulated) + a derived exception →
  same CANCELLED path. The marker, not the type, drives the decision.
* C — marker NOT set + a real exception → ``RunStatus.FAILED``;
  pre-change behaviour preserved.

Plus unit cover for ``_emit_cancelled_log``, ``_shielded_finish_run``,
``_make_cancel_outcome``.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from deeptrade.core.db import Database
from deeptrade.core.run_status import RunStatus
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from accumulation_probe_washout import cancellation
from accumulation_probe_washout.runner import (
    ApwRunner,
    RunOutcome,
    ScreenParams,
)
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import EventRenderer

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[1] / "migrations"
)


@pytest.fixture(autouse=True)
def _reset_marker() -> None:
    cancellation.reset_marker()
    yield
    cancellation.reset_marker()


@pytest.fixture
def runner_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "apw_cancel.duckdb")
    for sql_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql = sql_path.read_text(encoding="utf-8")
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                db.execute(stmt)
    yield db
    db.close()


class _CapturingRenderer:
    def __init__(self) -> None:
        self.events: list[StrategyEvent] = []
        self.finished: dict[str, Any] | None = None

    def on_run_started(self, **_kwargs: Any) -> None:
        return None

    def on_event(self, ev: StrategyEvent) -> None:
        self.events.append(ev)

    def on_run_finished(
        self,
        *,
        status: RunStatus,
        error: str | None,
        summary: dict[str, Any],
    ) -> None:
        self.finished = {"status": status, "error": error, "summary": summary}


def _make_runner(db: Database) -> ApwRunner:
    rt = ApwRuntime(
        db=db,
        config=MagicMock(),
        llms=MagicMock(),
    )
    rt.run_id = "11111111-1111-1111-1111-111111111111"
    # Seed the apw_runs row so _finish_run's UPDATE finds it.
    db.execute(
        "INSERT INTO apw_runs(run_id, mode, trade_date, is_intraday, status, "
        "started_at, params_json) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        (
            rt.run_id, "screen", "20260612", False,
            RunStatus.RUNNING.value, "{}",
        ),
    )
    runner = ApwRunner(rt, renderer=_CapturingRenderer())
    return runner


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_emit_cancelled_log_emits_two_warn_events(runner_db: Database) -> None:
    runner = _make_runner(runner_db)
    runner._emit_cancelled_log()
    events = runner.renderer.events  # type: ignore[attr-defined]
    assert len(events) == 2
    assert all(ev.type == EventType.LOG for ev in events)
    assert all(ev.level == EventLevel.WARN for ev in events)
    assert any("用户手动中断" in ev.message for ev in events)
    assert any("运行已取消" in ev.message for ev in events)


def test_shielded_finish_run_masks_sigint_then_restores(
    runner_db: Database,
) -> None:
    runner = _make_runner(runner_db)
    seen: dict[str, object] = {}

    def _fake_finish(_status, *, error=None, summary=None) -> None:
        seen["during"] = signal.getsignal(signal.SIGINT)

    runner._finish_run = _fake_finish  # type: ignore[assignment]
    prev_outside = signal.getsignal(signal.SIGINT)
    runner._shielded_finish_run(
        RunStatus.CANCELLED, error="用户手动中断"
    )
    assert seen["during"] is signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == prev_outside


def test_make_cancel_outcome_finalises_run_row(
    runner_db: Database,
) -> None:
    runner = _make_runner(runner_db)
    outcome = runner._make_cancel_outcome(
        runner.rt.run_id, "screen", _owns_run=True
    )
    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    row = runner_db.fetchone(
        "SELECT status, error FROM apw_runs WHERE run_id = ?",
        (runner.rt.run_id,),
    )
    assert row is not None
    assert row[0] == "cancelled"
    assert row[1] == "用户手动中断"


# ---------------------------------------------------------------------------
# Classification — drive execute_screen with KeyboardInterrupt / derived exc
# ---------------------------------------------------------------------------


def _make_boom_tushare(exc: BaseException) -> Any:
    """Return a tushare that serves a fake trade_cal then raises on later calls.

    Why two-phase: execute_screen calls fetch_trade_cal BEFORE the try
    block. Raising there would bypass our new except branch and propagate
    KeyboardInterrupt all the way up — not the user-visible scenario we
    want to model. The realistic path is: data sync starts, then Ctrl+C
    arrives mid-fetch. So the fake trade_cal succeeds, then the next call
    (fetch_stock_basic / fetch_st_codes) blows up inside the try block.
    """
    import pandas as pd

    # Build a minimal cal_df that TradeCalendar(...) and resolve_trade_date
    # can both consume: a 30-day window spanning the test trade_date.
    rows = []
    cur = pd.Timestamp("2026-05-01")
    end = pd.Timestamp("2026-09-30")
    while cur <= end:
        cal_date = cur.strftime("%Y%m%d")
        rows.append({"cal_date": cal_date, "is_open": 1, "pretrade_date": cal_date})
        cur += pd.Timedelta(days=1)
    cal_df = pd.DataFrame(rows)

    class _BoomTushare:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self._call_count = 0

        def call(self, api_name: str, *_args: Any, **_kwargs: Any) -> Any:
            self.calls.append(api_name)
            self._call_count += 1
            if api_name == "trade_cal":
                return cal_df.copy()
            raise exc

    return _BoomTushare()


def test_keyboardinterrupt_routes_to_cancelled(
    runner_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _make_runner(runner_db)
    runner.rt.tushare = _make_boom_tushare(KeyboardInterrupt())  # type: ignore[assignment]

    outcome = runner.execute_screen(ScreenParams(trade_date="20260612"))

    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    # apw_runs row must be finalised as 'cancelled' (not stranded at 'running').
    row = runner_db.fetchone(
        "SELECT status FROM apw_runs WHERE run_id = ?", (runner.rt.run_id,)
    )
    assert row is not None and row[0] == "cancelled"
    # Cancel log surfaced; no ERROR-level event in stream.
    events = runner.renderer.events  # type: ignore[attr-defined]
    assert any("用户手动中断" in ev.message for ev in events)
    assert all(ev.level != EventLevel.ERROR for ev in events)


def test_derived_exception_with_marker_set_routes_to_cancelled(
    runner_db: Database,
) -> None:
    cancellation._marker.set()  # type: ignore[attr-defined]
    runner = _make_runner(runner_db)
    runner.rt.tushare = _make_boom_tushare(  # type: ignore[assignment]
        RuntimeError("INTERRUPT Error: simulated DuckDB break")
    )

    outcome = runner.execute_screen(ScreenParams(trade_date="20260612"))

    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    row = runner_db.fetchone(
        "SELECT status FROM apw_runs WHERE run_id = ?", (runner.rt.run_id,)
    )
    assert row is not None and row[0] == "cancelled"
    events = runner.renderer.events  # type: ignore[attr-defined]
    assert all("Traceback" not in ev.message for ev in events)


def test_derived_exception_without_marker_routes_to_failed(
    runner_db: Database,
) -> None:
    assert cancellation.cancel_requested() is False
    runner = _make_runner(runner_db)
    runner.rt.tushare = _make_boom_tushare(RuntimeError("real bug"))  # type: ignore[assignment]

    outcome = runner.execute_screen(ScreenParams(trade_date="20260612"))

    assert outcome.status == RunStatus.FAILED
    assert "real bug" in (outcome.error or "")
    row = runner_db.fetchone(
        "SELECT status FROM apw_runs WHERE run_id = ?", (runner.rt.run_id,)
    )
    assert row is not None and row[0] == "failed"
    # Traceback is still surfaced via LOG ERROR for real failures.
    events = runner.renderer.events  # type: ignore[attr-defined]
    assert any(ev.level == EventLevel.ERROR for ev in events)
