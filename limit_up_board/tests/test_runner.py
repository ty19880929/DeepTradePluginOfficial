"""Unit tests for LubRunner internal helpers.

These tests target small, isolated runner methods that don't require a full
LLM / Tushare / pipeline orchestration — e.g. DB write helpers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deeptrade.core.db import Database

from limit_up_board.runner import LubRunner
from limit_up_board.runtime import LubRuntime

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql"
)


@pytest.fixture
def runner_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))

    from deeptrade.core import paths as core_paths

    db = Database(core_paths.db_path())
    sql_text = MIGRATION_FILE.read_text(encoding="utf-8")
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    return db


def _make_runner(db: Database) -> LubRunner:
    """Build a minimal LubRunner for helper-method tests.

    The runner constructor only requires a LubRuntime; we mock the remaining
    services because the tests under exercise only DB-touching helpers.
    """
    rt = LubRuntime(
        db=db,
        config=MagicMock(),
        llms=MagicMock(),
    )
    return LubRunner(rt)


def test_lub_runs_trade_date_backfilled_after_resolution(runner_db: Database) -> None:
    """P3-2: _backfill_run_trade_date must overwrite the empty trade_date column.

    ``_record_run_start`` writes ``params.trade_date or ""`` — when the CLI
    omitted ``--trade-date``, the row lands with ``trade_date=""``. Step 0
    resolves the real T and ``_backfill_run_trade_date`` must update the row
    so history / report joins work.
    """
    run_id = "11111111-1111-1111-1111-111111111111"
    # Simulate _record_run_start with empty trade_date (no --trade-date flag)
    runner_db.execute(
        "INSERT INTO lub_runs(run_id, trade_date, status, is_intraday, started_at, "
        "params_json) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        (run_id, "", "running", False, "{}"),
    )

    runner = _make_runner(runner_db)
    runner._backfill_run_trade_date(run_id, "20260530")

    row = runner_db.fetchone(
        "SELECT trade_date FROM lub_runs WHERE run_id=?", (run_id,)
    )
    assert row is not None
    assert row[0] == "20260530"


def test_backfill_trade_date_idempotent(runner_db: Database) -> None:
    """Second backfill with the same date is a no-op (just re-writes the same value)."""
    run_id = "22222222-2222-2222-2222-222222222222"
    runner_db.execute(
        "INSERT INTO lub_runs(run_id, trade_date, status, is_intraday, started_at, "
        "params_json) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        (run_id, "", "running", False, "{}"),
    )
    runner = _make_runner(runner_db)
    runner._backfill_run_trade_date(run_id, "20260530")
    runner._backfill_run_trade_date(run_id, "20260530")
    row = runner_db.fetchone(
        "SELECT trade_date FROM lub_runs WHERE run_id=?", (run_id,)
    )
    assert row is not None
    assert row[0] == "20260530"


# ---------------------------------------------------------------------------
# P1-A: _iter_pipeline / _iter_sync invoke _backfill_run_trade_date in Step 0
# ---------------------------------------------------------------------------

def _stub_step0(monkeypatch: pytest.MonkeyPatch, T: str = "20260530", T1: str = "20260601") -> None:
    """Monkeypatch all Step 0 collaborators so the generator can be advanced
    just past Step 0 without needing real Tushare / calendar fixtures."""
    from limit_up_board import runner as runner_mod

    monkeypatch.setattr(runner_mod, "TradeCalendar", lambda df: MagicMock())
    monkeypatch.setattr(runner_mod, "fetch_latest_trade_date", lambda ts: T)
    monkeypatch.setattr(
        runner_mod, "resolve_trade_date",
        lambda cal, latest_trade_date=None, user_specified=None: (T, T1),
    )


def _make_runner_with_run_id(
    db: Database, run_id: str
) -> LubRunner:
    """Build runner with run_id already persisted (mimics _record_run_start)."""
    rt = LubRuntime(db=db, config=MagicMock(), llms=MagicMock())
    rt.plugin_id = "limit-up-board"
    rt.run_id = run_id
    # tushare.call("trade_cal") must be callable; return value ignored downstream.
    rt.tushare = MagicMock()
    rt.tushare.call.return_value = MagicMock()
    runner = LubRunner(rt)
    db.execute(
        "INSERT INTO lub_runs(run_id, trade_date, status, is_intraday, started_at, "
        "params_json) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)",
        (run_id, "", "running", False, "{}"),
    )
    return runner


def test_iter_sync_backfills_trade_date_after_step0(
    runner_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-A: _iter_sync must call _backfill_run_trade_date once T is resolved.

    Prior to this fix, only the debate path (_do_step_0_and_1) backfilled
    trade_date; the sync path left ``lub_runs.trade_date=''`` whenever
    ``--trade-date`` was omitted, breaking history/report joins.
    """
    from limit_up_board.runner import RunParams
    from limit_up_board import runner as runner_mod

    run_id = "33333333-3333-3333-3333-333333333333"
    _stub_step0(monkeypatch, T="20260530", T1="20260601")
    runner = _make_runner_with_run_id(runner_db, run_id)

    # Short-circuit downstream so the generator stops right after Step 0.
    monkeypatch.setattr(runner_mod, "load_config", lambda db: MagicMock(
        min_float_mv_yi=0, max_float_mv_yi=0, max_close_yuan=0,
    ))
    monkeypatch.setattr(runner_mod, "_settings_log_event", lambda rt, cfg: MagicMock())

    def _stop(*a, **kw):
        raise StopIteration  # propagate as generator close
    monkeypatch.setattr(runner_mod, "collect_round1", _stop)

    params = RunParams(trade_date=None, force_sync=False)
    gen = runner._iter_sync(params)
    # Drain until generator hits collect_round1 (StopIteration short-circuit).
    try:
        for _ in gen:
            pass
    except (StopIteration, RuntimeError):
        # RuntimeError: PEP 479 wraps generator-internal StopIteration.
        pass

    row = runner_db.fetchone(
        "SELECT trade_date FROM lub_runs WHERE run_id=?", (run_id,)
    )
    assert row is not None
    assert row[0] == "20260530", f"expected backfilled 20260530, got {row[0]!r}"


def test_iter_pipeline_backfills_trade_date_after_step0(
    runner_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1-A: _iter_pipeline (single-LLM path) must backfill trade_date,
    mirroring the debate path's _do_step_0_and_1 behaviour."""
    from limit_up_board.runner import RunParams
    from limit_up_board import runner as runner_mod

    run_id = "44444444-4444-4444-4444-444444444444"
    _stub_step0(monkeypatch, T="20260530", T1="20260601")
    runner = _make_runner_with_run_id(runner_db, run_id)
    runner._rt.config.get_app_config = MagicMock(return_value=MagicMock(app_profile="balanced"))

    monkeypatch.setattr(runner_mod, "load_config", lambda db: MagicMock(
        min_float_mv_yi=0, max_float_mv_yi=0, max_close_yuan=0,
        lgb_min_score_floor=None, lgb_decile_in_prompt=False,
    ))
    monkeypatch.setattr(runner_mod, "_settings_log_event", lambda rt, cfg: MagicMock())

    def _stop(*a, **kw):
        raise StopIteration
    monkeypatch.setattr(runner_mod, "collect_round1", _stop)

    params = RunParams(trade_date=None, force_sync=False)
    gen = runner._iter_pipeline(params)
    try:
        for _ in gen:
            pass
    except (StopIteration, RuntimeError):
        pass

    row = runner_db.fetchone(
        "SELECT trade_date FROM lub_runs WHERE run_id=?", (run_id,)
    )
    assert row is not None
    assert row[0] == "20260530", f"expected backfilled 20260530, got {row[0]!r}"
