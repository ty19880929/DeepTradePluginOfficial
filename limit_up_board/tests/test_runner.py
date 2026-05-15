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
