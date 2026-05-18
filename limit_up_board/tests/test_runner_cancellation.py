"""LubRunner cancel-classification tests.

Verifies the three paths added in v0.6.9:

* case A — ``KeyboardInterrupt`` raised inside the pipeline iterator →
  ``RunStatus.CANCELLED`` with friendly ``terminal_error``; two friendly
  WARN log events emitted; ``_handle_runtime_exception`` NOT called; no
  ERROR-level events in the captured stream.
* case B — SIGINT marker already set (simulated) + a derived exception
  raised inside the iterator → same CANCELLED path. The exception type /
  message is intentionally not ``KeyboardInterrupt`` (we use
  ``RuntimeError`` here, but the marker, not the type, drives the
  decision).
* case C — marker NOT set + a real exception → ``RunStatus.FAILED``;
  ``_handle_runtime_exception`` IS called; pre-change behaviour preserved.

Also covers:

* ``_emit_cancelled_log`` — emits two WARN-level LOG events, neither at
  ERROR level, both with cancel-friendly text and no traceback substring.
* ``_shielded_record_run_finish`` — SIGINT is set to ``SIG_IGN`` during the
  inner ``_record_run_finish`` call, then restored. The shield closes
  even when ``_record_run_finish`` raises.
"""

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deeptrade.core.db import Database
from deeptrade.core.run_status import RunStatus
from deeptrade.plugins_api.events import EventLevel, EventType, StrategyEvent

from limit_up_board import cancellation
from limit_up_board.runner import LubRunner, RunParams
from limit_up_board.runtime import LubRuntime

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql"
)


@pytest.fixture(autouse=True)
def _reset_marker() -> None:
    cancellation.reset_marker()
    yield
    cancellation.reset_marker()


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


class _CapturingRenderer:
    """Renderer test double — captures dispatched events for assertions."""

    def __init__(self) -> None:
        self.events: list[StrategyEvent] = []

    def on_event(self, ev: StrategyEvent) -> None:
        self.events.append(ev)


def _make_runner(db: Database) -> LubRunner:
    rt = LubRuntime(
        db=db,
        config=MagicMock(),
        llms=MagicMock(),
    )
    rt.plugin_id = "limit-up-board"
    runner = LubRunner(rt)
    runner._renderer = _CapturingRenderer()  # type: ignore[assignment]
    return runner


# ---------------------------------------------------------------------------
# _emit_cancelled_log
# ---------------------------------------------------------------------------


def test_emit_cancelled_log_emits_two_warn_events(runner_db: Database) -> None:
    runner = _make_runner(runner_db)
    runner._emit_cancelled_log()
    events = runner._renderer.events  # type: ignore[attr-defined]
    assert len(events) == 2
    assert all(ev.type == EventType.LOG for ev in events)
    assert all(ev.level == EventLevel.WARN for ev in events)
    assert any("用户手动中断" in ev.message for ev in events)
    assert any("运行已取消" in ev.message for ev in events)
    # No traceback or Python exception text leaks into the user-facing log.
    blob = "\n".join(ev.message for ev in events)
    assert "Traceback" not in blob
    assert "RuntimeError" not in blob


# ---------------------------------------------------------------------------
# _shielded_record_run_finish
# ---------------------------------------------------------------------------


def test_shielded_record_finish_masks_sigint_then_restores(
    runner_db: Database,
) -> None:
    runner = _make_runner(runner_db)
    seen_handler: dict[str, object] = {}

    def _fake_finish(*_args, **_kwargs) -> None:
        seen_handler["during"] = signal.getsignal(signal.SIGINT)

    runner._record_run_finish = _fake_finish  # type: ignore[assignment]
    prev_outside = signal.getsignal(signal.SIGINT)
    runner._shielded_record_run_finish(
        "rid", RunStatus.CANCELLED, "用户手动中断", []
    )
    after = signal.getsignal(signal.SIGINT)
    assert seen_handler["during"] is signal.SIG_IGN
    # Handler is restored to what it was before the shield (typically the
    # previous handler; pytest may have installed its own).
    assert after == prev_outside


def test_shielded_record_finish_restores_handler_on_exception(
    runner_db: Database,
) -> None:
    runner = _make_runner(runner_db)

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("db write failed")

    runner._record_run_finish = _boom  # type: ignore[assignment]
    prev_outside = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match="db write failed"):
        runner._shielded_record_run_finish(
            "rid", RunStatus.FAILED, "boom", []
        )
    assert signal.getsignal(signal.SIGINT) == prev_outside


# ---------------------------------------------------------------------------
# classification — drive _execute_single through monkeypatched dependencies
# ---------------------------------------------------------------------------


def _patch_runner_for_classification(runner: LubRunner) -> dict[str, MagicMock]:
    """Mock everything _execute_single touches except the iterator + classification."""
    runner._validate_single_provider = MagicMock(return_value=None)  # type: ignore[assignment]
    runner._rt.llms.get_client = MagicMock(return_value=MagicMock())  # type: ignore[assignment]
    runner._record_run_start = MagicMock()  # type: ignore[assignment]
    runner._record_run_finish = MagicMock()  # type: ignore[assignment]
    runner._persist_event = MagicMock()  # type: ignore[assignment]
    handle = MagicMock(return_value="handled-error-summary")
    runner._handle_runtime_exception = handle  # type: ignore[assignment]
    return {"handle": handle}


def test_keyboardinterrupt_routes_to_cancelled(runner_db: Database) -> None:
    runner = _make_runner(runner_db)
    mocks = _patch_runner_for_classification(runner)

    def _iter(_params: RunParams):
        raise KeyboardInterrupt
        yield  # pragma: no cover — make it a generator

    runner._iter_pipeline = _iter  # type: ignore[assignment]

    outcome = runner._execute_single("rid-A", RunParams())

    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    mocks["handle"].assert_not_called()
    # Friendly cancel log surfaced; no ERROR events in stream.
    events = runner._renderer.events  # type: ignore[attr-defined]
    assert any("用户手动中断" in ev.message for ev in events)
    assert all(ev.level != EventLevel.ERROR for ev in events)


def test_derived_exception_with_marker_set_routes_to_cancelled(
    runner_db: Database,
) -> None:
    cancellation._marker.set()  # type: ignore[attr-defined]
    runner = _make_runner(runner_db)
    mocks = _patch_runner_for_classification(runner)

    def _iter(_params: RunParams):
        # Deliberately NOT a KeyboardInterrupt — a derived exception that
        # would historically have been classified as FAILED. The marker is
        # the only thing that should reroute it.
        raise RuntimeError("INTERRUPT Error: simulated DuckDB break")
        yield  # pragma: no cover

    runner._iter_pipeline = _iter  # type: ignore[assignment]

    outcome = runner._execute_single("rid-B", RunParams())

    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    mocks["handle"].assert_not_called()
    events = runner._renderer.events  # type: ignore[attr-defined]
    assert all(ev.level != EventLevel.ERROR for ev in events)
    # Traceback line-by-line surfacing must NOT happen on the cancel path.
    assert all("Traceback" not in ev.message for ev in events)


def test_derived_exception_without_marker_routes_to_failed(
    runner_db: Database,
) -> None:
    # marker stays unset; default
    assert cancellation.cancel_requested() is False
    runner = _make_runner(runner_db)
    mocks = _patch_runner_for_classification(runner)

    def _iter(_params: RunParams):
        raise RuntimeError("real bug")
        yield  # pragma: no cover

    runner._iter_pipeline = _iter  # type: ignore[assignment]

    outcome = runner._execute_single("rid-C", RunParams())

    assert outcome.status == RunStatus.FAILED
    assert outcome.error == "handled-error-summary"
    mocks["handle"].assert_called_once()
