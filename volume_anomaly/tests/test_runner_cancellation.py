"""VaRunner cancel-classification tests.

Mirrors limit_up_board/tests/test_runner_cancellation.py with VA's
``_drive(mode, params, iterator)`` shape.

Cases:

* A — ``KeyboardInterrupt`` raised inside the iterator →
  ``RunStatus.CANCELLED`` with friendly ``terminal_error``; two friendly
  WARN log events emitted; ``_handle_runtime_exception`` NOT called; no
  ERROR-level events in the captured stream.
* B — SIGINT marker already set (simulated) + a derived exception →
  same CANCELLED path. The exception type / message is intentionally not
  ``KeyboardInterrupt`` — the marker, not the type, drives the decision.
* C — marker NOT set + a real exception → ``RunStatus.FAILED``;
  ``_handle_runtime_exception`` IS called.

Plus unit-level cover for ``_emit_cancelled_log`` and
``_shielded_record_run_finish``.
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

from volume_anomaly import cancellation
from volume_anomaly.runner import VaRunner
from volume_anomaly.runtime import VaRuntime

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


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
    for sql_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        sql_text = sql_path.read_text(encoding="utf-8")
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                db.execute(stmt)
    return db


class _CapturingRenderer:
    def __init__(self) -> None:
        self.events: list[StrategyEvent] = []

    def on_run_start(self, **_kwargs: Any) -> None:
        return None

    def on_event(self, ev: StrategyEvent) -> None:
        self.events.append(ev)

    def on_run_finish(self, _outcome: Any) -> None:
        return None

    def close(self) -> None:
        return None


class _ScreenParams:
    def __init__(self) -> None:
        self.trade_date = "20260612"
        self.force_sync = False


def _make_runner(db: Database) -> VaRunner:
    rt = VaRuntime(
        db=db,
        config=MagicMock(),
        llms=MagicMock(),
    )
    runner = VaRunner(rt, renderer=_CapturingRenderer())
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
    seen: dict[str, object] = {}

    def _fake_finish(*_args, **_kwargs) -> None:
        seen["during"] = signal.getsignal(signal.SIGINT)

    runner._record_run_finish = _fake_finish  # type: ignore[assignment]
    prev_outside = signal.getsignal(signal.SIGINT)
    runner._shielded_record_run_finish(
        "rid", RunStatus.CANCELLED, "用户手动中断", []
    )
    assert seen["during"] is signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) == prev_outside


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
# Drive classification through _drive(...)
# ---------------------------------------------------------------------------


def _patch_runner_for_classification(runner: VaRunner) -> dict[str, MagicMock]:
    runner._persist_event = MagicMock()  # type: ignore[assignment]
    handle = MagicMock(return_value="handled-error-summary")
    runner._handle_runtime_exception = handle  # type: ignore[assignment]
    return {"handle": handle}


def test_keyboardinterrupt_routes_to_cancelled(runner_db: Database) -> None:
    runner = _make_runner(runner_db)
    mocks = _patch_runner_for_classification(runner)

    def _iter():
        raise KeyboardInterrupt
        yield  # pragma: no cover

    outcome = runner._drive("screen", _ScreenParams(), _iter())

    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    mocks["handle"].assert_not_called()
    events = runner._renderer.events  # type: ignore[attr-defined]
    assert any("用户手动中断" in ev.message for ev in events)
    assert all(ev.level != EventLevel.ERROR for ev in events)


def test_derived_exception_with_marker_set_routes_to_cancelled(
    runner_db: Database,
) -> None:
    cancellation._marker.set()  # type: ignore[attr-defined]
    runner = _make_runner(runner_db)
    mocks = _patch_runner_for_classification(runner)

    def _iter():
        raise RuntimeError("INTERRUPT Error: simulated DuckDB break")
        yield  # pragma: no cover

    outcome = runner._drive("screen", _ScreenParams(), _iter())

    assert outcome.status == RunStatus.CANCELLED
    assert outcome.error == "用户手动中断"
    mocks["handle"].assert_not_called()
    events = runner._renderer.events  # type: ignore[attr-defined]
    assert all(ev.level != EventLevel.ERROR for ev in events)
    assert all("Traceback" not in ev.message for ev in events)


def test_derived_exception_without_marker_routes_to_failed(
    runner_db: Database,
) -> None:
    assert cancellation.cancel_requested() is False
    runner = _make_runner(runner_db)
    mocks = _patch_runner_for_classification(runner)

    def _iter():
        raise RuntimeError("real bug")
        yield  # pragma: no cover

    outcome = runner._drive("screen", _ScreenParams(), _iter())

    assert outcome.status == RunStatus.FAILED
    assert outcome.error == "handled-error-summary"
    mocks["handle"].assert_called_once()
