"""StageStack — T4.3."""

from __future__ import annotations

import pytest

from accumulation_probe_washout.ui.stage_model import StageStack


class _Ev:
    def __init__(self, type_value: str, message: str = "", payload: dict | None = None):
        class _T: pass
        self.type = _T()
        self.type.value = type_value
        self.message = message
        self.payload = payload or {}


def test_for_mode_creates_correct_step_count():
    assert len(StageStack.for_mode("screen").steps) == 3
    assert len(StageStack.for_mode("analyze").steps) == 4
    # Run-mode StageStack adds a dedicated Step 2.5 ("读取 watchlist") between
    # screen's Step 2 and the LLM Step 3, so analyze's read-watchlist sub-stage
    # no longer collides with screen's Step 1 ("漏斗筛选") in the dashboard.
    assert len(StageStack.for_mode("run").steps) == 6


def test_data_sync_advances_step_0():
    s = StageStack.for_mode("screen")
    s.apply(_Ev("data.sync.started", "syncing"))
    assert s.get(0).state == "running"
    s.apply(_Ev("data.sync.finished", "done"))
    assert s.get(0).state == "done"


def test_step_started_and_finished():
    s = StageStack.for_mode("screen")
    s.apply(_Ev("step.started", "Step 1", payload={"step": 1}))
    assert s.get(1).state == "running"
    s.apply(_Ev("step.finished", "Step 1 done", payload={"step": 1}))
    assert s.get(1).state == "done"


def test_validation_failed_marks_running_step():
    s = StageStack.for_mode("analyze")
    s.apply(_Ev("step.started", "Step 3", payload={"step": 3}))
    s.apply(_Ev("validation.failed", "boom"))
    assert s.get(3).state == "failed"


def test_unknown_step_no_is_ignored():
    s = StageStack.for_mode("screen")
    s.apply(_Ev("step.started", "Step 99", payload={"step": 99}))
    for step in s.steps:
        assert step.state == "pending"


def test_run_mode_step_25_does_not_collide_with_step_1():
    """In run mode the analyze sub-stage emits ``step=2.5`` for
    "读取 watchlist". Asserting it lands on a dedicated slot is the regression
    test for 《APW run 空结果问题修复方案》—— previously the float was discarded
    and the read-watchlist message overwrote screen's done Step 1.
    """
    s = StageStack.for_mode("run")
    # Screen finished its Step 1 / Step 2 first.
    s.apply(_Ev("step.started", "Step 1", payload={"step": 1}))
    s.apply(_Ev("step.finished", "Step 1 done", payload={"step": 1}))
    s.apply(_Ev("step.started", "Step 2", payload={"step": 2}))
    s.apply(_Ev("step.finished", "Step 2 done", payload={"step": 2}))

    # Analyze (as run sub-stage) emits step=2.5 for read-watchlist.
    s.apply(_Ev("step.started", "读取 watchlist", payload={"step": 2.5}))
    assert s.get(2.5) is not None, "run-mode StageStack must contain Step 2.5"
    assert s.get(2.5).state == "running"
    # Screen's prior steps stay done — no collision.
    assert s.get(1).state == "done"
    assert s.get(2).state == "done"

    s.apply(_Ev("step.finished", "读取 watchlist 完成", payload={"step": 2.5}))
    assert s.get(2.5).state == "done"


def test_analyze_standalone_step_1_still_means_read_watchlist():
    """Standalone analyze keeps step=1 for read-watchlist (back-compat)."""
    s = StageStack.for_mode("analyze")
    s.apply(_Ev("step.started", "读取 watchlist", payload={"step": 1}))
    assert s.get(1).state == "running"
    assert s.get(1).label == "读取 watchlist"
