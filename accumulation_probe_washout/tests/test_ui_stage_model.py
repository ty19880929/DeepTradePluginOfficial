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
    assert len(StageStack.for_mode("run").steps) == 5


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
