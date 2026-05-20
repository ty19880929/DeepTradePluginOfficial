"""StageStack — pipeline progress derived from a :class:`RenderEvent` stream.

Mode-specific step list (matches orchestrator emit messages):

* ``scan``     — Step 0 universe / 1 features / 2 regime
* ``signals``  — Step 1 entries / 2 exits / 3 risk filter
* ``backtest`` — single ``session loop`` row that cycles on each
  ``SESSION_FINISHED`` event (per-day pipeline runs all 6 stages
  internally; surfacing them per-day would be too noisy for the dashboard).

The state machine reads ``ev.type`` + a leading ``Step N:`` token in
``ev.message`` (the orchestrators already emit that prefix; see ``scan.py``
/ ``signals.py``). Adding ``payload={"step": N}`` upstream would let us
skip the regex, but the message-parse path is robust to schema drift and
matches APW's convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .protocol import RenderEvent


_STEP_RE = re.compile(r"^Step\s+(\d+)\b")


@dataclass
class Step:
    no: int
    label: str
    state: str = "pending"   # pending / running / done / failed
    message: str = ""


_SCAN_STEPS = [(0, "universe"), (1, "features"), (2, "regime")]
_SIGNALS_STEPS = [(1, "entries"), (2, "exits"), (3, "risk filter")]
_BACKTEST_STEPS = [(0, "session loop")]


@dataclass
class StageStack:
    mode: str = "scan"
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def for_mode(cls, mode: str) -> "StageStack":
        tpl = _SCAN_STEPS
        if mode == "signals":
            tpl = _SIGNALS_STEPS
        elif mode == "backtest":
            tpl = _BACKTEST_STEPS
        return cls(mode=mode, steps=[Step(no=n, label=l) for n, l in tpl])

    def get(self, step_no: int) -> Step | None:
        for s in self.steps:
            if s.no == step_no:
                return s
        return None

    def apply(self, ev: "RenderEvent") -> None:
        """Mutate step states from one event.

        Resilient to unknown types / freeform messages — anything we don't
        recognise leaves the stack unchanged.
        """
        et = ev.type
        msg = ev.message or ""

        # Failures: mark the running step as failed.
        if et in ("RUN_FAILED", "STEP_FAILED") or ev.level == "error":
            for s in self.steps:
                if s.state == "running":
                    s.state = "failed"
                    s.message = msg
                    return

        # Backtest session cycle: each SESSION_FINISHED marks the single
        # stack entry as done and rolls its label forward.
        if et == "SESSION_FINISHED" and self.mode == "backtest" and self.steps:
            self.steps[0].state = "done"
            self.steps[0].message = msg
            return

        m = _STEP_RE.match(msg)
        if not m:
            return
        step_no = int(m.group(1))
        s = self.get(step_no)
        if s is None:
            return
        if et == "STEP_STARTED":
            s.state = "running"
            s.message = msg
        elif et == "STEP_FINISHED":
            s.state = "done"
            s.message = msg


__all__ = ["Step", "StageStack"]
