"""StageStack — derives stage progress from a StrategyEvent stream.

Each mode's StageStack has a fixed set of steps; events advance them based on
payload['step'] (when present) or fallback to event-type heuristics.

Mode → step list:
  screen  : Step 0 数据同步, Step 1 漏斗筛选, Step 2 持久化
  analyze : Step 0 数据同步, Step 1 读取 watchlist, Step 3 LLM, Step 5 写入
  run     : Step 0 数据同步, Step 1 漏斗筛选, Step 2 持久化命中, Step 3 LLM, Step 5 写入
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent


@dataclass
class Step:
    no: int
    label: str
    state: str = "pending"  # pending / running / done / failed
    message: str = ""


_SCREEN_STEPS = [(0, "数据同步"), (1, "漏斗筛选"), (2, "持久化命中")]
_ANALYZE_STEPS = [(0, "数据同步"), (1, "读取 watchlist"), (3, "LLM 走势分析"), (5, "写入结果")]
_RUN_STEPS = [
    (0, "数据同步"), (1, "漏斗筛选"), (2, "持久化命中"),
    (3, "LLM 走势分析"), (5, "写入结果"),
]


@dataclass
class StageStack:
    mode: str = "screen"
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def for_mode(cls, mode: str) -> "StageStack":
        if mode == "analyze":
            tpl = _ANALYZE_STEPS
        elif mode == "run":
            tpl = _RUN_STEPS
        else:
            tpl = _SCREEN_STEPS
        return cls(mode=mode, steps=[Step(no=n, label=l) for n, l in tpl])

    def get(self, step_no: int) -> Step | None:
        for s in self.steps:
            if s.no == step_no:
                return s
        return None

    def apply(self, ev: StrategyEvent) -> None:
        payload = ev.payload or {}
        step_no = payload.get("step")
        et = ev.type.value

        # Data sync is its own pair of events
        if et == "data.sync.started":
            s = self.get(0)
            if s:
                s.state = "running"
                s.message = ev.message
            return
        if et == "data.sync.finished":
            s = self.get(0)
            if s:
                s.state = "done"
                s.message = ev.message
            return

        if et == "step.started" and isinstance(step_no, int):
            s = self.get(step_no)
            if s:
                s.state = "running"
                s.message = ev.message
        elif et == "step.finished" and isinstance(step_no, int):
            s = self.get(step_no)
            if s:
                s.state = "done"
                s.message = ev.message
        elif et == "validation.failed":
            # Mark the active step as failed
            for s in self.steps:
                if s.state == "running":
                    s.state = "failed"
                    s.message = ev.message
                    break
