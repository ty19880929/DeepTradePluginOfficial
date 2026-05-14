"""Dynamic stage stack consumed by the dashboard.

Plan §3.2 — the stage list is **event-driven** rather than a hard-coded set
of N steps. ``STEP_STARTED`` (or ``DATA_SYNC_STARTED`` in screen mode) adds
(or reuses) a stage; ``STEP_FINISHED`` (or ``DATA_SYNC_FINISHED``) closes
it; ``LLM_BATCH_FINISHED`` ticks progress; ``VALIDATION_FAILED`` records a
failed batch on whichever stage is currently running. The ordering of
stages in the dashboard matches the order they were first ``push``-ed,
which mirrors pipeline emit order.

No rich / printing concerns live here — this module is pure data so it
stays easy to test (Plan §7.1 layer L1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StageStatus(str, Enum):
    """States a stage cycles through during a run.

    ``PARTIAL`` reads as "stage finished, but some batches inside it failed";
    ``SKIPPED`` is for conditional stages (e.g. empty-watchlist analyze).
    """

    WAITING = "waiting"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Stage:
    stage_id: str
    title: str
    status: StageStatus = StageStatus.WAITING
    detail: str = ""
    progress_done: int = 0
    progress_total: int | None = None
    failed_batches: list[str] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class StageStack:
    """Ordered list of stages with O(1) lookup by id.

    Internal invariant: ``stages`` is ordered by insertion time; ``_index``
    mirrors it for ``push_or_get`` lookups. Tests assert ordering against
    pipeline emit order, so do not sort — append-only.
    """

    stages: list[Stage] = field(default_factory=list)
    _index: dict[str, Stage] = field(default_factory=dict)

    def push_or_get(self, stage_id: str, title: str) -> Stage:
        if stage_id in self._index:
            return self._index[stage_id]
        st = Stage(stage_id=stage_id, title=title)
        self.stages.append(st)
        self._index[stage_id] = st
        return st

    def get(self, stage_id: str) -> Stage | None:
        return self._index.get(stage_id)

    def set_running(
        self,
        stage_id: str,
        *,
        total: int | None = None,
        now: float | None = None,
    ) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        st.status = StageStatus.RUNNING
        if total is not None:
            st.progress_total = total
        if now is not None and st.started_at is None:
            st.started_at = now

    def set_detail(self, stage_id: str, detail: str) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        st.detail = detail

    def tick_progress(self, stage_id: str) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        st.progress_done += 1

    def append_failed_batch(self, stage_id: str, batch_label: str) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        # Keep only the most recent 3 to avoid an unbounded list in a long
        # multi-batch run (the UI also clips at 3 — see layout.py).
        st.failed_batches.append(batch_label)
        if len(st.failed_batches) > 3:
            st.failed_batches = st.failed_batches[-3:]

    def mark_finished(
        self,
        stage_id: str,
        *,
        partial: bool = False,
        now: float | None = None,
    ) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        st.status = StageStatus.PARTIAL if partial else StageStatus.SUCCESS
        if now is not None:
            st.finished_at = now

    def mark_failed(self, stage_id: str, *, now: float | None = None) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        st.status = StageStatus.FAILED
        if now is not None:
            st.finished_at = now

    def mark_skipped(self, stage_id: str) -> None:
        st = self._index.get(stage_id)
        if st is None:
            return
        st.status = StageStatus.SKIPPED

    def latest_running(self) -> Stage | None:
        for st in reversed(self.stages):
            if st.status == StageStatus.RUNNING:
                return st
        return None


__all__ = ["Stage", "StageStack", "StageStatus"]
