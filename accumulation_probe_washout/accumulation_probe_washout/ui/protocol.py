"""EventRenderer protocol — minimum surface the runner depends on.

Concrete renderers (Legacy / RichDashboard) land in M4. M2 only needs the
protocol + a ``NullRenderer`` so the runner can run headlessly in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.run_status import RunStatus
    from deeptrade.plugins_api.events import StrategyEvent


@runtime_checkable
class EventRenderer(Protocol):
    """All renderers must implement this surface."""

    def on_run_started(
        self, *, mode: str, trade_date: str, run_id: str, params: dict[str, Any]
    ) -> None: ...

    def on_event(self, ev: StrategyEvent) -> None: ...

    def on_run_finished(
        self, *, status: RunStatus, error: str | None, summary: dict[str, Any]
    ) -> None: ...


class NullRenderer:
    """No-op renderer used by tests and when no UI is wired up."""

    def on_run_started(
        self, *, mode: str, trade_date: str, run_id: str, params: dict[str, Any]
    ) -> None:
        return

    def on_event(self, ev: StrategyEvent) -> None:
        return

    def on_run_finished(
        self, *, status: RunStatus, error: str | None, summary: dict[str, Any]
    ) -> None:
        return
