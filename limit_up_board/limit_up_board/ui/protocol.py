"""The ``EventRenderer`` Protocol and the trivial ``NullRenderer`` impl."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from limit_up_board.runner import RunOutcome, RunParams


@runtime_checkable
class EventRenderer(Protocol):
    """UI rendering hook used by ``LubRunner``.

    Lifecycle contract (enforced by the runner):

    * ``on_run_start`` is called **exactly once** before any event.
    * ``on_event`` is called **0..N** times, in pipeline-emit order.
    * ``on_run_finish`` is called **exactly once** after the last event,
      including on KeyboardInterrupt / unexpected exception paths.
    * ``close`` is called from a ``finally`` block, *always*, and must be
      idempotent — implementations should release terminal state (cursor,
      ANSI mode, alt-screen, etc.) here.

    Implementations **must not** raise from ``on_event``: the runner installs
    a defensive ``try/except`` around the call and will degrade to
    :class:`LegacyStreamRenderer` if the contract is violated (see
    ``LubRunner._dispatch_to_renderer``), but the renderer should still aim
    to swallow its own errors internally.
    """

    def on_run_start(
        self, *, run_id: str, params: RunParams, debate: bool
    ) -> None: ...

    def on_event(self, ev: StrategyEvent) -> None: ...

    def on_run_finish(self, outcome: RunOutcome) -> None: ...

    def close(self) -> None: ...


class NullRenderer:
    """Silently drops every event.

    Used by tests that want to drive ``LubRunner`` end-to-end without caring
    about stdout — and by the runner's ``__init__`` default when a caller
    forgets to inject a renderer (defensive only; ``cli.py`` always passes
    one through ``choose_renderer``).
    """

    def on_run_start(
        self, *, run_id: str, params: RunParams, debate: bool
    ) -> None:
        return None

    def on_event(self, ev: StrategyEvent) -> None:
        return None

    def on_run_finish(self, outcome: RunOutcome) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = ["EventRenderer", "NullRenderer"]
