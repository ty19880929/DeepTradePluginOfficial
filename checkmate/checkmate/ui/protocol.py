"""Renderer protocol + the tiny ``RenderEvent`` carrier.

Iter-5 PR-5.1 ships the contract here; the legacy renderer lands beside it
in :mod:`checkmate.ui.legacy`; the Rich dashboard arrives in PR-5.2.

Why a checkmate-internal :class:`RenderEvent` instead of reusing the
framework's ``deeptrade.plugins_api.events.StrategyEvent``: it keeps the UI
layer importable in tests without the framework on PYTHONPATH, and it lets
the runner pass plain dataclass payloads through without converting types.
Migrating to ``StrategyEvent`` is a one-method rename if v0.5+ stabilises
the framework's event schema.

Lifecycle contract — enforced by the orchestrators (scan / signals /
backtest):

* ``on_run_start`` — exactly once before any ``on_event``.
* ``on_event`` — 0..N times, in pipeline-emit order.
* ``on_run_finish`` — exactly once after the last event, including on
  ``KeyboardInterrupt`` / unexpected exception paths.
* ``close`` — always called from a ``finally`` block; must be idempotent.

Implementations should not raise from ``on_event``: the orchestrator wraps
the call in ``try / except`` and degrades to :class:`LegacyStreamRenderer`
if the contract is violated, but the renderer should still swallow its own
errors internally.

The ``mode`` parameter on :meth:`on_run_start` differentiates the three
long-running commands (``scan`` / ``signals`` / ``backtest``); the
dashboard (PR-5.2) uses it to pick stage titles + show / hide
mode-specific cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RenderEvent:
    """Plain-data event payload passed to :meth:`EventRenderer.on_event`."""

    type: str          # e.g. ``RUN_STARTED`` / ``STEP_STARTED`` / ``STEP_FINISHED``
    message: str
    level: str = "info"  # ``info`` / ``warn`` / ``error``
    payload: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EventRenderer(Protocol):
    """UI rendering hook used by the long-running checkmate runners."""

    def on_run_start(self, *, run_id: str, mode: str, params: Any) -> None: ...

    def on_event(self, ev: RenderEvent) -> None: ...

    def on_run_finish(self, outcome: Any) -> None: ...

    def close(self) -> None: ...


class NullRenderer:
    """Silently drops every event — tests' workhorse."""

    def on_run_start(self, *, run_id: str, mode: str, params: Any) -> None:  # noqa: ARG002
        return None

    def on_event(self, ev: RenderEvent) -> None:  # noqa: ARG002
        return None

    def on_run_finish(self, outcome: Any) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None


__all__ = ["EventRenderer", "NullRenderer", "RenderEvent"]
