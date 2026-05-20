"""UI rendering subsystem for the ``checkmate`` long-running commands.

This package owns everything that turns :class:`RenderEvent` instances into
terminal output. It is the **only** layer allowed to touch presentation; the
scan / signals / backtest orchestrators are strict producers of events and
remain unaware of how (or whether) events are rendered.

Implementations:

* :class:`LegacyStreamRenderer` — byte-identical to Iter-4 stream format;
  the safe default and fallback for non-TTY / CI / ``--no-dashboard``.
* :class:`RichDashboardRenderer` — animated dashboard (lands in PR-5.2; the
  factory falls back gracefully when the module is absent).
* :class:`NullRenderer` — silent; testing only.

:func:`choose_renderer` is the single factory CLI callers use.
"""

from __future__ import annotations

import os
import sys

from .legacy import LegacyStreamRenderer
from .protocol import EventRenderer, NullRenderer, RenderEvent


_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def choose_renderer(
    *,
    no_dashboard: bool = False,
    mode: str = "scan",
) -> EventRenderer:
    """Return the renderer best suited for the current environment.

    Fallback to :class:`LegacyStreamRenderer` if *any* of these hold:

    * caller passed ``no_dashboard=True``,
    * ``sys.stdout`` is not a TTY (pipe / redirect / pytest capture),
    * ``CI`` env var is truthy,
    * ``DEEPTRADE_NO_DASHBOARD`` env var is truthy,
    * ``TERM == "dumb"``,
    * the Rich dashboard module is unavailable or raises on construction.

    Otherwise return :class:`RichDashboardRenderer` (added in PR-5.2). The
    ``mode`` parameter is forwarded to the dashboard so it can pick stage
    titles + show / hide mode-specific cards.
    """
    if no_dashboard:
        return LegacyStreamRenderer()
    try:
        if not sys.stdout.isatty():
            return LegacyStreamRenderer()
    except Exception:  # noqa: BLE001 — some stdouts lack isatty()
        return LegacyStreamRenderer()
    if _truthy(os.environ.get("CI")):
        return LegacyStreamRenderer()
    if _truthy(os.environ.get("DEEPTRADE_NO_DASHBOARD")):
        return LegacyStreamRenderer()
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return LegacyStreamRenderer()

    # PR-5.2 wires the RichDashboardRenderer here. Until then, the import
    # below raises ModuleNotFoundError → we degrade to legacy. This branch
    # is also the runtime safety net once the dashboard ships.
    try:
        from .dashboard import RichDashboardRenderer  # type: ignore[import-not-found]  # noqa: PLC0415

        return RichDashboardRenderer(mode=mode)  # type: ignore[call-arg]
    except Exception:  # noqa: BLE001 — never block a run
        return LegacyStreamRenderer()


__all__ = [
    "EventRenderer",
    "LegacyStreamRenderer",
    "NullRenderer",
    "RenderEvent",
    "choose_renderer",
]
