"""UI rendering subsystem for the ``volume-anomaly`` runner.

This package owns everything that turns ``StrategyEvent`` instances into
terminal output. It is the **only** layer allowed to touch presentation; the
runner / pipeline / data / lgb subsystems are strict producers of events and
must remain unaware of how (or whether) events are rendered.

Implementations (Plan §3.1.2):

* :class:`LegacyStreamRenderer` — byte-identical to v0.7.x; the safe default
  and fallback for non-TTY / CI / ``--no-dashboard``.
* :class:`RichDashboardRenderer` — animated dashboard for ``screen`` /
  ``analyze`` modes (Plan §4.1, §4.2).
* :class:`NullRenderer` — silent; testing only.

:func:`choose_renderer` is the single factory CLI callers use. Fallback
rules per Plan §3.5.
"""

from __future__ import annotations

import os
import sys

from .legacy import LegacyStreamRenderer
from .protocol import EventRenderer, NullRenderer

# Lazy import — RichDashboardRenderer pulls in rich / theme; tests that
# don't need it should not pay the cost. We import inside the factory.

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(value: str | None) -> bool:
    """``True`` iff ``value`` is one of the strings users expect to mean true."""
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def choose_renderer(*, no_dashboard: bool = False) -> EventRenderer:
    """Return the renderer best suited for the current environment.

    Fallback to :class:`LegacyStreamRenderer` if *any* of these hold
    (Plan §3.5):

    * caller passed ``no_dashboard=True``,
    * ``sys.stdout`` is not a TTY (pipe / redirect / pytest capture),
    * ``CI`` env var is truthy,
    * ``DEEPTRADE_NO_DASHBOARD`` env var is truthy,
    * ``TERM == "dumb"``.

    Otherwise return :class:`RichDashboardRenderer`. ``NO_COLOR`` is
    respected by toggling the Console's ``no_color`` flag — the dashboard
    still runs (structured layout helps even mono users), it just renders
    without ANSI colour (Plan §3.5.1, https://no-color.org).

    Callers may further restrict the dashboard to specific subcommands by
    bypassing this factory and constructing :class:`LegacyStreamRenderer`
    directly (Plan §3.4.2: ``cmd_prune`` / ``cmd_evaluate`` do this).
    """
    if no_dashboard:
        return LegacyStreamRenderer()
    try:
        if not sys.stdout.isatty():
            return LegacyStreamRenderer()
    except Exception:  # noqa: BLE001 — defensive: some stdouts lack isatty
        return LegacyStreamRenderer()
    if _truthy(os.environ.get("CI")):
        return LegacyStreamRenderer()
    if _truthy(os.environ.get("DEEPTRADE_NO_DASHBOARD")):
        return LegacyStreamRenderer()
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return LegacyStreamRenderer()

    no_color = bool(os.environ.get("NO_COLOR", "").strip())

    from .dashboard import RichDashboardRenderer  # noqa: PLC0415

    try:
        return RichDashboardRenderer(no_color=no_color)
    except Exception:  # noqa: BLE001 — final safety net: never block a run
        return LegacyStreamRenderer()


__all__ = [
    "EventRenderer",
    "LegacyStreamRenderer",
    "NullRenderer",
    "choose_renderer",
]
