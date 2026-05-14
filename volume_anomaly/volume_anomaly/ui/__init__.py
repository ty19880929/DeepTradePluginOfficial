"""UI rendering subsystem for the ``volume-anomaly`` runner.

This package owns everything that turns ``StrategyEvent`` instances into
terminal output. It is the **only** layer allowed to touch presentation; the
runner / pipeline / data / lgb subsystems are strict producers of events and
must remain unaware of how (or whether) events are rendered.

Implementations (Plan §3.1.2):

* :class:`LegacyStreamRenderer` — byte-identical to v0.7.x; the safe default
  and fallback for non-TTY / CI / ``--no-dashboard``.
* :class:`RichDashboardRenderer` — animated dashboard for ``screen`` /
  ``analyze`` modes (Plan §4.1, §4.2). **Introduced in PR-2** — PR-1 only
  ships the protocol.
* :class:`NullRenderer` — silent; testing only.

:func:`choose_renderer` is the single factory CLI callers use. Fallback
rules per Plan §3.5.

PR-1 scope (no dashboard yet): :func:`choose_renderer` always returns a
``LegacyStreamRenderer`` so stdout stays byte-identical to v0.7.x. PR-2
extends it to apply the full §3.5 fallback matrix and instantiate
``RichDashboardRenderer`` when the environment supports it.
"""

from __future__ import annotations

import os
import sys

from .legacy import LegacyStreamRenderer
from .protocol import EventRenderer, NullRenderer

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(value: str | None) -> bool:
    """``True`` iff ``value`` is one of the strings users expect to mean true."""
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def choose_renderer(*, no_dashboard: bool = False) -> EventRenderer:
    """Return the renderer best suited for the current environment.

    PR-1 always returns :class:`LegacyStreamRenderer` to keep stdout
    byte-identical to v0.7.x while the protocol scaffolding lands. PR-2
    extends this factory to apply the full Plan §3.5 fallback matrix:

    * caller passed ``no_dashboard=True``,
    * ``sys.stdout`` is not a TTY (pipe / redirect / pytest capture),
    * ``CI`` env var is truthy,
    * ``DEEPTRADE_NO_DASHBOARD`` env var is truthy,
    * ``TERM == "dumb"``,

    → otherwise ``RichDashboardRenderer`` (with ``NO_COLOR`` honoured).

    The eager TTY / env probing is already wired here so PR-2 only adds the
    final dashboard branch; behaviour-equivalent today.
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
    # PR-2 wires the dashboard here; until then we stay on legacy so stdout
    # is unchanged from v0.7.x.
    return LegacyStreamRenderer()


__all__ = [
    "EventRenderer",
    "LegacyStreamRenderer",
    "NullRenderer",
    "choose_renderer",
]
