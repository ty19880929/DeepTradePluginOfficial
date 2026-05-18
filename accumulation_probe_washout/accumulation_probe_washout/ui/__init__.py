"""UI rendering factory.

M2 ships a minimal ``choose_renderer`` returning ``NullRenderer`` — concrete
``LegacyStreamRenderer`` and ``RichDashboardRenderer`` land in M4. The factory
shape (no_dashboard + mode) is locked here so M4 can fill it in without
touching callers.
"""

from __future__ import annotations

import os
import sys

from .legacy import LegacyStreamRenderer
from .protocol import EventRenderer, NullRenderer


def choose_renderer(*, no_dashboard: bool = False, mode: str = "screen") -> EventRenderer:
    """Pick the right renderer for the current execution context.

    Fallback rules (mirrors VA / LUB):
      * caller passed ``no_dashboard=True``
      * ``CI`` env truthy
      * ``DEEPTRADE_NO_DASHBOARD`` env truthy
      * ``TERM=dumb``
      * ``sys.stdout`` is not a TTY (pipes / pytest capture / redirects)
    → fall back to the line-per-event ``LegacyStreamRenderer``. Until M4 lands
    the rich dashboard, "non-legacy" still means legacy too.
    """
    if no_dashboard:
        return LegacyStreamRenderer()
    if os.environ.get("CI"):
        return LegacyStreamRenderer()
    if os.environ.get("DEEPTRADE_NO_DASHBOARD"):
        return LegacyStreamRenderer()
    if os.environ.get("TERM") == "dumb":
        return LegacyStreamRenderer()
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return LegacyStreamRenderer()

    # On a real TTY in screen / analyze / run modes — use the rich dashboard.
    try:
        from .dashboard import RichDashboardRenderer  # noqa: PLC0415
        return RichDashboardRenderer(mode=mode)
    except Exception:  # pragma: no cover - defensive
        return LegacyStreamRenderer()


__all__ = ["EventRenderer", "LegacyStreamRenderer", "NullRenderer", "choose_renderer"]
