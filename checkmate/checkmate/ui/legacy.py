"""Legacy stream renderer — preserves Iter-4 stdout byte-for-byte.

Default renderer when the dashboard cannot run (non-TTY, CI, ``--no-dashboard``,
``DEEPTRADE_NO_DASHBOARD=1``, ``TERM=dumb``); the safe fallback when the rich
dashboard raises mid-run.

Output format (stable contract for user scripts / CI greps)::

    ``  {glyph} [{event_type}] {message}``

with ``glyph`` ∈ ``✔ ⚠ ✘`` driven by :attr:`RenderEvent.level`. Any
deviation breaks the v0.1.x compatibility promise.
"""

from __future__ import annotations

from typing import Any, Callable

from .protocol import RenderEvent


_GLYPHS: dict[str, str] = {
    "info": "✔",
    "warn": "⚠",
    "error": "✘",
}


class LegacyStreamRenderer:
    """Prints each event as a single line to ``sink`` (default: stdout).

    The ``sink`` indirection lets tests capture the stream by passing
    ``list.append``; CLI callers leave the default ``print`` (flush=True via
    a thin shim). Iter-4 used a hand-rolled ``echo=print`` parameter on the
    orchestrators; v0.2.0 routes everything through this renderer so the
    on-disk format stays in one place.
    """

    def __init__(self, sink: Callable[[str], None] | None = None) -> None:
        # Default: stdout via builtin print, flushed so CI pipes see lines live.
        self._sink: Callable[[str], None] = sink or (lambda line: print(line, flush=True))

    def on_run_start(self, *, run_id: str, mode: str, params: Any) -> None:  # noqa: ARG002
        # Silent — runner / CLI prints its own banners around the stream.
        return None

    def on_event(self, ev: RenderEvent) -> None:
        glyph = _GLYPHS.get(ev.level, "✔")
        self._sink(f"  {glyph} [{ev.type}] {ev.message}")

    def on_run_finish(self, outcome: Any) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None


__all__ = ["LegacyStreamRenderer"]
