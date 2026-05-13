"""Legacy stream renderer — preserves v0.5.x stdout byte-for-byte.

This is the default renderer when the dashboard cannot run (non-TTY, CI,
``--no-dashboard``, ``DEEPTRADE_NO_DASHBOARD=1``, ``TERM=dumb``) and the
safe fallback when the rich dashboard raises mid-run. Output here is
parsed by user scripts and CI greps, so the line format is frozen:

    ``  {glyph} [{event_type}] {message}``

with ``glyph`` ∈ ``✔ ⚠ ✘`` driven by ``StrategyEvent.level``. Any
deviation breaks the v0.6 compatibility promise (Plan §4.3, §10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deeptrade.plugins_api.events import EventLevel

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from limit_up_board.runner import RunOutcome, RunParams


class LegacyStreamRenderer:
    """Prints each event as a single stdout line in the v0.5.x format."""

    def on_run_start(
        self, *, run_id: str, params: RunParams, debate: bool
    ) -> None:
        # No banner — keeping stdout byte-identical to v0.5.x means run_start
        # and run_finish are silent here. The CLI prints the post-run status
        # line and the terminal summary separately.
        return None

    def on_event(self, ev: StrategyEvent) -> None:
        if ev.level == EventLevel.INFO:
            glyph = "✔"
        elif ev.level == EventLevel.WARN:
            glyph = "⚠"
        else:
            glyph = "✘"
        print(f"  {glyph} [{ev.type.value}] {ev.message}", flush=True)

    def on_run_finish(self, outcome: RunOutcome) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = ["LegacyStreamRenderer"]
