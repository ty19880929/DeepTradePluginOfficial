"""RichDashboardRenderer — single Live region driven by StrategyEvent stream.

mode in {"screen", "analyze", "run"}; screen/run also render the funnel card.
UI failures must NEVER crash the run — caller wraps on_event in try/except and
degrades to LegacyStreamRenderer if anything raises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live

from .layout import (
    render_funnel,
    render_header,
    render_log,
    render_stage_stack,
)
from .stage_model import StageStack

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.run_status import RunStatus
    from deeptrade.plugins_api.events import StrategyEvent


class RichDashboardRenderer:
    """Owns the Live region for one run."""

    def __init__(self, *, mode: str = "screen", console: Console | None = None) -> None:
        self.mode = mode
        self.console = console or Console()
        self.stack = StageStack.for_mode(mode)
        self.funnel_payload: dict[str, Any] = {}
        self.log_lines: list[str] = []
        self._header = render_header(mode, "—", "—")
        self._live: Live | None = None

    # ---- lifecycle ----

    def on_run_started(
        self, *, mode: str, trade_date: str, run_id: str, params: dict[str, Any]
    ) -> None:
        self.mode = mode
        self.stack = StageStack.for_mode(mode)
        self._header = render_header(mode, trade_date, run_id)
        self.run_id = run_id
        self.trade_date = trade_date
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start(refresh=True)

    def on_event(self, ev: StrategyEvent) -> None:
        # Update stage stack
        self.stack.apply(ev)
        # Capture funnel payload on data sync finished
        if ev.type.value == "data.sync.finished":
            self.funnel_payload.update(ev.payload or {})
        if ev.type.value == "step.finished" and (ev.payload or {}).get("step") == 1:
            # screen funnel detail comes from Step 1 too
            self.funnel_payload.update(ev.payload or {})
        # Log line
        line = f"[{ev.type.value}] {ev.message}"
        self.log_lines.append(line)
        if self._live is not None:
            self._live.update(self._render())

    def on_run_finished(
        self, *, status: RunStatus, error: str | None, summary: dict[str, Any]
    ) -> None:
        if self._live is not None:
            self.log_lines.append(f"=== run finished: {status.value}")
            if error:
                self.log_lines.append(f"error: {error}")
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    # ---- render ----

    def _render(self) -> Group:
        parts: list = [self._header, render_stage_stack(self.stack)]
        if self.mode in ("screen", "run") and self.funnel_payload:
            parts.append(render_funnel(self.funnel_payload))
        parts.append(render_log(self.log_lines))
        return Group(*parts)
