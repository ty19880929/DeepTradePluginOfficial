"""RichDashboardRenderer — single Live region driven by RenderEvent stream.

``mode`` in ``{"scan", "signals", "backtest"}`` controls which mode-specific
cards appear:

* **scan** — Header / Config / Funnel / Stages / Log
* **signals** — Header / Config / Regime / Stages / Log
* **backtest** — Header / Config / Regime / Portfolio / Stages / Log

UI failures must NEVER crash the run: every Rich call lives inside ``try /
except`` so the dashboard degrades gracefully (the renderer becomes a
silent no-op rather than the orchestrator catching mid-stream). The
``choose_renderer`` factory still wraps construction in another ``try`` so
an outright import failure (e.g. ``rich`` missing) falls back to legacy.

For tests, construct with ``console=Console(record=True)`` (no Live) and
call :meth:`_render` directly to inspect the renderable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any

from rich.console import Console, Group
from rich.live import Live

from .layout import (
    render_config_card,
    render_funnel_card,
    render_header,
    render_log_panel,
    render_positions_card,
    render_regime_card,
    render_stage_stack,
)
from .protocol import RenderEvent
from .stage_model import StageStack

logger = logging.getLogger(__name__)


def _params_to_cfg_lines(params: Any) -> list[tuple[str, str]]:
    """Extract a handful of key/value strings from a Params dataclass."""
    if params is None:
        return []
    try:
        d = asdict(params) if is_dataclass(params) else dict(params)
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[str, str]] = []

    def _push(key: str, *fallbacks: str) -> None:
        for k in (key, *fallbacks):
            if k in d and d[k] is not None:
                out.append((key, str(d[k])))
                return

    _push("start")
    _push("end")
    _push("trade_date")
    _push("initial_cash")
    _push("portfolio_value")
    _push("resume")
    return out[:6]


class RichDashboardRenderer:
    """Owns the Live region for one run."""

    def __init__(self, *, mode: str = "scan", console: Console | None = None) -> None:
        self.mode = mode
        self.console = console or Console()
        self.stack = StageStack.for_mode(mode)
        self.log_lines: list[str] = []
        self.run_id: str | None = None
        self.start_ts: float | None = None
        self.cfg_lines: list[tuple[str, str]] = []
        self.regime_payload: dict[str, Any] = {}
        self.funnel_payload: dict[str, Any] = {}
        self.positions_payload: dict[str, Any] = {}
        self._live: Live | None = None

    # ---- lifecycle ----

    def on_run_start(self, *, run_id: str, mode: str, params: Any) -> None:
        self.mode = mode
        self.run_id = run_id
        self.start_ts = time.time()
        self.stack = StageStack.for_mode(mode)
        self.cfg_lines = _params_to_cfg_lines(params)
        # Start a Live region. If construction fails (eg console doesn't
        # support it, or rich changed its API), swallow + fall through to
        # render-on-demand mode.
        try:
            self._live = Live(
                self._render(),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start(refresh=True)
        except Exception:  # noqa: BLE001
            logger.warning("RichDashboardRenderer: Live failed to start; "
                           "events still update internal state.", exc_info=True)
            self._live = None

    def on_event(self, ev: RenderEvent) -> None:
        # Defensive: no event must crash the dashboard.
        try:
            self.stack.apply(ev)
            payload = ev.payload or {}

            # Funnel update — scan-mode Step 0 finish carries the universe
            # counts; Step 1 finish carries the features count.
            if ev.type == "STEP_FINISHED":
                if "total" in payload and "eligible" in payload:
                    self.funnel_payload.update({
                        k: payload.get(k, 0)
                        for k in ("total", "eligible", "excluded")
                    })
                if "features_rows" in payload:
                    self.funnel_payload["features_rows"] = payload["features_rows"]

            # Regime update — scan & signals & backtest Step 2 / 3 finishes.
            if "regime" in payload:
                self.regime_payload["regime"] = payload["regime"]
            if "exposure_cap" in payload:
                self.regime_payload["exposure_cap"] = payload["exposure_cap"]
            if "breadth_ma120" in payload:
                self.regime_payload["breadth_ma120"] = payload["breadth_ma120"]

            # Positions update — backtest SESSION_FINISHED.
            if ev.type == "SESSION_FINISHED":
                self.positions_payload = {
                    k: payload.get(k) for k in
                    ("trade_date", "equity", "drawdown_pct", "n_fills",
                     "regime", "n_cancels")
                }

            self.log_lines.append(f"[{ev.type}] {ev.message}")
            if self._live is not None:
                self._live.update(self._render())
        except Exception:  # noqa: BLE001
            # Never propagate UI exceptions back into the orchestrator.
            logger.exception("dashboard on_event failed for %s", ev.type)

    def on_run_finish(self, outcome: Any) -> None:
        try:
            self.log_lines.append("=== run finished")
            if self._live is not None:
                self._live.update(self._render())
        except Exception:  # noqa: BLE001
            logger.exception("dashboard on_run_finish failed")
        self.close()

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001
                pass
            self._live = None

    # ---- render ----

    def _elapsed(self) -> str:
        if self.start_ts is None:
            return "—"
        d = int(time.time() - self.start_ts)
        m, s = divmod(d, 60)
        return f"{m:02d}:{s:02d}"

    def _render(self) -> Group:
        parts: list[Any] = []
        parts.append(render_header(
            mode=self.mode, run_id=self.run_id or "—",
            elapsed=self._elapsed(),
        ))
        if self.cfg_lines:
            parts.append(render_config_card(self.cfg_lines))
        if self.mode in ("signals", "backtest") and self.regime_payload:
            parts.append(render_regime_card(self.regime_payload))
        if self.mode == "scan" and self.funnel_payload:
            parts.append(render_funnel_card(self.funnel_payload))
        if self.mode == "backtest" and self.positions_payload:
            parts.append(render_positions_card(self.positions_payload))
        parts.append(render_stage_stack(self.stack))
        parts.append(render_log_panel(self.log_lines))
        return Group(*parts)


__all__ = ["RichDashboardRenderer"]
