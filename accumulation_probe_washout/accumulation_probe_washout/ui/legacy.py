"""LegacyStreamRenderer — line-per-event stdout fallback (M2 + M4 baseline).

Format is byte-stable so users / CI logs don't churn between releases:
    {glyph} [{event_type}] {message}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.run_status import RunStatus
    from deeptrade.plugins_api.events import StrategyEvent


_GLYPHS: dict[str, str] = {
    "step.started": "▶",
    "step.progress": "·",
    "step.finished": "✓",
    "data.sync.started": "↻",
    "data.sync.finished": "✓",
    "tushare.call": "·",
    "tushare.fallback": "⚠",
    "tushare.unauthorized": "✘",
    "llm.batch.started": "▶",
    "llm.batch.finished": "✓",
    "llm.final_ranking": "★",
    "validation.failed": "⚠",
    "result.persisted": "✓",
    "log": "·",
    "live.status": "·",
}


class LegacyStreamRenderer:
    """Emit one line per StrategyEvent. Renderer-side state is intentionally None."""

    def on_run_started(
        self,
        *,
        mode: str,
        trade_date: str,
        run_id: str,
        params: dict[str, Any],
    ) -> None:
        typer.echo(f"\n=== accumulation-probe-washout {mode} (T={trade_date}, run_id={run_id}) ===")

    def on_event(self, ev: StrategyEvent) -> None:
        glyph = _GLYPHS.get(ev.type.value, "·")
        typer.echo(f"  {glyph} [{ev.type.value}] {ev.message}")

    def on_run_finished(
        self,
        *,
        status: RunStatus,
        error: str | None,
        summary: dict[str, Any],
    ) -> None:
        if status.value == "cancelled":
            typer.echo("  ⏹ 用户手动中断，已停止当前策略执行。")
        elif error:
            typer.echo(f"  ✘ run finished: {status.value} — {error}")
        else:
            typer.echo(f"  ✓ run finished: {status.value}")
