"""Debate-mode summary grid.

Plan §3.4 + §4.2 — the v1 decision is *not* a per-provider concurrent
swimlane (since worker events arrive in bursts at ``as_completed`` time, a
"realtime" per-batch progress would lie to the user). Instead we render a
provider × phase status matrix with R1/R2 yield counts in the body.

State updates feed off the tagged events that the runner re-emits from
worker buffers (``runner._tag_event`` adds ``llm_provider`` and
``debate_phase`` to payload). The dashboard routes any event carrying
``llm_provider`` here; everything else stays with the StageStack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.table import Table

from .stage_model import StageStatus

if TYPE_CHECKING:  # pragma: no cover
    from rich.console import RenderableType


_STATUS_GLYPHS: dict[StageStatus, str] = {
    StageStatus.WAITING: "○",
    StageStatus.RUNNING: "⠋",
    StageStatus.SUCCESS: "✔",
    StageStatus.PARTIAL: "✔",
    StageStatus.FAILED: "✘",
    StageStatus.SKIPPED: "—",
}

_STATUS_STYLES: dict[StageStatus, str] = {
    StageStatus.WAITING: "status.pending",
    StageStatus.RUNNING: "status.running",
    StageStatus.SUCCESS: "status.success",
    StageStatus.PARTIAL: "status.success",
    StageStatus.FAILED: "status.error",
    StageStatus.SKIPPED: "status.pending",
}


@dataclass
class DebateRow:
    """One row in the debate summary grid (a single LLM provider)."""

    provider: str
    phase_a_status: StageStatus = StageStatus.WAITING
    phase_b_status: StageStatus = StageStatus.WAITING
    r1_count: int | None = None
    r2_count: int | None = None
    revised_count: int | None = None
    note: str = ""  # error message or short remark


@dataclass
class DebateGrid:
    """Ordered list of provider rows + phase banner.

    The grid mirrors the order in which providers were declared
    (``runner._select_debate_providers``) — the LOG event emitted at run
    start (``[辩论模式] 启用，参与 LLM = [...]``) carries that list in
    ``payload["providers"]`` and is the canonical seed.
    """

    rows: list[DebateRow] = field(default_factory=list)
    _index: dict[str, DebateRow] = field(default_factory=dict)
    banner: str | None = None
    current_phase: str = "init"  # "init" / "phase_a" / "phase_b" / "done"

    def seed(self, providers: list[str]) -> None:
        for name in providers:
            self.row_for(name)

    def row_for(self, provider: str) -> DebateRow:
        if provider not in self._index:
            row = DebateRow(provider=provider)
            self.rows.append(row)
            self._index[provider] = row
        return self._index[provider]

    def transition_to_phase(self, phase: str) -> None:
        """Move all rows that are still running in the previous phase to
        SUCCESS (or leave them if they're already FAILED). Called when the
        runner emits the next phase's banner LIVE_STATUS."""
        if phase == "phase_b":
            for row in self.rows:
                if row.phase_a_status == StageStatus.RUNNING:
                    row.phase_a_status = StageStatus.SUCCESS
        elif phase == "done":
            for row in self.rows:
                if row.phase_a_status == StageStatus.RUNNING:
                    row.phase_a_status = StageStatus.SUCCESS
                if row.phase_b_status == StageStatus.RUNNING:
                    row.phase_b_status = StageStatus.SUCCESS
        self.current_phase = phase


def _cell_for_phase_a(row: DebateRow) -> tuple[str, str]:
    glyph = _STATUS_GLYPHS[row.phase_a_status]
    style = _STATUS_STYLES[row.phase_a_status]
    if row.phase_a_status == StageStatus.WAITING:
        return f"{glyph} 等待", style
    if row.phase_a_status == StageStatus.RUNNING:
        return f"{glyph} 运行中", style
    if row.phase_a_status == StageStatus.FAILED:
        return f"{glyph} 失败", style
    # SUCCESS / PARTIAL — show R1/R2 counts (None → "?" for safety)
    r1 = "?" if row.r1_count is None else str(row.r1_count)
    r2 = "?" if row.r2_count is None else str(row.r2_count)
    suffix = "⚠" if row.phase_a_status == StageStatus.PARTIAL else ""
    return f"{glyph}{suffix} R1={r1} R2={r2}", style


def _cell_for_phase_b(row: DebateRow) -> tuple[str, str]:
    # If phase A failed, phase B never ran — render as em-dash.
    if row.phase_a_status == StageStatus.FAILED:
        return "—", _STATUS_STYLES[StageStatus.SKIPPED]
    glyph = _STATUS_GLYPHS[row.phase_b_status]
    style = _STATUS_STYLES[row.phase_b_status]
    if row.phase_b_status == StageStatus.WAITING:
        return f"{glyph} 等待", style
    if row.phase_b_status == StageStatus.RUNNING:
        return f"{glyph} 运行中", style
    if row.phase_b_status == StageStatus.FAILED:
        return f"{glyph} 失败", style
    rev = "?" if row.revised_count is None else str(row.revised_count)
    return f"{glyph} 修订 {rev} 只", style


def render_grid_table(grid: DebateGrid) -> Table:
    """Build the rich Table that goes in the debate summary panel."""
    table = Table(show_header=True, header_style="table.header", expand=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Phase A (R1+R2)")
    table.add_column("R3 修订")
    table.add_column("备注", overflow="fold")

    for row in grid.rows:
        pa_text, pa_style = _cell_for_phase_a(row)
        pb_text, pb_style = _cell_for_phase_b(row)
        note = row.note or "—"
        # Trim long notes — full text is in the log panel anyway.
        if len(note) > 40:
            note = note[:37] + "…"
        table.add_row(
            row.provider,
            f"[{pa_style}]{pa_text}[/{pa_style}]",
            f"[{pb_style}]{pb_text}[/{pb_style}]",
            note,
        )

    return table


__all__ = ["DebateGrid", "DebateRow", "render_grid_table"]
