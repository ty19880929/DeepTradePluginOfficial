"""Rendering helpers that turn the dashboard state into ``rich`` objects.

The :class:`RichDashboardRenderer` owns the live region; this module owns
*what* gets drawn inside it. Keeping the two split makes it possible to
test the rendered output piece-wise without needing a ``Live`` context.

Layout goal (Plan §4.1):

    ┌─ DeepTrade · 打板策略 ───────┐
    │ run_id / start time          │
    ├─ 运行配置 ───────────────────┤
    │ 交易日期 / 筛选 / LGB        │
    ├─ 执行进度 ───────────────────┤
    │ ✔ 阶段 0..N + detail/progress│
    ├─ 日志 ───────────────────────┤
    │ recent INFO/WARN/ERROR lines │
    └──────────────────────────────┘

When the terminal is narrower than 80 cols, we render a *compact* form
without panel borders (Plan §4.4).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .debate_view import DebateGrid, render_grid_table
from .stage_model import Stage, StageStack, StageStatus

if TYPE_CHECKING:  # pragma: no cover
    from rich.console import RenderableType


# ---------------------------------------------------------------------------
# Status presentation
# ---------------------------------------------------------------------------


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

_STATUS_LABELS: dict[StageStatus, str] = {
    StageStatus.WAITING: "等待",
    StageStatus.RUNNING: "运行",
    StageStatus.SUCCESS: "完成",
    StageStatus.PARTIAL: "完成⚠",
    StageStatus.FAILED: "失败",
    StageStatus.SKIPPED: "跳过",
}


# ---------------------------------------------------------------------------
# Dashboard state inputs
# ---------------------------------------------------------------------------


@dataclass
class ConfigSummary:
    """Distilled run-configuration row shown in the top panel.

    Populated incrementally from the ``LOG`` event emitted by
    :func:`limit_up_board.runner._settings_log_event` (Plan §3.3 mapping)
    and from ``Step 0: STEP_FINISHED.payload`` for the trade date. Any
    field left ``None`` is hidden — the dashboard renders only what it
    knows.
    """

    trade_date: str | None = None
    next_trade_date: str | None = None
    min_float_mv_yi: float | None = None
    max_float_mv_yi: float | None = None
    max_close_yuan: float | None = None
    lgb_enabled: bool | None = None
    tushare_fallback_count: int = 0


@dataclass
class DashboardState:
    """The full state the renderer needs to draw a frame.

    Owned by :class:`RichDashboardRenderer`; this module only reads.
    """

    run_id: str = ""
    started_at: datetime | None = None
    plugin_version: str = "?"
    debate: bool = False
    config: ConfigSummary = field(default_factory=ConfigSummary)
    stages: StageStack = field(default_factory=StageStack)
    # Debate-mode grid; ``None`` outside of debate mode (Plan §3.4 / §4.2).
    debate_grid: DebateGrid | None = None
    # Log ring buffer — keeps the most recent N lines for the bottom panel.
    # Bumped to 12 in v0.6.5 so an error traceback (≈ 5–10 lines) fits in-frame.
    log_lines: deque[tuple[str, str, str]] = field(
        default_factory=lambda: deque(maxlen=12)
    )
    # Top-of-screen banner; set on TUSHARE_UNAUTH or CANCELLED outcome.
    banner: str | None = None
    banner_style: str = "status.error"


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _format_header(state: DashboardState) -> Text:
    started = state.started_at.strftime("%H:%M:%S") if state.started_at else "—"
    short = state.run_id[:8] if state.run_id else "—"
    label = "辩论模式" if state.debate else "单 LLM 模式"
    t = Text()
    t.append("🚀 DeepTrade · 打板策略", style="title")
    t.append(f"  v{state.plugin_version}  ", style="dim")
    t.append(f"({label})", style="subtitle")
    t.append(f"   run_id={short}  开始 {started}", style="dim")
    return t


def _format_config(cfg: ConfigSummary) -> Text:
    parts: list[tuple[str, str]] = []
    if cfg.trade_date:
        nxt = cfg.next_trade_date or "—"
        parts.append(("📅", f"交易日期 {cfg.trade_date} (T) → {nxt} (T+1)"))
    if cfg.min_float_mv_yi is not None and cfg.max_float_mv_yi is not None:
        parts.append(
            ("⚙️", f"{cfg.min_float_mv_yi}亿 < 流通市值 < {cfg.max_float_mv_yi}亿")
        )
    if cfg.max_close_yuan is not None:
        parts.append(("💰", f"股价 < {cfg.max_close_yuan}元"))
    if cfg.lgb_enabled is not None:
        parts.append(("🧠", f"LGB: {'已开启' if cfg.lgb_enabled else '未启用'}"))
    if cfg.tushare_fallback_count:
        parts.append(
            ("⚠", f"Tushare 缓存兜底 ×{cfg.tushare_fallback_count}")
        )

    if not parts:
        return Text("(等待配置加载…)", style="dim")

    line = Text()
    for i, (icon, body) in enumerate(parts):
        if i:
            line.append(" | ", style="dim")
        line.append(f"{icon} ", style="k.label")
        line.append(body)
    return line


def _format_progress_bar(done: int, total: int, *, width: int = 30) -> str:
    """ASCII progress bar; avoids ``rich.progress.Progress`` because we
    redraw the whole frame on every event."""
    if total <= 0:
        return ""
    pct = max(0.0, min(1.0, done / total))
    filled = int(round(pct * width))
    return "[" + "━" * filled + " " * (width - filled) + f"] {int(pct * 100)}%"


def _stage_lines(stage: Stage) -> list[Text]:
    """Render one stage as 1..3 lines of rich Text (status + optional detail + optional progress)."""
    glyph = _STATUS_GLYPHS[stage.status]
    style = _STATUS_STYLES[stage.status]
    label = _STATUS_LABELS[stage.status]
    out: list[Text] = []
    head = Text()
    head.append(f"{glyph} ", style=style)
    head.append(f"阶段 {stage.stage_id}: ", style="k.label")
    head.append(stage.title)
    head.append("  ", style="dim")
    head.append(f"[ {label} ]", style=style)
    out.append(head)

    # Detail lingers on the final state only for stages that carry a
    # meaningful per-completion summary — currently just Step 4.5's
    # ``✔ 全局重排完成 (in=X / out=Y)`` from LLM_FINAL_RANK. For Step 2/4 the
    # ``[ 完成 ]`` label is sufficient and any leftover LIVE_STATUS is noise.
    show_detail = stage.detail and (
        stage.status in (StageStatus.RUNNING, StageStatus.FAILED)
        or stage.stage_id == "4.5"
    )
    if show_detail:
        out.append(Text(f"   └─ {stage.detail}", style="dim"))

    if stage.progress_total and stage.progress_total > 0 and stage.status in (
        StageStatus.RUNNING,
        StageStatus.SUCCESS,
        StageStatus.PARTIAL,
    ):
        bar = _format_progress_bar(stage.progress_done, stage.progress_total)
        out.append(
            Text(
                f"   └─ {bar}  ({stage.progress_done}/{stage.progress_total})",
                style="progress.percentage",
            )
        )

    for fb in stage.failed_batches[-3:]:
        out.append(Text(f"   ✘ {fb}", style="status.error"))
    return out


def _format_stages(stack: StageStack) -> Group:
    if not stack.stages:
        return Group(Text("(等待 pipeline 启动…)", style="dim"))
    renderables: list[RenderableType] = []
    for st in stack.stages:
        renderables.extend(_stage_lines(st))
    return Group(*renderables)


def _format_log(
    lines: deque[tuple[str, str, str]], *, max_rows: int
) -> Group:
    if not lines:
        return Group(Text("(暂无日志)", style="dim"))
    rows = list(lines)[-max_rows:]
    out: list[RenderableType] = []
    for ts, level, msg in rows:
        style = {
            "INFO": "dim",
            "WARN": "status.running",
            "ERROR": "status.error",
        }.get(level, "dim")
        t = Text()
        t.append(f"{ts} ", style="log.time")
        t.append(f"[{level}] ", style=style)
        t.append(msg)
        out.append(t)
    return Group(*out)


# ---------------------------------------------------------------------------
# Full layout
# ---------------------------------------------------------------------------


def _format_debate_panel(grid: DebateGrid) -> Group:
    parts: list[RenderableType] = []
    if grid.banner:
        parts.append(Text(grid.banner, style="status.running"))
    parts.append(render_grid_table(grid))
    return Group(*parts)


def render_dashboard(state: DashboardState, *, width: int) -> RenderableType:
    """Compose the full dashboard frame.

    ``width`` is the current terminal width; we use it to decide between
    *full* (≥ 80 cols, panels with borders) and *compact* (< 80 cols, no
    borders, fewer log rows) layouts (Plan §4.4).
    """
    compact = width < 80
    # v0.6.5 — when an error banner is up (FAILED / CANCELLED / TUSHARE_UNAUTH),
    # widen the log window so traceback / context fits. Normal runs keep the
    # tighter 5-row window so the panel doesn't crowd progress.
    error_mode = state.banner is not None and "error" in state.banner_style
    if error_mode:
        log_rows = 12 if width >= 100 else (8 if width >= 80 else 0)
    else:
        log_rows = 5 if width >= 100 else (3 if width >= 80 else 0)

    header = _format_header(state)
    config_panel = _format_config(state.config)
    stages_panel = _format_stages(state.stages)
    debate_panel = (
        _format_debate_panel(state.debate_grid) if state.debate_grid else None
    )
    log_panel = _format_log(state.log_lines, max_rows=log_rows) if log_rows else None

    if compact:
        # No borders, single column. The Group concatenates renderables.
        items: list[RenderableType] = [header, Text(""), config_panel, Text(""), stages_panel]
        if debate_panel is not None:
            items.extend([Text(""), debate_panel])
        if log_panel is not None:
            items.extend([Text(""), log_panel])
        if state.banner:
            items.insert(0, Text(state.banner, style=state.banner_style))
            items.insert(1, Text(""))
        return Group(*items)

    panels: list[RenderableType] = []
    if state.banner:
        panels.append(
            Panel(
                Text(state.banner, style=state.banner_style),
                border_style="panel.border.error",
                padding=(0, 1),
            )
        )
    panels.append(
        Panel(
            header,
            border_style="panel.border.primary",
            padding=(0, 1),
        )
    )
    panels.append(
        Panel(
            config_panel,
            title="运行配置",
            border_style="panel.border.primary",
            padding=(0, 1),
        )
    )
    panels.append(
        Panel(
            stages_panel,
            title="执行进度",
            border_style="panel.border.primary",
            padding=(0, 1),
        )
    )
    if debate_panel is not None:
        panels.append(
            Panel(
                debate_panel,
                title="辩论汇总",
                border_style="panel.border.primary",
                padding=(0, 1),
            )
        )
    if log_panel is not None:
        panels.append(
            Panel(
                log_panel,
                title=f"日志 (最近 {log_rows} 条)",
                border_style="panel.border.primary",
                padding=(0, 1),
            )
        )
    return Group(*panels)


__all__ = [
    "ConfigSummary",
    "DashboardState",
    "render_dashboard",
]
