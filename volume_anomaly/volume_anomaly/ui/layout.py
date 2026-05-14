"""Rendering helpers that turn the dashboard state into ``rich`` objects.

The :class:`RichDashboardRenderer` owns the live region; this module owns
*what* gets drawn inside it. Keeping the two split makes it possible to
test the rendered output piece-wise without needing a ``Live`` context.

Layout goal (Plan §4.1 — analyze, §4.2 — screen):

    ┌─ DeepTrade · 成交量异动 ───┐
    │ run_id / start / mode      │
    ├─ 运行配置 ─────────────────┤
    │ 交易日期 / 筛选 / LGB      │
    ├─ 执行进度 ─────────────────┤
    │ ✔ 阶段 0..N + detail/prog  │
    ├─ 筛选漏斗（仅 screen） ────┤
    │ 主板  ████████  3210       │
    ├─ 日志 ─────────────────────┤
    │ recent INFO/WARN/ERROR     │
    └────────────────────────────┘

When the terminal is narrower than 80 cols we render a *compact* form
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

from .funnel import FunnelSummary, render_funnel_compact, render_funnel_full
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

    Populated incrementally from the LOG event emitted by
    :func:`volume_anomaly.runner._va_settings_log_event` (Plan §3.3 mapping)
    and from ``Step 0: STEP_FINISHED.payload`` for the trade date. Field
    names mirror :class:`volume_anomaly.data.ScreenRules` /
    ``AppConfig.app_profile`` exactly so the runner can splat them in via
    ``setattr``. Any field left ``None`` is hidden — the dashboard renders
    only what it knows.
    """

    trade_date: str | None = None
    next_trade_date: str | None = None
    profile: str | None = None
    main_board_only: bool | None = None
    pct_chg_min: float | None = None
    pct_chg_max: float | None = None
    turnover_min: float | None = None
    turnover_max: float | None = None
    vol_ratio_5d_min: float | None = None
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
    mode: str = "analyze"
    config: ConfigSummary = field(default_factory=ConfigSummary)
    stages: StageStack = field(default_factory=StageStack)
    # Populated only in screen mode (Plan §3.4 / §4.2).
    funnel: FunnelSummary | None = None
    # Log ring buffer — keeps the most recent N lines for the bottom panel.
    log_lines: deque[tuple[str, str, str]] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    # Top-of-screen banner; set on TUSHARE_UNAUTH or CANCELLED outcome.
    banner: str | None = None
    banner_style: str = "status.error"


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


_MODE_LABELS: dict[str, str] = {
    "analyze": "走势分析",
    "screen": "异动筛选",
    "prune": "watchlist 清理",
    "evaluate": "回测评估",
}


def _format_header(state: DashboardState) -> Text:
    started = (
        state.started_at.strftime("%H:%M:%S") if state.started_at else "—"
    )
    short = state.run_id[:8] if state.run_id else "—"
    mode_label = _MODE_LABELS.get(state.mode, state.mode)
    t = Text()
    t.append("🚀 DeepTrade · 成交量异动", style="title")
    t.append(f"  v{state.plugin_version}  ", style="dim")
    t.append(f"({mode_label})", style="subtitle")
    t.append(f"   run_id={short}  开始 {started}", style="dim")
    return t


def _format_config(cfg: ConfigSummary, mode: str) -> Text:
    parts: list[tuple[str, str]] = []
    if cfg.trade_date:
        if mode == "analyze" and cfg.next_trade_date:
            parts.append(
                (
                    "📅",
                    f"交易日期 {cfg.trade_date} (T) → {cfg.next_trade_date} (T+1)",
                )
            )
        else:
            parts.append(("📅", f"交易日期 {cfg.trade_date}"))
    rule_parts: list[str] = []
    if cfg.main_board_only:
        rule_parts.append("主板")
    if cfg.pct_chg_min is not None and cfg.pct_chg_max is not None:
        rule_parts.append(f"涨幅 {cfg.pct_chg_min}~{cfg.pct_chg_max}%")
    elif cfg.pct_chg_min is not None:
        rule_parts.append(f"涨幅 ≥ {cfg.pct_chg_min}%")
    if cfg.turnover_min is not None and cfg.turnover_max is not None:
        rule_parts.append(f"换手 {cfg.turnover_min}~{cfg.turnover_max}%")
    elif cfg.turnover_min is not None:
        rule_parts.append(f"换手 ≥ {cfg.turnover_min}%")
    if cfg.vol_ratio_5d_min is not None:
        rule_parts.append(f"量比 ≥ {cfg.vol_ratio_5d_min}")
    if rule_parts:
        parts.append(("⚙️", " | ".join(rule_parts)))
    if cfg.profile:
        parts.append(("🎚", f"profile: {cfg.profile}"))
    if cfg.lgb_enabled is not None and mode == "analyze":
        parts.append(("🧠", f"LGB: {'已开启' if cfg.lgb_enabled else '未启用'}"))
    if cfg.tushare_fallback_count:
        parts.append(("⚠", f"Tushare 缓存兜底 ×{cfg.tushare_fallback_count}"))

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
    return (
        "[" + "━" * filled + " " * (width - filled) + f"] {int(pct * 100)}%"
    )


def _stage_lines(stage: Stage) -> list[Text]:
    """Render one stage as 1..3 lines of rich Text (status + optional detail
    + optional progress)."""
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

    # Detail lingers on RUNNING / FAILED only. Once a stage hits SUCCESS the
    # ``[ 完成 ]`` label is sufficient and any leftover LIVE_STATUS becomes
    # noise.
    if stage.detail and stage.status in (
        StageStatus.RUNNING,
        StageStatus.FAILED,
    ):
        out.append(Text(f"   └─ {stage.detail}", style="dim"))

    if (
        stage.progress_total
        and stage.progress_total > 0
        and stage.status
        in (StageStatus.RUNNING, StageStatus.SUCCESS, StageStatus.PARTIAL)
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


def _format_funnel_panel(
    funnel: FunnelSummary, *, compact: bool, bar_width: int
):
    """Build the funnel renderable; ``None`` callers skip the panel entirely.

    Defined here (rather than directly importing the funnel renderer in
    :func:`render_dashboard`) so the layout module owns *all* shaping
    decisions including bar widths and compact/full switching.
    """
    if compact:
        return render_funnel_compact(funnel)
    return render_funnel_full(funnel, bar_width=bar_width)


# ---------------------------------------------------------------------------
# Full layout
# ---------------------------------------------------------------------------


def render_dashboard(state: DashboardState, *, width: int):
    """Compose the full dashboard frame.

    ``width`` is the current terminal width; we use it to decide between
    *full* (≥ 80 cols, panels with borders) and *compact* (< 80 cols, no
    borders, fewer log rows) layouts (Plan §4.4).
    """
    compact = width < 80
    log_rows = 5 if width >= 100 else (3 if width >= 80 else 0)
    funnel_bar_width = 32 if width >= 100 else 24

    header = _format_header(state)
    config_panel = _format_config(state.config, state.mode)
    stages_panel = _format_stages(state.stages)
    funnel_panel = (
        _format_funnel_panel(
            state.funnel, compact=compact, bar_width=funnel_bar_width
        )
        if state.funnel is not None
        else None
    )
    log_panel = (
        _format_log(state.log_lines, max_rows=log_rows)
        if log_rows
        else None
    )

    if compact:
        items: list[RenderableType] = [
            header,
            Text(""),
            config_panel,
            Text(""),
            stages_panel,
        ]
        if funnel_panel is not None:
            items.extend([Text(""), funnel_panel])
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
    if funnel_panel is not None:
        panels.append(
            Panel(
                funnel_panel,
                title="筛选漏斗",
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
