"""Rich layout primitives for the checkmate dashboard.

Pure rendering — takes immutable state, returns rich Renderables. No side
effects, no event handling. Bookkeeping lives in :class:`RichDashboardRenderer`.
"""

from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .stage_model import StageStack


_STATE_GLYPH = {
    "pending": "○",
    "running": "▶",
    "done": "✓",
    "failed": "✘",
}
_STATE_STYLE = {
    "pending": "dim",
    "running": "yellow",
    "done": "green",
    "failed": "red",
}


def render_header(*, mode: str, run_id: str, elapsed: str) -> Panel:
    text = Text.from_markup(
        f"[bold cyan]checkmate[/]  "
        f"mode=[bold]{mode}[/]  "
        f"run_id=[dim]{run_id[:8]}[/]  "
        f"elapsed=[bold]{elapsed}[/]"
    )
    return Panel(text, border_style="cyan", padding=(0, 1))


def render_stage_stack(stack: StageStack) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("glyph", no_wrap=True)
    tbl.add_column("step", no_wrap=True)
    tbl.add_column("msg")
    for s in stack.steps:
        tbl.add_row(
            Text(_STATE_GLYPH[s.state], style=_STATE_STYLE[s.state]),
            Text(f"Step {s.no} {s.label}", style=_STATE_STYLE[s.state]),
            Text(s.message or "", style="dim"),
        )
    return Panel(tbl, title="Stages", border_style="blue", padding=(0, 1))


def render_config_card(cfg_lines: list[tuple[str, str]]) -> Panel:
    """Render a key/value config card. Caller picks 3-9 rows to surface."""
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("key", no_wrap=True)
    tbl.add_column("value", no_wrap=True)
    for k, v in cfg_lines:
        tbl.add_row(Text(k, style="cyan"), Text(v, style="bold"))
    return Panel(tbl, title="Config", border_style="grey50", padding=(0, 1))


def render_regime_card(payload: dict[str, Any]) -> Panel:
    """Regime classifier readout (signals / backtest modes)."""
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("key", no_wrap=True)
    tbl.add_column("value", no_wrap=True)
    tbl.add_row(Text("regime", style="cyan"),
                Text(str(payload.get("regime") or "—"), style="bold yellow"))
    tbl.add_row(Text("exposure_cap", style="cyan"),
                Text(str(payload.get("exposure_cap") or "—"), style="bold"))
    breadth = payload.get("breadth_ma120")
    tbl.add_row(Text("breadth_ma120", style="cyan"),
                Text(f"{breadth*100:.1f}%" if breadth is not None else "—",
                     style="bold"))
    return Panel(tbl, title="Regime", border_style="magenta", padding=(0, 1))


def render_funnel_card(payload: dict[str, Any]) -> Panel:
    """Scan-mode funnel: universe → eligible → top_scored — 5 horizontal bars."""
    rows = [
        ("候选总池", payload.get("total", 0)),
        ("eligible", payload.get("eligible", 0)),
        ("已排除", payload.get("excluded", 0)),
        ("特征已算", payload.get("features_rows", 0)),
        ("top scored", payload.get("top_scored", 0)),
    ]
    max_val = max([v for _, v in rows] + [1])
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("label", no_wrap=True)
    tbl.add_column("bar")
    tbl.add_column("count", justify="right", no_wrap=True)
    for label, val in rows:
        bar_len = int((val / max_val) * 30) if max_val else 0
        bar = "█" * bar_len
        tbl.add_row(
            Text(label, style="cyan"),
            Text(bar, style="green"),
            Text(str(val), style="bold"),
        )
    return Panel(tbl, title="筛选漏斗", border_style="green", padding=(0, 1))


def render_regime_breakdown_card(by_regime: dict[str, dict[str, Any]]) -> Panel:
    """PR-6.3: 4 × N grid where N = number of distinct regimes seen.

    Columns: regime / n_trades / total_pnl / win_rate / avg_hold_days
    Rows : one per regime, sorted by regime name.
    """
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("regime", no_wrap=True)
    tbl.add_column("n_trades", justify="right")
    tbl.add_column("total_pnl", justify="right")
    tbl.add_column("win_rate", justify="right")
    tbl.add_column("avg_hold", justify="right")
    tbl.add_row(
        Text("regime", style="bold dim"),
        Text("n_trades", style="bold dim"),
        Text("total_pnl", style="bold dim"),
        Text("win_rate", style="bold dim"),
        Text("avg_hold", style="bold dim"),
    )
    for regime, stats in sorted(by_regime.items()):
        pnl = stats.get("total_pnl", 0.0)
        pnl_style = "bold green" if pnl >= 0 else "bold red"
        wr = stats.get("win_rate")
        wr_text = f"{wr*100:.1f}%" if wr is not None else "—"
        ahd = stats.get("avg_hold_days")
        ahd_text = f"{ahd:.1f}" if ahd is not None else "—"
        tbl.add_row(
            Text(regime, style="bold yellow"),
            Text(str(stats.get("n_trades", 0)), style="bold"),
            Text(f"{pnl:,.2f}", style=pnl_style),
            Text(wr_text, style="bold"),
            Text(ahd_text, style="bold"),
        )
    return Panel(tbl, title="Regime breakdown", border_style="magenta", padding=(0, 1))


def render_positions_card(payload: dict[str, Any]) -> Panel:
    """Backtest-mode portfolio card: equity + last session stats.

    The full per-position breakdown / industry bars are deferred to Iter-6
    (v0.3.0) where dashboards get richer (see iteration_tasks.md PR-6.3).
    """
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("key", no_wrap=True)
    tbl.add_column("value", no_wrap=True)
    tbl.add_row(Text("session", style="cyan"),
                Text(str(payload.get("trade_date") or "—"), style="bold"))
    equity = payload.get("equity")
    tbl.add_row(Text("equity", style="cyan"),
                Text(f"{equity:,.2f}" if equity is not None else "—",
                     style="bold green"))
    dd = payload.get("drawdown_pct")
    tbl.add_row(Text("drawdown", style="cyan"),
                Text(f"{dd*100:.2f}%" if dd is not None else "—",
                     style="bold red"))
    tbl.add_row(Text("fills (today)", style="cyan"),
                Text(str(payload.get("n_fills", 0)), style="bold"))
    tbl.add_row(Text("regime", style="cyan"),
                Text(str(payload.get("regime") or "—"), style="bold yellow"))
    return Panel(tbl, title="Portfolio", border_style="green", padding=(0, 1))


def render_log_panel(lines: list[str], *, max_lines: int = 10) -> Panel:
    tail = lines[-max_lines:]
    tbl = Table.grid(padding=(0, 0))
    tbl.add_column("line")
    for line in tail:
        tbl.add_row(Text(line, style="dim"))
    return Panel(tbl, title="Log", border_style="grey50", padding=(0, 1))


__all__ = [
    "render_header",
    "render_stage_stack",
    "render_config_card",
    "render_regime_card",
    "render_regime_breakdown_card",
    "render_funnel_card",
    "render_positions_card",
    "render_log_panel",
]
