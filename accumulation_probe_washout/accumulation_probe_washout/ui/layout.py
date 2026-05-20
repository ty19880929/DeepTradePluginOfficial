"""Rich layout primitives for the APW dashboard.

Pure rendering — takes immutable state, returns rich Renderables. No side
effects, no event handling.
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


def render_header(mode: str, trade_date: str, run_id: str) -> Panel:
    text = Text.from_markup(
        f"[bold cyan]accumulation-probe-washout[/]  mode=[bold]{mode}[/]  "
        f"T=[bold]{trade_date}[/]  run_id=[dim]{run_id[:8]}[/]"
    )
    return Panel(text, border_style="cyan", padding=(0, 1))


def _format_step_no(no: float) -> str:
    """Drop the trailing .0 on whole-number step ids ("Step 2", not "Step 2.0")."""
    if isinstance(no, float) and no.is_integer():
        return str(int(no))
    return str(no)


def render_stage_stack(stack: StageStack) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("glyph", no_wrap=True)
    tbl.add_column("step", no_wrap=True)
    tbl.add_column("msg")
    for s in stack.steps:
        tbl.add_row(
            Text(_STATE_GLYPH[s.state], style=_STATE_STYLE[s.state]),
            Text(f"Step {_format_step_no(s.no)} {s.label}", style=_STATE_STYLE[s.state]),
            Text(s.message or "", style="dim"),
        )
    return Panel(tbl, title="Stages", border_style="blue", padding=(0, 1))


def render_funnel(payload: dict[str, Any]) -> Panel:
    """Screen-mode funnel: horizontal bars from 主板池 → launch_ready."""
    rows = [
        ("主板池", payload.get("n_main_board", 0)),
        ("ST/停牌后", payload.get("n_after_st_susp", 0)),
        ("流动性后", payload.get("n_after_liquidity", 0)),
        ("吸筹后", payload.get("n_after_accumulation", 0)),
        ("试盘后", payload.get("n_after_probe", 0)),
        ("洗盘后", payload.get("n_after_washout", 0)),
        ("launch_ready", payload.get("n_after_launch_ready", 0)),
    ]
    max_val = max([v for _, v in rows] + [1])
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("label", no_wrap=True)
    tbl.add_column("bar")
    tbl.add_column("count", justify="right")
    for label, val in rows:
        bar_len = int((val / max_val) * 40) if max_val else 0
        bar = "█" * bar_len
        tbl.add_row(
            Text(label, style="cyan"),
            Text(bar, style="green"),
            Text(str(val), style="bold"),
        )
    return Panel(tbl, title="筛选漏斗", border_style="green", padding=(0, 1))


def render_log(lines: list[str], *, max_lines: int = 12) -> Panel:
    tail = lines[-max_lines:]
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("msg")
    for line in tail:
        tbl.add_row(Text(line, style="dim"))
    return Panel(tbl, title="日志", border_style="grey50", padding=(0, 1))
