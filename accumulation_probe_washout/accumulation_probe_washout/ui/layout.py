"""Rich layout primitives for the APW dashboard.

Pure rendering — takes immutable state, returns rich Renderables. No side
effects, no event handling.
"""

from __future__ import annotations

from typing import Any

from rich import box
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
_PREDICTION_LABELS = {
    "launch_ready": "启动就绪",
    "watch_breakout": "观察突破",
    "still_washing": "仍在洗盘",
    "probe_failed": "试盘失败",
    "avoid": "回避",
}
_CONFIDENCE_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
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


def render_result_summary(rows: list[dict[str, Any]], *, total: int | None = None) -> Panel:
    tbl = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE,
        expand=True,
        padding=(0, 1),
    )
    tbl.add_column("排名", justify="right", no_wrap=True)
    tbl.add_column("代码", no_wrap=True)
    tbl.add_column("名称", no_wrap=True)
    tbl.add_column("当前价格", justify="right", no_wrap=True)
    tbl.add_column("启动分", justify="right", no_wrap=True)
    tbl.add_column("判断", no_wrap=True)
    tbl.add_column("置信度", no_wrap=True)
    tbl.add_column("LLM意见", ratio=2)

    for row in rows:
        tbl.add_row(
            _fmt_int(row.get("rank")),
            str(row.get("ts_code") or ""),
            str(row.get("name") or ""),
            _fmt_float(row.get("current_price"), digits=2),
            _fmt_float(row.get("launch_score"), digits=1),
            _label(_PREDICTION_LABELS, row.get("prediction")),
            _label(_CONFIDENCE_LABELS, row.get("confidence")),
            str(row.get("llm_opinion") or ""),
        )
    title = "结果摘要"
    if total is not None and total != len(rows):
        title = f"结果摘要（前 {len(rows)} / 共 {total}）"
    return Panel(tbl, title=title, border_style="magenta", padding=(0, 1))


def render_log(lines: list[str], *, max_lines: int = 12) -> Panel:
    tail = lines[-max_lines:]
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column("msg")
    for line in tail:
        tbl.add_row(Text(line, style="dim"))
    return Panel(tbl, title="日志", border_style="grey50", padding=(0, 1))


def _fmt_float(value: Any, *, digits: int) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "—"


def _label(labels: dict[str, str], value: Any) -> str:
    raw = str(value or "")
    return labels.get(raw, raw)
