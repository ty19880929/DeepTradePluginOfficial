"""Rich Live 双面板渲染器（设计 §11.1 / §11.2）.

布局：
    Header        — code ｜ 市场时区当前时间 ｜ trade_date ｜ run_id ｜ 状态
    Live 指标     — last / VWAP / σ / z / band / 累计量 / 采样数（standby 时为倒计时）
    执行记录(左)  — 采样/信号/风控/异常 事件滚动（vwr_events 实时投影）
    交易记录(右)  — 模拟成交滚动（payload.kind == "trade"，P2 起有内容）
    Footer        — 当日累计统计（P2 起：成交笔数/盈亏；P1 显示采样统计）

渲染失败绝不外抛 —— daemon 侧还有一层 catch + 降级 legacy（设计 §14）。
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .protocol import RunMeta

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.plugins_api.events import StrategyEvent

    from ..clock import MarketClock

_LEVEL_STYLE = {"info": "white", "warn": "yellow", "error": "bold red"}
_MAX_ROWS = 12


class RichDashboardRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._live: Live | None = None
        self._meta: RunMeta | None = None
        self._clock: MarketClock | None = None
        self._status: str = "init"
        self._metrics: dict[str, Any] = {}
        self._exec_rows: deque[tuple[str, str, str, str]] = deque(maxlen=_MAX_ROWS)
        self._trade_rows: deque[tuple[str, str]] = deque(maxlen=_MAX_ROWS)
        self._n_samples = 0
        self._n_bars = 0

    # ---- EventRenderer ----------------------------------------------------

    def start(self, meta: RunMeta, clock: MarketClock) -> None:
        self._meta = meta
        self._clock = clock
        self._live = Live(
            self._render(), console=self._console, refresh_per_second=4, transient=False
        )
        self._live.start()

    def handle(self, event: StrategyEvent) -> None:
        payload = event.payload or {}
        kind = payload.get("kind")
        if payload.get("status"):
            self._status = str(payload["status"])
        if kind == "sample":
            self._n_samples = int(payload.get("n_samples", self._n_samples + 1))
            self._n_bars = int(payload.get("n_bars", self._n_bars))
            for key in ("last", "vwap", "sigma", "z", "band_upper", "band_lower",
                        "cum_vol", "cum_amount", "warmup"):
                if key in payload:
                    self._metrics[key] = payload[key]
        # 交易侧实时账务（P2：sample/trade/eod 事件都带 live_payload）
        for key in ("position", "cash", "net_pnl", "n_fills", "circuit_broken"):
            if key in payload:
                self._metrics[key] = payload[key]
        if kind == "standby":
            self._metrics["countdown_s"] = payload.get("countdown_s")
        ts = self._fmt_now()
        if kind == "trade":
            self._trade_rows.appendleft((ts, event.message))
        else:
            self._exec_rows.appendleft(
                (ts, event.type.value, str(event.level.value), event.message)
            )
        if self._live is not None:
            self._live.update(self._render())

    def finish(self) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    # ---- 渲染 --------------------------------------------------------------

    def _fmt_now(self) -> str:
        return self._clock.now().strftime("%H:%M:%S") if self._clock else "--:--:--"

    def _render(self) -> Group:
        meta = self._meta
        header = Text.assemble(
            ("vwap-reversion ", "bold cyan"),
            (f"{meta.mode}  " if meta else "", "magenta"),
            (f"{meta.code}  " if meta else "", "bold"),
            (f"{meta.trade_date}  " if meta else "", ""),
            (f"{self._fmt_now()}  ", "green"),
            (f"[{self._status.upper()}]", _status_style(self._status)),
        )

        if self._status == "standby":
            cd = self._metrics.get("countdown_s")
            body = (
                f"距开盘还有 {_fmt_countdown(cd)}（到点自动开始采集）"
                if cd is not None
                else "待机中…"
            )
            metrics_panel = Panel(body, title="STANDBY", border_style="yellow")
        else:
            metrics_panel = Panel(self._metrics_table(), title="Live 指标", border_style="cyan")

        grid = Table.grid(expand=True)
        grid.add_column(ratio=3)
        grid.add_column(ratio=2)
        grid.add_row(
            Panel(self._exec_table(), title="策略执行记录", border_style="blue"),
            Panel(self._trades_table(), title="交易记录", border_style="magenta"),
        )

        footer = Text(
            f"采样 {self._n_samples} 次 ｜ 有效 bar {self._n_bars} 根 ｜ "
            f"params: {meta.params_summary if meta else ''}",
            style="dim",
        )
        return Group(header, metrics_panel, grid, footer)

    def _metrics_table(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="cyan", justify="right")
        t.add_column()
        m = self._metrics

        def f(key: str, fmt: str = "{:.4f}") -> str:
            v = m.get(key)
            if v is None:
                return "—"
            try:
                return fmt.format(float(v))
            except (TypeError, ValueError):
                return str(v)

        warm = "（预热中，不交易）" if m.get("warmup") else ""
        t.add_row("last / VWAP", f"{f('last')} / {f('vwap')} {warm}")
        t.add_row("σ / z", f"{f('sigma', '{:.5f}')} / {f('z', '{:+.2f}')}")
        t.add_row("band", f"[{f('band_lower')} , {f('band_upper')}]")
        t.add_row("累计量/额", f"{f('cum_vol', '{:,.0f}')} 股 / {f('cum_amount', '{:,.0f}')} 元")
        circuit = "  ⛔熔断" if m.get("circuit_broken") else ""
        t.add_row(
            "持仓 / 现金",
            f"{f('position', '{:,.0f}')} 股 / {f('cash', '{:,.2f}')} 元{circuit}",
        )
        t.add_row("净盈亏", f"{f('net_pnl', '{:+,.2f}')} 元（成交 {f('n_fills', '{:.0f}')} 笔）")
        return t

    def _exec_table(self) -> Table:
        t = Table(show_header=False, expand=True, box=None, pad_edge=False)
        t.add_column(width=8, style="green")
        t.add_column(overflow="ellipsis", no_wrap=True)
        if not self._exec_rows:
            t.add_row("", Text("(等待事件…)", style="dim"))
        for ts, _etype, level, msg in self._exec_rows:
            t.add_row(ts, Text(msg, style=_LEVEL_STYLE.get(level, "white")))
        return t

    def _trades_table(self) -> Table:
        t = Table(show_header=False, expand=True, box=None, pad_edge=False)
        t.add_column(width=8, style="green")
        t.add_column(overflow="ellipsis", no_wrap=True)
        if not self._trade_rows:
            t.add_row("", Text("(暂无成交)", style="dim"))
        for ts, msg in self._trade_rows:
            t.add_row(ts, msg)
        return t


def _status_style(status: str) -> str:
    return {
        "standby": "bold yellow",
        "running": "bold green",
        "done": "bold blue",
        "aborted": "bold red",
    }.get(status, "white")


def _fmt_countdown(seconds: Any) -> str:
    try:
        s = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
