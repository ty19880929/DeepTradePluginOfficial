"""Plugin-managed CLI for market-review (PR-1 skeleton).

PR-1 only ships the top-level typer app so ``deeptrade market-review --help``
works after install. All subcommands are stubbed: invoking them prints a
"not yet implemented" notice + exit code 2, matching the v0.1.0 §18 PR-1
charter ("cli skeleton（仅 --help）"). Real subcommand bodies land in PR-2..6:

- ``run`` / ``sync``  — PR-6 (wraps runner + data layer from PR-2..5)
- ``history``         — PR-6
- ``report``          — PR-6
- ``settings``        — PR-6

Invoked via the framework's pure pass-through dispatch:
    deeptrade market-review <subcommand> [...]
"""

from __future__ import annotations

import click
import typer
from rich.console import Console

app = typer.Typer(
    name="market-review",
    help=(
        "市场复盘 — A 股单日 / 区间复盘：板块轮动 / 情绪 / 资金 / 龙头 / "
        "风格 / 风险 / 展望 七章节 LLM 综合报告。"
    ),
    no_args_is_help=True,
    add_completion=False,
)

settings_app = typer.Typer(
    name="settings",
    help="本插件可持久化的运行参数（窗口上限、情绪权重、板块口径、LLM provider 等）。",
    no_args_is_help=False,
    add_completion=False,
    invoke_without_command=True,
)
app.add_typer(settings_app, name="settings")


_STUB_EXIT_CODE = 2


def _stub(subcommand: str) -> None:
    """Common body for PR-1 subcommand stubs."""
    console = Console()
    console.print(
        f"[yellow]market-review {subcommand}[/] 尚未实现（v0.1.0 PR-1 仅交付骨架）。"
        "\n后续 PR-2..6 将逐步落地数据层 / 指标 / LLM / 报告。"
    )
    raise typer.Exit(code=_STUB_EXIT_CODE)


@app.command("run", help="单日或区间复盘（PR-6 实现）。")
def run_cmd(
    trade_date: str | None = typer.Option(
        None, "--trade-date", help="YYYYMMDD，单日复盘。与 --start/--end 互斥。"
    ),
    start: str | None = typer.Option(None, "--start", help="YYYYMMDD，区间起点。"),
    end: str | None = typer.Option(None, "--end", help="YYYYMMDD，区间末日。"),
) -> None:
    _ = (trade_date, start, end)  # accepted for --help discoverability
    _stub("run")


@app.command("sync", help="仅落 Tushare 数据，不调 LLM（PR-2/6 实现）。")
def sync_cmd(
    trade_date: str | None = typer.Option(None, "--trade-date"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
) -> None:
    _ = (trade_date, start, end)
    _stub("sync")


@app.command("history", help="列出最近的复盘 run（PR-6 实现）。")
def history_cmd(
    limit: int = typer.Option(20, "--limit"),
) -> None:
    _ = limit
    _stub("history")


@app.command("report", help="重渲历史 run 的终端摘要 / 完整 markdown（PR-6 实现）。")
def report_cmd(
    run_id: str = typer.Argument(..., help="run_id（UUID）"),
    full: bool = typer.Option(False, "--full"),
    section: str | None = typer.Option(None, "--section"),
) -> None:
    _ = (run_id, full, section)
    _stub("report")


@settings_app.callback()
def settings_callback(ctx: typer.Context) -> None:
    """PR-6 will add ``show`` / ``set`` / ``reset`` here."""
    if ctx.invoked_subcommand is None:
        _stub("settings")


def main(argv: list[str]) -> int:
    """Framework dispatch entrypoint.

    The framework hands ``argv`` straight through from
    ``deeptrade market-review <argv...>``; we forward to typer's app so that
    ``--help`` / unknown command messages render through typer's normal path.

    Notes on the exit-code dance (click 8.3+):

    - With ``standalone_mode=False`` click no longer raises ``Exit`` —
      it returns the int ``exit_code`` as the ``app(...)`` return value.
      We forward that integer back to the framework.
    - ``typer.Exit`` / ``SystemExit`` are still caught defensively in case
      a user-installed click variant restores the legacy raise-on-Exit
      semantics, or a stray ``sys.exit`` slips into a stub.
    """
    try:
        rv = app(list(argv), standalone_mode=False)
    except typer.Exit as exc:  # noqa: PERF203 — defensive vs. legacy click
        return int(exc.exit_code or 0)
    except click.exceptions.ClickException as exc:
        # click 8+ keeps ``UsageError`` / ``NoArgsIsHelpError`` raising in
        # non-standalone mode. Reproduce standalone behavior: show the
        # formatted message (NoArgsIsHelpError already carries the help text)
        # and forward the documented exit code.
        exc.show()
        return int(exc.exit_code or 1)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1
    if isinstance(rv, int):
        return rv
    return 0
