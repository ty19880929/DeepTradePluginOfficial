"""Plugin-managed CLI for limit-up-board.

Subcommands:
    run     — full pipeline (Step 0..5)
    sync    — data-only path (no LLM)
    history — list recent runs
    report  — re-render a finished run's terminal summary

Invoked via the framework's pure pass-through dispatch:
    deeptrade limit-up-board <subcommand> [...]
"""

from __future__ import annotations

import sys

import questionary
import typer
from rich.console import Console
from rich.table import Table

from deeptrade.core import paths
from deeptrade.core.config import ConfigService
from deeptrade.core.db import Database
from deeptrade.core.llm_manager import LLMManager

from .config import LubConfig, list_for_show, load_config, save_config
from .runner import LubRunner, PreconditionError, RunParams, render_finished_run
from .runtime import LubRuntime

app = typer.Typer(
    name="limit-up-board",
    help="打板策略 — A 股涨停板双轮 LLM 漏斗。",
    no_args_is_help=True,
    add_completion=False,
)

settings_app = typer.Typer(
    name="settings",
    help="本插件可持久化的运行参数（流通市值 / 当前股价 上限）。",
    no_args_is_help=False,
    add_completion=False,
    invoke_without_command=True,
)
app.add_typer(settings_app, name="settings")


def _open_runtime() -> tuple[Database, LubRuntime]:
    db = Database(paths.db_path())
    cfg = ConfigService(db)
    rt = LubRuntime(db=db, config=cfg, llms=LLMManager(db, cfg))
    return db, rt


@app.command("run")
def cmd_run(
    trade_date: str | None = typer.Option(None, "--trade-date", help="YYYYMMDD"),
    allow_intraday: bool = typer.Option(False, "--allow-intraday"),
    force_sync: bool = typer.Option(False, "--force-sync"),
    daily_lookback: int = typer.Option(30, "--daily-lookback"),
    moneyflow_lookback: int = typer.Option(5, "--moneyflow-lookback"),
    debate: bool = typer.Option(
        False,
        "--debate",
        help="启用多 LLM 辩论模式（需要 ≥2 个已配置的 LLM provider）",
    ),
    debate_llms: str | None = typer.Option(
        None,
        "--debate-llms",
        help=(
            "逗号分隔的 LLM provider 子集（如 'deepseek,kimi'），"
            "必须配合 --debate 使用；不指定则使用全部已配置 provider"
        ),
    ),
) -> None:
    """Run the full打板策略 pipeline."""
    debate_llms_list: list[str] | None = None
    if debate_llms is not None:
        if not debate:
            typer.echo("✘ --debate-llms 必须配合 --debate 使用")
            raise typer.Exit(2)
        debate_llms_list = [s.strip() for s in debate_llms.split(",") if s.strip()]
        if not debate_llms_list:
            typer.echo("✘ --debate-llms 解析后为空")
            raise typer.Exit(2)

    db, rt = _open_runtime()
    try:
        params = RunParams(
            trade_date=trade_date,
            allow_intraday=allow_intraday,
            force_sync=force_sync,
            daily_lookback=daily_lookback,
            moneyflow_lookback=moneyflow_lookback,
            debate=debate,
            debate_llms=debate_llms_list,
        )
        runner = LubRunner(rt)
        outcome = runner.execute(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
        # Print the terminal summary right after a successful run.
        render_finished_run(outcome.run_id)
    finally:
        db.close()


@app.command("sync")
def cmd_sync(
    trade_date: str | None = typer.Option(None, "--trade-date", help="YYYYMMDD"),
    allow_intraday: bool = typer.Option(False, "--allow-intraday"),
    force_sync: bool = typer.Option(False, "--force-sync"),
    daily_lookback: int = typer.Option(30, "--daily-lookback"),
    moneyflow_lookback: int = typer.Option(5, "--moneyflow-lookback"),
) -> None:
    """Fetch + persist data only (no LLM stages)."""
    db, rt = _open_runtime()
    try:
        params = RunParams(
            trade_date=trade_date,
            allow_intraday=allow_intraday,
            force_sync=force_sync,
            daily_lookback=daily_lookback,
            moneyflow_lookback=moneyflow_lookback,
        )
        runner = LubRunner(rt)
        outcome = runner.execute_sync_only(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value != "success":
            raise typer.Exit(1)
    finally:
        db.close()


@app.command("history")
def cmd_history(limit: int = typer.Option(20, "--limit")) -> None:
    """List recent runs of this plugin."""
    db = Database(paths.db_path())
    try:
        rows = db.fetchall(
            "SELECT run_id, trade_date, status, started_at, finished_at FROM lub_runs "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
    finally:
        db.close()
    if not rows:
        typer.echo("(no runs)")
        return
    for r in rows:
        typer.echo(f"{r[0]}  {r[1]:<10}  {r[2]:<15}  {r[3]} → {r[4] or '-'}")


@app.command("report")
def cmd_report(
    run_id: str = typer.Argument(..., help="Run UUID to view"),
    full: bool = typer.Option(
        False, "--full", help="Print the full markdown summary instead of the concise view."
    ),
) -> None:
    """Re-display a finished run's report."""
    if full:
        from rich.console import Console
        from rich.markdown import Markdown

        from deeptrade.theme import EVA_THEME

        report_dir = paths.reports_dir() / run_id
        summary = report_dir / "summary.md"
        if not summary.is_file():
            typer.echo(f"✘ no report at {summary}")
            raise typer.Exit(2)
        Console(theme=EVA_THEME).print(Markdown(summary.read_text(encoding="utf-8")))
        typer.echo(f"\nReport directory: {summary.parent}")
        return
    render_finished_run(run_id)


# ---------------------------------------------------------------------------
# settings — plugin-local persisted run filters (v0.4)
# ---------------------------------------------------------------------------


def _prompt_positive_float(prompt: str, current: float) -> float:
    raw = questionary.text(f"{prompt} [{current}]:", default=str(current)).ask()
    if raw is None:
        raise typer.Exit(1)
    raw = raw.strip()
    if not raw:
        return current
    try:
        v = float(raw)
    except ValueError as e:
        typer.echo(f"✘ 无法解析为数字: {raw!r}")
        raise typer.Exit(2) from e
    if v <= 0:
        typer.echo(f"✘ 必须大于 0: {v}")
        raise typer.Exit(2)
    return v


@settings_app.callback()
def cmd_settings(ctx: typer.Context) -> None:
    """交互式编辑当前持久化设置（不带子命令时进入交互；`show` 子命令仅展示）。"""
    if ctx.invoked_subcommand is not None:
        return
    db = Database(paths.db_path())
    try:
        cur = load_config(db)
        new_mv = _prompt_positive_float("流通市值上限（亿）", cur.max_float_mv_yi)
        new_close = _prompt_positive_float("当前股价上限（元）", cur.max_close_yuan)
        new_cfg = LubConfig(max_float_mv_yi=new_mv, max_close_yuan=new_close)
        save_config(db, new_cfg)
        typer.echo(
            f"✔ Saved: 流通市值 < {new_cfg.max_float_mv_yi}亿、股价 < {new_cfg.max_close_yuan}元"
        )
    finally:
        db.close()


@settings_app.command("show")
def cmd_settings_show() -> None:
    """展示当前生效的设置（来源 = persisted / default）。"""
    db = Database(paths.db_path())
    try:
        rows = list_for_show(db)
    finally:
        db.close()
    console = Console()
    table = Table(title="limit-up-board settings")
    table.add_column("Key", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_column("Source", style="yellow")
    for key, value, source in rows:
        table.add_row(key, "" if value is None else str(value), source)
    console.print(table)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """Plugin's dispatch entry. Returns exit code."""
    try:
        app(argv, standalone_mode=False)
        return 0
    except typer.Exit as e:
        return int(e.exit_code or 0)
    except SystemExit as e:
        try:
            return int(e.code or 0)
        except (TypeError, ValueError):
            return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n✘ cancelled by user\n")
        return 130
    except PreconditionError as e:
        sys.stderr.write(f"✘ {e}\n")
        return 2
    except Exception as e:  # noqa: BLE001 — reflect to framework as exit 1
        sys.stderr.write(f"✘ {type(e).__name__}: {e}\n")
        return 1
