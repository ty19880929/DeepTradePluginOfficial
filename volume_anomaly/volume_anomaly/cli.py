"""Plugin-managed CLI for volume-anomaly.

Subcommands:
    screen   — apply local rules → upsert va_watchlist (no LLM)
    analyze  — read va_watchlist → call LLM (continuation_prediction stage)
    prune    — drop watchlist rows whose tracked age ≥ N calendar days
    history  — list recent runs
    report   — re-render a finished run's terminal summary
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

from deeptrade.core import paths
from deeptrade.core.config import ConfigService
from deeptrade.core.db import Database
from deeptrade.core.llm_manager import LLMManager

from .runner import (
    DEFAULT_PRUNE_DAYS,
    AnalyzeParams,
    EvaluateParams,
    PruneParams,
    ScreenParams,
    VaRunner,
    render_finished_run,
)
from .runtime import VaRuntime

app = typer.Typer(
    name="volume-anomaly",
    help="成交量异动策略 — 主板异动筛选 + LLM 主升浪启动预测。",
    no_args_is_help=True,
    add_completion=False,
)


def _open_runtime() -> tuple[Database, VaRuntime]:
    db = Database(paths.db_path())
    cfg = ConfigService(db)
    rt = VaRuntime(db=db, config=cfg, llms=LLMManager(db, cfg))
    return db, rt


@app.command("screen")
def cmd_screen(
    trade_date: Optional[str] = typer.Option(None, "--trade-date", help="YYYYMMDD"),
    allow_intraday: bool = typer.Option(False, "--allow-intraday"),
    force_sync: bool = typer.Option(False, "--force-sync"),
) -> None:
    """Apply local screening rules → upsert va_watchlist (no LLM)."""
    db, rt = _open_runtime()
    try:
        params = ScreenParams(
            trade_date=trade_date, allow_intraday=allow_intraday, force_sync=force_sync
        )
        outcome = VaRunner(rt).execute_screen(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
        render_finished_run(outcome.run_id)
    finally:
        db.close()


@app.command("analyze")
def cmd_analyze(
    trade_date: Optional[str] = typer.Option(None, "--trade-date", help="YYYYMMDD"),
    allow_intraday: bool = typer.Option(False, "--allow-intraday"),
    force_sync: bool = typer.Option(False, "--force-sync"),
) -> None:
    """LLM-driven trend analysis on the current watchlist."""
    db, rt = _open_runtime()
    try:
        params = AnalyzeParams(
            trade_date=trade_date, allow_intraday=allow_intraday, force_sync=force_sync
        )
        outcome = VaRunner(rt).execute_analyze(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
        render_finished_run(outcome.run_id)
    finally:
        db.close()


@app.command("prune")
def cmd_prune(
    days: int = typer.Option(
        DEFAULT_PRUNE_DAYS, "--days", help="Drop watchlist rows tracked for ≥ N days (0 = all)."
    ),
    trade_date: Optional[str] = typer.Option(None, "--trade-date", help="YYYYMMDD"),
    allow_intraday: bool = typer.Option(False, "--allow-intraday"),
) -> None:
    """Drop watchlist rows whose tracked age ≥ N calendar days."""
    db, rt = _open_runtime()
    try:
        params = PruneParams(trade_date=trade_date, allow_intraday=allow_intraday, days=days)
        outcome = VaRunner(rt).execute_prune(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value != "success":
            raise typer.Exit(1)
    finally:
        db.close()


@app.command("evaluate")
def cmd_evaluate(
    lookback_days: int = typer.Option(
        30, "--lookback-days",
        help="Evaluate hits whose anomaly_date is within the trailing N calendar days.",
    ),
    trade_date: Optional[str] = typer.Option(None, "--trade-date", help="YYYYMMDD"),
    backfill_all: bool = typer.Option(
        False, "--backfill-all",
        help="Override --lookback-days and evaluate every hit in va_anomaly_history.",
    ),
    force_recompute: bool = typer.Option(
        False, "--force-recompute",
        help="Re-evaluate rows already marked 'complete'.",
    ),
    force_sync: bool = typer.Option(False, "--force-sync"),
    allow_intraday: bool = typer.Option(False, "--allow-intraday"),
) -> None:
    """Compute T+N realized returns for past anomaly hits (writes va_realized_returns)."""
    db, rt = _open_runtime()
    try:
        params = EvaluateParams(
            trade_date=trade_date,
            allow_intraday=allow_intraday,
            lookback_days=lookback_days,
            backfill_all=backfill_all,
            force_recompute=force_recompute,
            force_sync=force_sync,
        )
        outcome = VaRunner(rt).execute_evaluate(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
        render_finished_run(outcome.run_id)
    finally:
        db.close()


@app.command("stats")
def cmd_stats(
    from_date: Optional[str] = typer.Option(
        None, "--from", help="Anomaly date lower bound, YYYYMMDD"
    ),
    to_date: Optional[str] = typer.Option(
        None, "--to", help="Anomaly date upper bound, YYYYMMDD"
    ),
    by: str = typer.Option(
        "prediction", "--by",
        help="Aggregation key: prediction | pattern | launch_score_bin",
    ),
) -> None:
    """Summarise va_realized_returns by prediction / pattern / launch_score_bin.

    Pure read-only query; does not fetch fresh data. Run `evaluate` first to
    populate va_realized_returns.
    """
    from .stats import run_stats_query

    db = Database(paths.db_path())
    try:
        rows, title = run_stats_query(
            db, from_date=from_date, to_date=to_date, by=by
        )
    finally:
        db.close()

    from .render import render_stats_table
    render_stats_table(rows, by=by, title=title)


@app.command("history")
def cmd_history(limit: int = typer.Option(20, "--limit")) -> None:
    """List recent runs."""
    db = Database(paths.db_path())
    try:
        rows = db.fetchall(
            "SELECT run_id, mode, trade_date, status, started_at, finished_at FROM va_runs "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
    finally:
        db.close()
    if not rows:
        typer.echo("(no runs)")
        return
    for r in rows:
        typer.echo(f"{r[0]}  {r[1]:<8}  {r[2]:<10}  {r[3]:<15}  {r[4]} → {r[5] or '-'}")


@app.command("report")
def cmd_report(
    run_id: str = typer.Argument(...),
    full: bool = typer.Option(False, "--full"),
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


def main(argv: list[str]) -> int:
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
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"✘ {type(e).__name__}: {e}\n")
        return 1
