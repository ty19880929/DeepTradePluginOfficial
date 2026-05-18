"""Plugin-managed CLI for accumulation-probe-washout.

Subcommands (in v0.1):
    screen   — apply local rules → write apw_signal_history + apw_watchlist (no LLM)
    analyze  — read watchlist → LLM batch → apw_stage_results (M3)
    run      — screen → analyze (M4)
    evaluate — T+N realised returns (M5)
    settings show / set — read/write apw_config (M6)
    history  — recent apw_runs (M6)
    report --run-id — re-render a finished run (M6)
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

from deeptrade.core import paths
from deeptrade.core.config import ConfigService
from deeptrade.core.db import Database
from deeptrade.core.llm_manager import LLMManager
from deeptrade.core.run_status import RunStatus

from .cancellation import install_sigint_marker
from .runner import AnalyzeParams, ApwRunner, EvaluateParams, RunParams, ScreenParams
from .runtime import ApwRuntime
from .ui.legacy import LegacyStreamRenderer

app = typer.Typer(
    name="accumulation-probe-washout",
    help="吸筹试盘洗盘主升浪策略 — 主板吸筹/试盘/洗盘链路 + LLM 主升浪启动预测。",
    no_args_is_help=True,
    add_completion=False,
)

settings_app = typer.Typer(
    name="settings",
    help="读写本插件可调参数（apw_config）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(settings_app, name="settings")


def _open_runtime() -> tuple[Database, ApwRuntime]:
    db = Database(paths.db_path())
    cfg = ConfigService(db)
    rt = ApwRuntime(db=db, config=cfg, llms=LLMManager(db, cfg))
    return db, rt


# ---------------------------------------------------------------------------
# screen (M2)
# ---------------------------------------------------------------------------


@app.command("screen")
def cmd_screen(
    trade_date: Optional[str] = typer.Option(None, "--date", help="YYYYMMDD"),
    max_candidates: Optional[int] = typer.Option(
        None, "--max-candidates", help="上限：每轮 LLM 批次喂入的候选数量（默认读 apw_config）"
    ),
    force_sync: bool = typer.Option(False, "--force-sync"),
    no_dashboard: bool = typer.Option(  # noqa: ARG001 — wired in M4
        False, "--no-dashboard",
        help="禁用动态仪表盘 (M4 起生效)。",
    ),
) -> None:
    """Apply local screening rules → write apw_signal_history + apw_watchlist (no LLM)."""
    from .ui import choose_renderer

    db, rt = _open_runtime()
    try:
        params = ScreenParams(
            trade_date=trade_date,
            force_sync=force_sync,
            max_candidates=max_candidates,
        )
        renderer = choose_renderer(no_dashboard=no_dashboard, mode="screen")
        outcome = ApwRunner(rt, renderer=renderer).execute_screen(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.status == RunStatus.CANCELLED:
            typer.echo("message: 用户手动中断，已停止当前策略执行。")
            raise typer.Exit(130)
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# analyze / run / evaluate — wired in M3 / M4 / M5
# ---------------------------------------------------------------------------


@app.command("analyze")
def cmd_analyze(
    trade_date: Optional[str] = typer.Option(None, "--date", help="YYYYMMDD"),
    max_candidates: Optional[int] = typer.Option(None, "--max-candidates"),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm",
        help="LLM provider 名（覆盖框架默认；未配置则 PreconditionError 不写 run 行）",
    ),
    prediction_filter: Optional[str] = typer.Option(
        None, "--prediction",
        help="只分析指定 phase 的候选股（如 launch_ready）",
    ),
    no_dashboard: bool = typer.Option(False, "--no-dashboard"),
) -> None:
    """Read apw_watchlist → LLM → apw_stage_results."""
    from .ui import choose_renderer

    db, rt = _open_runtime()
    try:
        params = AnalyzeParams(
            trade_date=trade_date,
            max_candidates=max_candidates,
            llm_provider=llm_provider,
            prediction_filter=prediction_filter,
        )
        renderer = choose_renderer(no_dashboard=no_dashboard, mode="analyze")
        outcome = ApwRunner(rt, renderer=renderer).execute_analyze(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.status == RunStatus.CANCELLED:
            typer.echo("message: 用户手动中断，已停止当前策略执行。")
            raise typer.Exit(130)
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
    finally:
        db.close()


@app.command("run")
def cmd_run(
    trade_date: Optional[str] = typer.Option(None, "--date", help="YYYYMMDD"),
    max_candidates: Optional[int] = typer.Option(None, "--max-candidates"),
    force_sync: bool = typer.Option(False, "--force-sync"),
    llm_provider: Optional[str] = typer.Option(None, "--llm"),
    no_dashboard: bool = typer.Option(False, "--no-dashboard"),
) -> None:
    """One-shot screen → analyze (用户最常用入口)."""
    from .ui import choose_renderer

    db, rt = _open_runtime()
    try:
        params = RunParams(
            trade_date=trade_date,
            force_sync=force_sync,
            max_candidates=max_candidates,
            llm_provider=llm_provider,
        )
        renderer = choose_renderer(no_dashboard=no_dashboard, mode="run")
        outcome = ApwRunner(rt, renderer=renderer).execute_run(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.status == RunStatus.CANCELLED:
            typer.echo("message: 用户手动中断，已停止当前策略执行。")
            raise typer.Exit(130)
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
    finally:
        db.close()


@app.command("evaluate")
def cmd_evaluate(
    from_date: Optional[str] = typer.Option(None, "--from-date", help="YYYYMMDD"),
    to_date: Optional[str] = typer.Option(None, "--to-date", help="YYYYMMDD"),
    horizons: str = typer.Option("1,3,5,10", "--horizons"),
    include_early_phases: bool = typer.Option(
        False, "--include-early-phases",
        help="D3: 默认只统计 ≥washing_after_probe；本 flag 纳入 accumulating / probe_seen",
    ),
    force_recompute: bool = typer.Option(
        False, "--force-recompute", help="重算已有 data_status=complete 行",
    ),
) -> None:
    """T+N realised returns + group reports (forced legacy renderer)."""
    db, rt = _open_runtime()
    try:
        params = EvaluateParams(
            from_date=from_date,
            to_date=to_date,
            horizons=horizons,
            include_early_phases=include_early_phases,
            force_recompute=force_recompute,
        )
        # evaluate is forced legacy — the dashboard's single-day StageStack
        # would mislead for a multi-day backfill loop.
        outcome = ApwRunner(rt, renderer=LegacyStreamRenderer()).execute_evaluate(params)
        typer.echo(f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}")
        if outcome.status == RunStatus.CANCELLED:
            typer.echo("message: 用户手动中断，已停止当前策略执行。")
            raise typer.Exit(130)
        if outcome.error:
            typer.echo(f"error: {outcome.error}")
        if outcome.status.value not in {"success", "partial_failed"}:
            raise typer.Exit(1)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# settings / history / report — wired in M6
# ---------------------------------------------------------------------------


@settings_app.command("show")
def cmd_settings_show() -> None:
    """Print current apw_config (overrides) + default values."""
    from .config import ALLOWED_KEYS, ApwConfigStore, to_dict

    db, _rt = _open_runtime()
    try:
        store = ApwConfigStore(db)
        cfg = store.load()
        defaults = to_dict(cfg)
        # Show DB-resident overrides explicitly.
        overrides = dict(store.items())
        typer.echo("\n=== apw settings ===")
        for key in sorted(ALLOWED_KEYS):
            cur = defaults.get(key)
            ovr = overrides.get(key)
            tag = "  (override)" if key in overrides else ""
            shown = ovr if key in overrides else cur
            typer.echo(f"  {key:40s} = {shown}{tag}")
    finally:
        db.close()


@settings_app.command("set")
def cmd_settings_set(
    key: str = typer.Argument(...),
    value: str = typer.Argument(...),
) -> None:
    """Set one apw_config key (value parsed as JSON; falls back to raw string)."""
    import json
    from .config import ALLOWED_KEYS, ApwConfigStore

    if key not in ALLOWED_KEYS:
        typer.echo(f"✘ unknown key: {key!r}")
        typer.echo(f"  allowed: {', '.join(sorted(ALLOWED_KEYS))}")
        raise typer.Exit(2)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value

    db, _rt = _open_runtime()
    try:
        ApwConfigStore(db).set(key, parsed)
        typer.echo(f"✓ {key} = {parsed!r}")
    finally:
        db.close()


@app.command("history")
def cmd_history(
    limit: int = typer.Option(20, "--limit", help="最近 N 条"),
) -> None:
    """List recent runs from apw_runs."""
    from rich.console import Console
    from rich.table import Table

    db, _rt = _open_runtime()
    try:
        rows = db.fetchall(
            "SELECT run_id, mode, trade_date, status, started_at, finished_at "
            "FROM apw_runs ORDER BY started_at DESC LIMIT ?",
            [limit],
        ) or []
        console = Console()
        tbl = Table(title="apw_runs (latest first)")
        tbl.add_column("run_id"); tbl.add_column("mode"); tbl.add_column("trade_date")
        tbl.add_column("status"); tbl.add_column("started_at"); tbl.add_column("finished_at")
        for r in rows:
            tbl.add_row(
                str(r[0])[:8], str(r[1]), str(r[2]), str(r[3]),
                str(r[4])[:19] if r[4] else "",
                str(r[5])[:19] if r[5] else "",
            )
        console.print(tbl)
    finally:
        db.close()


@app.command("report")
def cmd_report(
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Re-render a finished analyze run from apw_stage_results."""
    from rich.console import Console
    from rich.table import Table

    db, _rt = _open_runtime()
    try:
        rows = db.fetchall(
            """
            SELECT rank, ts_code, prediction, main_pattern, confidence,
                   launch_score, phase, rationale
            FROM apw_stage_results
            WHERE run_id = ?
            ORDER BY rank ASC
            """,
            [run_id],
        ) or []
        if not rows:
            typer.echo(f"✘ no stage_results rows for run_id {run_id}")
            raise typer.Exit(1)
        console = Console()
        tbl = Table(title=f"apw analyze report — run_id {run_id[:8]}")
        for col in ("rank", "ts_code", "prediction", "main_pattern", "confidence",
                    "launch_score", "phase", "rationale"):
            tbl.add_column(col, overflow="fold")
        for r in rows:
            tbl.add_row(
                str(r[0]), str(r[1]), str(r[2]), str(r[3]),
                str(r[4]), str(round(r[5], 2)), str(r[6]),
                str(r[7] or "")[:80],
            )
        console.print(tbl)
    finally:
        db.close()


def main(argv: list[str]) -> int:
    """Entry called by the framework via Plugin.dispatch().

    With ``standalone_mode=False`` typer/click do NOT call sys.exit; they
    return the exit code from app(...) directly (and ALSO surface
    typer.Exit if raised before invoke completes).
    """
    # v0.1.1 — install the process-wide SIGINT marker before any runner work
    # so derived exceptions (DuckDB InterruptException, requests SSL break,
    # …) that bubble out of a Ctrl+C'd run can be reclassified as CANCELLED
    # instead of FAILED. See cancellation.py for the contract.
    install_sigint_marker()
    try:
        rc = app(args=argv, standalone_mode=False)
    except typer.Exit as e:
        return int(e.exit_code)
    except SystemExit as e:
        return int(e.code or 0)
    except KeyboardInterrupt:
        sys.stderr.write("\n⏹ 用户手动中断，已停止当前策略执行。\n")
        return 130
    if isinstance(rc, int):
        return rc
    return 0
