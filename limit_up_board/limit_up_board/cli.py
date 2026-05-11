"""Plugin-managed CLI for limit-up-board.

Subcommands:
    run     — full pipeline (Step 0..5)
    sync    — data-only path (no LLM)
    history — list recent runs
    report  — re-render a finished run's terminal summary
    settings — interactive / show 运行参数
    lgb     — LightGBM 模型生命周期管理（v0.5+）

Invoked via the framework's pure pass-through dispatch:
    deeptrade limit-up-board <subcommand> [...]
"""

from __future__ import annotations

import sys
from dataclasses import replace

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
    help="本插件可持久化的运行参数（流通市值 / 当前股价 上限、LGB 评分开关等）。",
    no_args_is_help=False,
    add_completion=False,
    invoke_without_command=True,
)
app.add_typer(settings_app, name="settings")

lgb_app = typer.Typer(
    name="lgb",
    help="LightGBM 连板概率评分模型生命周期管理（v0.5+）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(lgb_app, name="lgb")


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
    no_lgb: bool = typer.Option(
        False,
        "--no-lgb",
        help=(
            "本次 run 禁用 LightGBM 评分；候选股的 lgb_score 字段为 None，"
            "R1/R2 prompt 仍能跑通。等价于 LubConfig.lgb_enabled=false 的一次性覆盖。"
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
            lgb_enabled=not no_lgb,
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
        # 用 dataclasses.replace 保留所有未交互的字段（如 v0.5 的 lgb_* 配置）。
        new_cfg = replace(cur, max_float_mv_yi=new_mv, max_close_yuan=new_close)
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
# lgb — LightGBM 评分模型生命周期（v0.5+ 骨架，逐 PR 填实现）
# ---------------------------------------------------------------------------


_LGB_PENDING_HINT = "（实现将在后续 PR 中补齐——本次仅落地 CLI 骨架与持久化表）"


def _fetch_lgb_models(db: Database) -> list[tuple]:
    """Return all rows in ``lub_lgb_models`` ordered by created_at DESC.

    Returns ``[]`` when the table doesn't exist (e.g. plugin newly installed but
    migration not yet applied via daily run) — keeps the CLI from raising.
    """
    try:
        return db.fetchall(
            "SELECT model_id, schema_version, train_start_date, train_end_date, "
            "n_samples, n_positive, cv_auc_mean, cv_auc_std, feature_count, "
            "plugin_version, file_path, is_active, created_at "
            "FROM lub_lgb_models ORDER BY created_at DESC"
        )
    except Exception:  # noqa: BLE001 — duckdb CatalogException 不公开导出
        return []


def _fetch_lgb_model(db: Database, model_id: str) -> tuple | None:
    try:
        return db.fetchone(
            "SELECT model_id, schema_version, train_start_date, train_end_date, "
            "n_samples, n_positive, cv_auc_mean, cv_auc_std, cv_logloss_mean, "
            "feature_count, feature_list_json, hyperparams_json, framework_version, "
            "plugin_version, git_commit, file_path, is_active, created_at "
            "FROM lub_lgb_models WHERE model_id = ?",
            (model_id,),
        )
    except Exception:  # noqa: BLE001
        return None


def _fetch_active_lgb_model(db: Database) -> tuple | None:
    try:
        return db.fetchone(
            "SELECT model_id FROM lub_lgb_models WHERE is_active = TRUE "
            "ORDER BY created_at DESC LIMIT 1"
        )
    except Exception:  # noqa: BLE001
        return None


@lgb_app.command("list")
def cmd_lgb_list() -> None:
    """列出所有已注册的 LightGBM 模型。★ 标记当前 active 的那一行。"""
    db = Database(paths.db_path())
    try:
        rows = _fetch_lgb_models(db)
    finally:
        db.close()
    if not rows:
        typer.echo("(no models registered)")
        return
    console = Console()
    table = Table(title="lub_lgb_models")
    table.add_column("", width=2)
    table.add_column("model_id", style="cyan")
    table.add_column("schema", justify="right")
    table.add_column("train window", overflow="fold")
    table.add_column("samples", justify="right")
    table.add_column("pos", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("features", justify="right")
    table.add_column("plugin_ver")
    table.add_column("created_at")
    for r in rows:
        (
            model_id, schema_v, t0, t1, n, n_pos,
            auc_mean, auc_std, feat_count, plugin_v, _file, is_active, created,
        ) = r
        active_mark = "★" if is_active else ""
        auc_repr = (
            f"{auc_mean:.3f}±{(auc_std or 0):.3f}" if auc_mean is not None else "—"
        )
        table.add_row(
            active_mark,
            str(model_id),
            str(schema_v),
            f"{t0}..{t1}",
            str(n),
            str(n_pos),
            auc_repr,
            str(feat_count),
            str(plugin_v),
            str(created),
        )
    console.print(table)


@lgb_app.command("info")
def cmd_lgb_info(
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="模型 ID；不指定则展示当前 active 模型；无 active 时报错。",
    ),
) -> None:
    """展示单个模型的详细信息（CV 指标 / 特征数 / 超参 / 文件路径）。"""
    db = Database(paths.db_path())
    try:
        if model_id is None:
            active = _fetch_active_lgb_model(db)
            if active is None:
                typer.echo("(no active model — 用 --model-id 指定查询目标，或先运行 lgb train)")
                raise typer.Exit(2)
            model_id = str(active[0])
        row = _fetch_lgb_model(db, model_id)
    finally:
        db.close()
    if row is None:
        typer.echo(f"✘ model_id not found: {model_id!r}")
        raise typer.Exit(2)
    (
        mid, schema_v, t0, t1, n, n_pos, auc_mean, auc_std, logloss_mean,
        feat_count, feat_list_json, hp_json, fw_ver, plugin_v, git_commit,
        file_path, is_active, created,
    ) = row
    console = Console()
    table = Table(title=f"lgb model: {mid}", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value", overflow="fold")
    table.add_row("active", "★ yes" if is_active else "no")
    table.add_row("schema_version", str(schema_v))
    table.add_row("train window", f"{t0} .. {t1}")
    table.add_row("n_samples / n_positive", f"{n} / {n_pos}")
    if auc_mean is not None:
        table.add_row("CV AUC (mean±std)", f"{auc_mean:.4f} ± {(auc_std or 0):.4f}")
    if logloss_mean is not None:
        table.add_row("CV logloss", f"{logloss_mean:.4f}")
    table.add_row("feature_count", str(feat_count))
    table.add_row("plugin_version", str(plugin_v))
    if fw_ver:
        table.add_row("framework_version", str(fw_ver))
    if git_commit:
        table.add_row("git_commit", str(git_commit))
    table.add_row("file_path", str(file_path))
    table.add_row("created_at", str(created))
    table.add_row("feature_list_json (preview)", str(feat_list_json)[:120] + "...")
    table.add_row("hyperparams_json", str(hp_json))
    console.print(table)


@lgb_app.command("train")
def cmd_lgb_train(
    start: str = typer.Option(..., "--start", help="训练窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="训练窗口结束日期 YYYYMMDD"),
    folds: int = typer.Option(5, "--folds", help="GroupKFold 折数"),
    no_activate: bool = typer.Option(False, "--no-activate", help="训练完成后不切换 active 模型"),
) -> None:
    """训练新的 LightGBM 模型并落库（后续 PR 实现）。"""
    typer.echo(
        f"Not yet implemented in this iteration: lgb train --start {start} --end {end} "
        f"--folds {folds} --no-activate={no_activate}"
    )
    typer.echo(_LGB_PENDING_HINT)
    raise typer.Exit(2)


@lgb_app.command("evaluate")
def cmd_lgb_evaluate(
    start: str = typer.Option(..., "--start", help="评估窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="评估窗口结束日期 YYYYMMDD"),
    model_id: str | None = typer.Option(None, "--model-id"),
    drift: bool = typer.Option(False, "--drift", help="顺便输出特征 drift 报表（PR-3.3）"),
) -> None:
    """对指定窗口跑离线评估（后续 PR 实现）。"""
    typer.echo(
        f"Not yet implemented in this iteration: lgb evaluate --start {start} --end {end} "
        f"--model-id={model_id} --drift={drift}"
    )
    typer.echo(_LGB_PENDING_HINT)
    raise typer.Exit(2)


@lgb_app.command("activate")
def cmd_lgb_activate(
    model_id: str = typer.Argument(..., help="要激活的模型 ID"),
) -> None:
    """原子切换 active 模型（后续 PR 实现）。"""
    typer.echo(f"Not yet implemented in this iteration: lgb activate {model_id}")
    typer.echo(_LGB_PENDING_HINT)
    raise typer.Exit(2)


@lgb_app.command("prune")
def cmd_lgb_prune(
    keep: int = typer.Option(5, "--keep", help="保留最近的 N 个模型（含 active）"),
    keep_rows: bool = typer.Option(False, "--keep-rows", help="仅删模型文件，保留 DB 行"),
) -> None:
    """清理旧模型文件 / 行（后续 PR 实现）。"""
    typer.echo(
        f"Not yet implemented in this iteration: lgb prune --keep {keep} --keep-rows={keep_rows}"
    )
    typer.echo(_LGB_PENDING_HINT)
    raise typer.Exit(2)


@lgb_app.command("refresh-features")
def cmd_lgb_refresh_features(
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
) -> None:
    """仅拉历史数据 / 不训练（后续 PR 实现）。"""
    typer.echo(
        f"Not yet implemented in this iteration: lgb refresh-features --start={start} --end={end}"
    )
    typer.echo(_LGB_PENDING_HINT)
    raise typer.Exit(2)


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
