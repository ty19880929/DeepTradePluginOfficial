"""Plugin-managed CLI for volume-anomaly.

Subcommands:
    screen   — apply local rules → upsert va_watchlist (no LLM)
    analyze  — read va_watchlist → call LLM (continuation_prediction stage)
    prune    — drop watchlist rows whose tracked age ≥ N calendar days
    history  — list recent runs
    report   — re-render a finished run's terminal summary
    stats    — read-only aggregates over va_realized_returns
    lgb      — LightGBM 主升浪启动概率模型生命周期管理（v0.7+）
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from deeptrade.core import paths
from deeptrade.core.config import ConfigService
from deeptrade.core.db import Database
from deeptrade.core.llm_manager import LLMManager
from deeptrade.plugins_api import render_exception

from .lgb import cleanup as lgb_cleanup
from .lgb import paths as lgb_paths
from .lgb import registry as lgb_registry
from .lgb.config import VaLgbConfig, list_for_show, load_config, save_config
from .runner import (
    DEFAULT_PRUNE_DAYS,
    AnalyzeParams,
    BackfillHistoryParams,
    EvaluateParams,
    PruneParams,
    ScreenParams,
    VaRunner,
    render_finished_run,
)
from .runtime import VaRuntime
from .ui import choose_renderer
from .ui.legacy import LegacyStreamRenderer

app = typer.Typer(
    name="volume-anomaly",
    help="成交量异动策略 — 主板异动筛选 + LLM 主升浪启动预测。",
    no_args_is_help=True,
    add_completion=False,
)

lgb_app = typer.Typer(
    name="lgb",
    help="LightGBM 主升浪启动概率评分模型生命周期管理（v0.7+）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(lgb_app, name="lgb")


_LGB_PENDING_HINT = (
    "提示：lgb train / evaluate / refresh-features 计划在后续 PR 中落地。"
    "本 PR 仅交付 CLI 骨架；模型注册 / 激活 / 清理已可用。"
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
    backfill_history: bool = typer.Option(
        False,
        "--backfill-history",
        help=(
            "v0.9.0 — 批量重放 screen 规则到历史区间，"
            "纯填充 va_anomaly_history，不进 LLM、不动 va_watchlist。"
            "为新用户 bootstrap LightGBM 训练样本。需要 --start / --end。"
        ),
    ),
    start: Optional[str] = typer.Option(
        None, "--start",
        help="--backfill-history 模式下起始 trade_date (YYYYMMDD, 含)",
    ),
    end: Optional[str] = typer.Option(
        None, "--end",
        help="--backfill-history 模式下结束 trade_date (YYYYMMDD, 含)",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help=(
            "--backfill-history 模式下,对已存在 va_anomaly_history 行的日期"
            "也重新筛选 (DELETE 后 INSERT)。默认跳过已有日期 (作为 resume)。"
        ),
    ),
    no_dashboard: bool = typer.Option(
        False,
        "--no-dashboard",
        help=(
            "禁用动态仪表盘，使用与 v0.7.x 兼容的流式日志输出。"
            "也可通过环境变量 DEEPTRADE_NO_DASHBOARD=1 全局禁用。"
        ),
    ),
) -> None:
    """Apply local screening rules → upsert va_watchlist (no LLM).

    With ``--backfill-history --start --end`` switches to LLM-free batch
    replay mode: iterates every open trade_date in [start, end] and writes
    hits to ``va_anomaly_history`` only (does NOT touch ``va_watchlist``).
    """
    if backfill_history:
        if not start or not end:
            typer.echo("✘ --backfill-history requires both --start and --end")
            raise typer.Exit(2)
        if len(start) != 8 or len(end) != 8 or not start.isdigit() or not end.isdigit():
            typer.echo("✘ --start / --end must be YYYYMMDD")
            raise typer.Exit(2)
        if start > end:
            typer.echo(f"✘ --start ({start}) > --end ({end})")
            raise typer.Exit(2)
        if trade_date is not None:
            typer.echo("✘ --trade-date is incompatible with --backfill-history")
            raise typer.Exit(2)
        if allow_intraday:
            typer.echo("✘ --allow-intraday is incompatible with --backfill-history")
            raise typer.Exit(2)
        db, rt = _open_runtime()
        try:
            bh_params = BackfillHistoryParams(
                start_date=start,
                end_date=end,
                force_sync=force_sync,
                overwrite=overwrite,
            )
            # backfill_history is intentionally forced-legacy (mirrors prune /
            # evaluate): the dashboard's StageStack model is single-day; a
            # multi-day loop would render misleading progress.
            outcome = VaRunner(
                rt, renderer=LegacyStreamRenderer()
            ).execute_backfill_history(bh_params)
            typer.echo(
                f"\nstatus: {outcome.status.value}  run_id: {outcome.run_id}"
            )
            if outcome.error:
                typer.echo(f"error: {outcome.error}")
            if outcome.status.value != "success":
                raise typer.Exit(1)
        finally:
            db.close()
        return

    # Live screen path (--start / --end / --overwrite forbidden here).
    if start is not None or end is not None:
        typer.echo("✘ --start / --end only valid with --backfill-history")
        raise typer.Exit(2)
    if overwrite:
        typer.echo("✘ --overwrite only valid with --backfill-history")
        raise typer.Exit(2)

    db, rt = _open_runtime()
    try:
        params = ScreenParams(
            trade_date=trade_date, allow_intraday=allow_intraday, force_sync=force_sync
        )
        renderer = choose_renderer(no_dashboard=no_dashboard)
        outcome = VaRunner(rt, renderer=renderer).execute_screen(params)
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
    no_lgb: bool = typer.Option(
        False,
        "--no-lgb",
        help=(
            "本次 analyze 禁用 LightGBM 评分；候选股的 lgb_score 字段为 None，"
            "LLM 走势分析仍能跑通。等价于 VaLgbConfig.lgb_enabled=false 的一次性覆盖。"
        ),
    ),
    no_dashboard: bool = typer.Option(
        False,
        "--no-dashboard",
        help=(
            "禁用动态仪表盘，使用与 v0.7.x 兼容的流式日志输出。"
            "也可通过环境变量 DEEPTRADE_NO_DASHBOARD=1 全局禁用。"
        ),
    ),
) -> None:
    """LLM-driven trend analysis on the current watchlist."""
    db, rt = _open_runtime()
    try:
        params = AnalyzeParams(
            trade_date=trade_date,
            allow_intraday=allow_intraday,
            force_sync=force_sync,
            lgb_enabled=not no_lgb,
        )
        renderer = choose_renderer(no_dashboard=no_dashboard)
        outcome = VaRunner(rt, renderer=renderer).execute_analyze(params)
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
        # Plan §3.4.2 / §7 P-7 — prune is intentionally a legacy-only path;
        # no --no-dashboard flag is exposed and choose_renderer() is bypassed.
        outcome = VaRunner(rt, renderer=LegacyStreamRenderer()).execute_prune(params)
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
        # Plan §3.4.2 / §7 P-7 — evaluate is intentionally a legacy-only path;
        # no --no-dashboard flag is exposed and choose_renderer() is bypassed.
        outcome = VaRunner(rt, renderer=LegacyStreamRenderer()).execute_evaluate(params)
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
        help=(
            "Aggregation key: prediction | pattern | launch_score_bin | "
            "dimension_scores | lgb_score_bin"
        ),
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
        from rich.console import Console as _Console
        from rich.markdown import Markdown

        from deeptrade.theme import EVA_THEME

        report_dir = paths.reports_dir() / run_id
        summary = report_dir / "summary.md"
        if not summary.is_file():
            typer.echo(f"✘ no report at {summary}")
            raise typer.Exit(2)
        _Console(theme=EVA_THEME).print(Markdown(summary.read_text(encoding="utf-8")))
        typer.echo(f"\nReport directory: {summary.parent}")
        return
    render_finished_run(run_id)


# ---------------------------------------------------------------------------
# settings (LGB config visibility) — minimal show/set, PR-0.3
# ---------------------------------------------------------------------------

settings_app = typer.Typer(
    name="settings",
    help="本插件可持久化的运行参数（当前仅 LGB 评分相关字段；v0.7+）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(settings_app, name="settings")


@settings_app.command("show")
def cmd_settings_show() -> None:
    """List every va_config field with its persisted / default source."""
    db = Database(paths.db_path())
    try:
        rows = list_for_show(db)
    finally:
        db.close()
    console = Console()
    table = Table(title="volume-anomaly settings")
    table.add_column("key", style="cyan")
    table.add_column("value")
    table.add_column("source")
    for key, value, source in rows:
        table.add_row(key, str(value), source)
    console.print(table)


@settings_app.command("reset")
def cmd_settings_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Wipe all va_config rows → next read falls back to dataclass defaults."""
    if not yes:
        confirm = typer.confirm("确认重置所有 va_config 行？此操作不可恢复")
        if not confirm:
            typer.echo("✘ 已取消")
            raise typer.Exit(1)
    db = Database(paths.db_path())
    try:
        try:
            db.execute("DELETE FROM va_config")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"⚠ va_config truncate failed: {e}")
            raise typer.Exit(2) from e
    finally:
        db.close()
    typer.echo("✔ va_config 已清空")


# ---------------------------------------------------------------------------
# lgb subgroup — PR-0.3 scaffolding
# ---------------------------------------------------------------------------


def _plugin_version() -> str:
    yaml_path = Path(__file__).resolve().parent.parent / "deeptrade_plugin.yaml"
    if not yaml_path.is_file():
        return "unknown"
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        return str(data.get("version", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def _framework_version() -> str | None:
    try:
        import deeptrade  # noqa: PLC0415

        return getattr(deeptrade, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def _lightgbm_version() -> str | None:
    try:
        import lightgbm  # noqa: PLC0415

        return getattr(lightgbm, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def _git_short_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _safe_list_models(db: Database) -> list[lgb_registry.ModelRecord]:
    try:
        return lgb_registry.list_models(db)
    except Exception:  # noqa: BLE001 — table missing on legacy DBs
        return []


def _safe_get_model(db: Database, model_id: str) -> lgb_registry.ModelRecord | None:
    try:
        return lgb_registry.get_model(db, model_id)
    except Exception:  # noqa: BLE001
        return None


def _safe_get_active(db: Database) -> lgb_registry.ModelRecord | None:
    try:
        return lgb_registry.get_active(db)
    except Exception:  # noqa: BLE001
        return None


@lgb_app.command("list")
def cmd_lgb_list() -> None:
    """列出所有已注册的 LightGBM 模型。★ 标记当前 active 的那一行。"""
    db = Database(paths.db_path())
    try:
        records = _safe_list_models(db)
    finally:
        db.close()
    if not records:
        typer.echo("(no models registered)")
        return
    console = Console()
    table = Table(title="va_lgb_models")
    table.add_column("", width=2)
    table.add_column("model_id", style="cyan")
    table.add_column("schema", justify="right")
    table.add_column("train window", overflow="fold")
    table.add_column("samples", justify="right")
    table.add_column("pos", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("label", overflow="fold")
    table.add_column("features", justify="right")
    table.add_column("plugin_ver")
    table.add_column("created_at")
    for r in records:
        auc_repr = (
            f"{r.cv_auc_mean:.3f}±{(r.cv_auc_std or 0):.3f}"
            if r.cv_auc_mean is not None
            else "—"
        )
        label_repr = f"{r.label_source}≥{r.label_threshold_pct:g}%"
        table.add_row(
            "★" if r.is_active else "",
            r.model_id,
            str(r.schema_version),
            f"{r.train_start_date}..{r.train_end_date}",
            str(r.n_samples),
            str(r.n_positive),
            auc_repr,
            label_repr,
            str(r.feature_count),
            r.plugin_version,
            str(r.created_at) if r.created_at is not None else "—",
        )
    console.print(table)


def _safe_predictions_usage(db: Database, model_id: str) -> dict[str, int] | None:
    """Aggregate (n_runs, n_trade_dates, n_rows) for a given model_id from
    ``va_lgb_predictions``. Returns None if the table is missing."""
    try:
        row = db.fetchone(
            "SELECT COUNT(DISTINCT run_id), COUNT(DISTINCT trade_date), COUNT(*) "
            "FROM va_lgb_predictions WHERE model_id = ?",
            (model_id,),
        )
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return {"n_runs": 0, "n_trade_dates": 0, "n_rows": 0}
    return {
        "n_runs": int(row[0] or 0),
        "n_trade_dates": int(row[1] or 0),
        "n_rows": int(row[2] or 0),
    }


def _safe_predictions_recent(
    db: Database, model_id: str, recent_n: int
) -> list[dict[str, Any]]:
    """Per-day score distribution snapshot from va_lgb_predictions (most recent
    ``recent_n`` trade_dates)."""
    if recent_n <= 0:
        return []
    try:
        rows = db.fetchall(
            "SELECT trade_date, COUNT(*) AS n, MIN(lgb_score) AS lo, "
            "quantile_cont(lgb_score, 0.25) AS q25, "
            "quantile_cont(lgb_score, 0.5)  AS med, "
            "quantile_cont(lgb_score, 0.75) AS q75, "
            "MAX(lgb_score) AS hi "
            "FROM va_lgb_predictions WHERE model_id = ? "
            "GROUP BY trade_date "
            "ORDER BY trade_date DESC "
            "LIMIT ?",
            (model_id, recent_n),
        )
    except Exception:  # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "trade_date": str(r[0]),
                "n": int(r[1] or 0),
                "min": float(r[2]) if r[2] is not None else 0.0,
                "p25": float(r[3]) if r[3] is not None else 0.0,
                "median": float(r[4]) if r[4] is not None else 0.0,
                "p75": float(r[5]) if r[5] is not None else 0.0,
                "max": float(r[6]) if r[6] is not None else 0.0,
            }
        )
    return list(reversed(out))  # ascending date for readability


@lgb_app.command("info")
def cmd_lgb_info(
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        help="模型 ID；不指定则展示当前 active 模型；无 active 时报错。",
    ),
    recent_n: int = typer.Option(
        0,
        "--recent-N",
        help=(
            "额外列出最近 N 天每天的 candidate × score 分布快照"
            "（数据来源：va_lgb_predictions；由 `analyze` 在线上推理时写入。"
            "刚训练完未跑过 analyze 时为空属正常现象）"
        ),
    ),
) -> None:
    """展示单个模型的详细信息（CV 指标 / 特征数 / 超参 / 标签语义 / 文件路径 +
    使用统计 + 可选的最近 N 天 score 分布）。"""
    db = Database(paths.db_path())
    try:
        if model_id is None:
            active = _safe_get_active(db)
            if active is None:
                typer.echo(
                    "(no active model — 用 --model-id 指定查询目标，或先运行 lgb train)"
                )
                raise typer.Exit(2)
            model_id = active.model_id
        record = _safe_get_model(db, model_id)
        usage = _safe_predictions_usage(db, model_id) if record is not None else None
        recent = (
            _safe_predictions_recent(db, model_id, recent_n)
            if record is not None and recent_n > 0
            else []
        )
    finally:
        db.close()
    if record is None:
        typer.echo(f"✘ model_id not found: {model_id!r}")
        raise typer.Exit(2)
    console = Console()
    table = Table(title=f"lgb model: {record.model_id}", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value", overflow="fold")
    table.add_row("active", "★ yes" if record.is_active else "no")
    table.add_row("schema_version", str(record.schema_version))
    table.add_row("train window", f"{record.train_start_date} .. {record.train_end_date}")
    table.add_row("n_samples / n_positive", f"{record.n_samples} / {record.n_positive}")
    if record.cv_auc_mean is not None:
        table.add_row(
            "CV AUC (mean±std)",
            f"{record.cv_auc_mean:.4f} ± {(record.cv_auc_std or 0):.4f}",
        )
    if record.cv_logloss_mean is not None:
        table.add_row("CV logloss", f"{record.cv_logloss_mean:.4f}")
    table.add_row(
        "label", f"{record.label_source} ≥ {record.label_threshold_pct:g}%"
    )
    table.add_row("feature_count", str(record.feature_count))
    table.add_row("plugin_version", record.plugin_version)
    if record.framework_version:
        table.add_row("framework_version", record.framework_version)
    if record.git_commit:
        table.add_row("git_commit", record.git_commit)
    table.add_row("file_path", record.file_path)
    table.add_row("created_at", str(record.created_at) if record.created_at else "—")
    table.add_row("feature_list_json (preview)", record.feature_list_json[:120] + "...")
    table.add_row("hyperparams_json", record.hyperparams_json)
    if usage is not None:
        if usage["n_rows"]:
            table.add_row(
                "predictions usage",
                (
                    f"runs={usage['n_runs']}  trade_dates={usage['n_trade_dates']}  "
                    f"rows={usage['n_rows']}"
                ),
            )
        else:
            table.add_row(
                "predictions usage",
                "(尚无推理记录 — 用 `deeptrade volume-anomaly analyze` 触发一次 LGB 评分)",
            )
    console.print(table)

    if recent_n > 0:
        recent_table = Table(
            title=(
                f"recent {recent_n} trade dates · run-time score distribution "
                f"(from va_lgb_predictions)"
            ),
            show_header=True,
            header_style="cyan",
        )
        recent_table.add_column("trade_date")
        recent_table.add_column("n", justify="right")
        recent_table.add_column("min", justify="right")
        recent_table.add_column("p25", justify="right")
        recent_table.add_column("median", justify="right")
        recent_table.add_column("p75", justify="right")
        recent_table.add_column("max", justify="right")
        if not recent:
            recent_table.add_row("(none)", "—", "—", "—", "—", "—", "—")
        else:
            for row in recent:
                recent_table.add_row(
                    row["trade_date"],
                    str(row["n"]),
                    f"{row['min']:.1f}",
                    f"{row['p25']:.1f}",
                    f"{row['median']:.1f}",
                    f"{row['p75']:.1f}",
                    f"{row['max']:.1f}",
                )
        console.print(recent_table)
        if not recent:
            typer.echo(
                "提示：此表数据由 `deeptrade volume-anomaly analyze` 在线上推理时写入；"
                "刚训练完未跑过 analyze 时为空属正常现象。"
            )


@lgb_app.command("activate")
def cmd_lgb_activate(
    model_id: str = typer.Argument(..., help="要激活的模型 ID"),
) -> None:
    """原子切换 active 模型。"""
    db = Database(paths.db_path())
    try:
        try:
            ok = lgb_registry.set_active(db, model_id)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            typer.echo(f"✘ model_id not found: {model_id!r}")
            raise typer.Exit(2)
        record = _safe_get_model(db, model_id)
        if record is not None:
            lgb_paths.ensure_layout()
            lgb_paths.latest_pointer().write_text(record.file_path, encoding="utf-8")
        typer.echo(f"✔ activated: {model_id}")
    finally:
        db.close()


@lgb_app.command("prune")
def cmd_lgb_prune(
    keep: int = typer.Option(5, "--keep", help="保留最近的 N 个模型（含 active）"),
    keep_rows: bool = typer.Option(False, "--keep-rows", help="仅删模型文件，保留 DB 行"),
) -> None:
    """清理旧模型文件 / 行；active 模型永远保留。"""
    if keep < 1:
        typer.echo("✘ --keep 必须 ≥ 1")
        raise typer.Exit(2)
    db = Database(paths.db_path())
    try:
        records = _safe_list_models(db)
        keep_ids: set[str] = set()
        kept = 0
        for r in records:
            if r.is_active or kept < keep:
                keep_ids.add(r.model_id)
                if not r.is_active:
                    kept += 1
        removed = 0
        for r in records:
            if r.model_id in keep_ids:
                continue
            model_file = lgb_paths.plugin_data_dir() / r.file_path
            meta_file = lgb_paths.models_dir() / lgb_paths.meta_file_name(r.model_id)
            dataset_file = lgb_paths.datasets_dir() / lgb_paths.dataset_file_name(r.model_id)
            for f in (model_file, meta_file, dataset_file):
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass
            if not keep_rows:
                lgb_registry.delete_model(db, r.model_id)
            removed += 1
        typer.echo(
            f"✔ pruned {removed} model(s); kept {len(keep_ids)} "
            f"(active preserved; keep_rows={keep_rows})"
        )
    finally:
        db.close()


@lgb_app.command("purge")
def cmd_lgb_purge(
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过交互确认"),
    datasets: bool = typer.Option(
        False, "--datasets", help="清空 datasets/*.parquet 训练矩阵快照"
    ),
    models: bool = typer.Option(
        False,
        "--models",
        help="清空 models/ 落盘文件 + va_lgb_models 注册行（active 模型也会被删）",
    ),
    predictions: bool = typer.Option(
        False, "--predictions", help="清空 va_lgb_predictions 评分审计表"
    ),
    checkpoints: bool = typer.Option(
        False,
        "--checkpoints",
        help=(
            "清空 checkpoints/ 目录下所有 Phase-1 续训 shard"
            "（典型于上次训练崩在抓数阶段、想从头再来时）"
        ),
    ),
    all_flag: bool = typer.Option(
        False,
        "--all",
        help="等价于 --datasets --models --predictions --checkpoints",
    ),
) -> None:
    """清空 LightGBM 训练数据 / 模型 / 审计行 / checkpoint（destructive）。

    与 ``lgb prune`` 的区别：``prune`` 是日常维护（按 --keep N 保留最近若干 +
    active），``purge`` 是按范围彻底清空——常用于"重训前重置"或"磁盘回收"。

    至少指定一个范围标志（--datasets / --models / --predictions /
    --checkpoints / --all）；交互模式下会列出将要删除的数量并要求确认。
    """
    if all_flag:
        datasets = True
        models = True
        predictions = True
        checkpoints = True
    if not (datasets or models or predictions or checkpoints):
        typer.echo(
            "✘ 必须指定 --datasets / --models / --predictions / --checkpoints / --all 之一"
        )
        raise typer.Exit(2)

    db = Database(paths.db_path())
    try:
        preview = lgb_cleanup.count_artifacts(db)
        typer.echo("将清空以下范围：")
        if models:
            typer.echo(
                f"  • models/ 文件: {preview.n_model_files} model + "
                f"{preview.n_meta_files} meta"
                + ("  + latest.txt" if preview.latest_pointer_removed else "")
            )
            typer.echo(f"  • va_lgb_models 行: {preview.n_model_rows}")
        if datasets:
            typer.echo(f"  • datasets/*.parquet: {preview.n_dataset_files}")
        if predictions:
            typer.echo(f"  • va_lgb_predictions 行: {preview.n_prediction_rows}")
        if checkpoints:
            typer.echo(
                f"  • checkpoints/: {preview.n_checkpoint_dirs} dir, "
                f"{preview.n_checkpoint_shards} shard"
            )
        total_files = (
            (preview.n_model_files + preview.n_meta_files if models else 0)
            + (preview.n_dataset_files if datasets else 0)
            + (preview.n_checkpoint_shards if checkpoints else 0)
            + (1 if models and preview.latest_pointer_removed else 0)
        )
        total_rows = (
            (preview.n_model_rows if models else 0)
            + (preview.n_prediction_rows if predictions else 0)
        )
        if (
            total_files == 0
            and total_rows == 0
            and not (checkpoints and preview.n_checkpoint_dirs > 0)
        ):
            typer.echo("(nothing to clean — 已是干净状态)")
            return
        if not yes:
            confirm = typer.confirm("确认清空？此操作不可恢复")
            if not confirm:
                typer.echo("✘ 已取消")
                raise typer.Exit(1)
        report = lgb_cleanup.purge_lgb_artifacts(
            db,
            datasets=datasets,
            models=models,
            predictions=predictions,
            checkpoints=checkpoints,
        )
        typer.echo(
            f"✔ 已清空：files={report.total_files_removed}  "
            f"rows={report.n_model_rows + report.n_prediction_rows}"
        )
        if report.errors:
            typer.echo(f"⚠ {len(report.errors)} 个非致命错误：")
            for e in report.errors[:10]:
                typer.echo(f"  · {e}")
            if len(report.errors) > 10:
                typer.echo(f"  · …({len(report.errors) - 10} more)")
        if models:
            typer.echo(
                "提示：active 模型已删，lgb_score 现在会显示为 None；"
                "运行 `lgb train` 重新训练。"
            )
    finally:
        db.close()


def _emit_under_threshold_diagnostics(
    db: Database,
    *,
    start: str,
    end: str,
    label_source: str,
    n_labeled: int,
    min_samples: int,
) -> None:
    """Print branch-specific guidance when `lgb train` falls under the sample
    threshold. Four states are distinguished so the user never gets sent into
    a self-defeating loop (the v0.8.2 message recommended `evaluate` even when
    rows were already there but all 'pending' — re-running evaluate cannot
    flip pending→complete unless real T+N trade days have elapsed)."""
    n_anomaly_dates = db.fetchone(
        "SELECT COUNT(DISTINCT trade_date) FROM va_anomaly_history "
        "WHERE trade_date BETWEEN ? AND ?",
        (start, end),
    )[0]
    rr_row = db.fetchone(
        f"SELECT COUNT(*), "  # noqa: S608  -- label_source whitelisted upstream
        f"  SUM(CASE WHEN data_status = 'pending' THEN 1 ELSE 0 END), "
        f"  SUM(CASE WHEN data_status IN ('complete','partial') "
        f"           AND {label_source} IS NOT NULL THEN 1 ELSE 0 END), "
        f"  MAX(anomaly_date) "
        f"FROM va_realized_returns WHERE anomaly_date BETWEEN ? AND ?",
        (start, end),
    )
    n_rr_total = int(rr_row[0] or 0)
    n_rr_pending = int(rr_row[1] or 0)
    n_rr_usable = int(rr_row[2] or 0)
    max_anomaly_date = rr_row[3]
    # status='complete'/'partial' but label column is NULL → data gap
    # (suspension/delisting) or wrong label_source for what evaluate filled.
    n_rr_gap = max(n_rr_total - n_rr_pending - n_rr_usable, 0)

    typer.echo(
        f"✘ labeled samples = {n_labeled} < "
        f"lgb_train_min_samples={min_samples}"
    )
    typer.echo(
        f"  诊断: anomaly_dates={n_anomaly_dates}, "
        f"realized_rows={n_rr_total} (其中可用={n_rr_usable})"
    )

    if n_anomaly_dates == 0:
        typer.echo(
            "  → 窗口内 va_anomaly_history 无异动记录。先用 "
            "`deeptrade volume-anomaly screen` / `analyze` 累积样本。"
        )
        return

    if n_rr_total == 0:
        typer.echo(
            "  → va_realized_returns 在窗口内无任何行。运行策略层回填："
        )
        typer.echo(
            "     deeptrade volume-anomaly evaluate "
            "--backfill-all --force-recompute"
        )
        typer.echo(
            "    （注意：是 `volume-anomaly evaluate`，不是 `lgb evaluate`；"
            "后者是模型 AUC 评估。）"
        )
        return

    if n_rr_pending > 0:
        # B2 — T+N not yet elapsed; re-running evaluate cannot help. Avoid
        # the loop the v0.8.2 message put users into.
        max_str = str(max_anomaly_date) if max_anomaly_date else "<未知>"
        typer.echo(
            f"  → 有 {n_rr_pending}/{n_rr_total} 行 data_status='pending' "
            f"(窗口内最近异动日={max_str}, T+10 需 ~14 自然日才能就绪)。"
            "再跑 evaluate 不会让 pending 翻成 complete。"
        )
        typer.echo(
            f"    建议: (a) 当前窗口只有 {n_anomaly_dates} 个异动日, "
            f"与 min_samples={min_samples} 差距大 —— 用 `screen` 累积更多"
            "历史异动日 (扩 --start 向更早回溯, 在那些日期跑 screen);"
        )
        typer.echo(
            "    (b) 等 T+10 全部就绪后再次 evaluate; "
            "(c) 临时改用 `--label-source ret_t3` 缩短等待窗口 "
            "(语义不同, 仅当你能接受)。"
        )
        return

    if n_rr_usable == 0:
        typer.echo(
            f"  → 全部 {n_rr_gap} 行 status 已完成但 {label_source} 为 NULL。"
            "可能数据缺口 (停牌/退市) 或 label_source 与 evaluate 落库的列不"
            "匹配; 尝试 `--label-source max_ret_5d` (默认)。"
        )
        return

    typer.echo(
        "  → 实现收益已就绪但样本数仍少于阈值; 扩大 --start..--end 区间 "
        "或运行更多 `screen` 累积异动日。"
    )


@lgb_app.command("train")
def cmd_lgb_train(
    start: str = typer.Option(..., "--start", help="训练窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="训练窗口结束日期 YYYYMMDD"),
    label_threshold: float | None = typer.Option(
        None, "--label-threshold", help="覆盖 VaLgbConfig.lgb_label_threshold_pct"
    ),
    label_source: str | None = typer.Option(
        None, "--label-source", help="覆盖 VaLgbConfig.lgb_label_source"
    ),
    folds: int = typer.Option(5, "--folds"),
    no_activate: bool = typer.Option(False, "--no-activate"),
    fresh: bool = typer.Option(False, "--fresh"),
    keep_checkpoint: bool = typer.Option(False, "--keep-checkpoint"),
    force_sync: bool = typer.Option(False, "--force-sync"),
) -> None:
    """训练新的 LightGBM 模型并落库 (v0.7+).

    Phase-1 checkpoint resume is on by default: same (window + label config +
    schema) restarts pick up shards under
    ``~/.deeptrade/volume_anomaly/checkpoints/<digest>/``. ``--fresh`` wipes
    the dir first. On success the checkpoint is deleted unless
    ``--keep-checkpoint``.
    """
    import json  # noqa: PLC0415

    from .calendar import TradeCalendar  # noqa: PLC0415
    from .lgb import checkpoint as ckpt  # noqa: PLC0415
    from .lgb.dataset import collect_training_window  # noqa: PLC0415
    from .lgb.features import FEATURE_NAMES, SCHEMA_VERSION  # noqa: PLC0415
    from .lgb.trainer import train_lightgbm  # noqa: PLC0415
    from .runtime import build_tushare_client  # noqa: PLC0415

    if start > end:
        typer.echo("✘ --start 必须 ≤ --end")
        raise typer.Exit(2)

    db, rt = _open_runtime()
    try:
        cfg = load_config(db)
        threshold = (
            float(label_threshold)
            if label_threshold is not None
            else cfg.lgb_label_threshold_pct
        )
        source = label_source or cfg.lgb_label_source
        # Validate label config via dataclass for a single source of truth.
        VaLgbConfig(
            lgb_label_threshold_pct=threshold, lgb_label_source=source
        ).validate()

        tushare = build_tushare_client(rt)
        cal_df = tushare.call("trade_cal", force_sync=force_sync)
        calendar = TradeCalendar(cal_df)

        fingerprint = ckpt.CheckpointFingerprint(
            start_date=start,
            end_date=end,
            schema_version=SCHEMA_VERSION,
            label_threshold_pct=threshold,
            label_source=source,
            daily_lookback=250,
            moneyflow_lookback=5,
            main_board_only=True,
            baseline_index_code="000300.SH",
        )
        digest = fingerprint.digest()
        if fresh:
            ckpt.delete_checkpoint(digest)

        typer.echo(
            f"📊 拉取训练数据 {start}..{end}  "
            f"(label={source}≥{threshold:g}%)"
        )
        typer.echo(f"  📦 checkpoint digest={digest}")

        def _on_day(T: str, n: int, cum: int) -> None:
            if n < 0:
                typer.echo(f"  [{T}] (resumed)")
            else:
                typer.echo(f"  [{T}] +{n} samples (cum. {cum})")

        ds = collect_training_window(
            tushare=tushare,
            db=db,
            calendar=calendar,
            start_date=start,
            end_date=end,
            label_source=source,
            label_threshold_pct=threshold,
            force_sync=force_sync,
            checkpoint_resume=not fresh,
            plugin_version=_plugin_version(),
            on_day=_on_day,
        )
        ds = ds.filter_labeled()

        if ds.n_samples < cfg.lgb_train_min_samples:
            _emit_under_threshold_diagnostics(
                db,
                start=start,
                end=end,
                label_source=source,
                n_labeled=ds.n_samples,
                min_samples=cfg.lgb_train_min_samples,
            )
            raise typer.Exit(2)
        if ds.n_positive == 0 or ds.n_positive == ds.n_samples:
            typer.echo(
                f"✘ 标签类别退化 (n_positive={ds.n_positive}, "
                f"n_samples={ds.n_samples})，无法训练二分类"
            )
            raise typer.Exit(2)

        typer.echo(
            f"🚂 训练 (folds={folds}, n_samples={ds.n_samples}, "
            f"n_positive={ds.n_positive}) ..."
        )
        result = train_lightgbm(ds, folds=folds)
        if result.cv_auc_mean is not None and result.cv_auc_mean < 0.55:
            typer.echo(
                f"⚠ CV AUC mean = {result.cv_auc_mean:.4f} < 0.55；质量较弱"
            )

        git_commit = _git_short_commit()
        base_id = lgb_registry.mint_model_id(
            train_end_date=end,
            schema_version=SCHEMA_VERSION,
            git_commit=git_commit,
        )
        model_id = lgb_registry.ensure_unique_model_id(db, base_id)

        lgb_paths.ensure_layout()
        model_path = lgb_paths.models_dir() / lgb_paths.model_file_name(model_id)
        meta_path = lgb_paths.models_dir() / lgb_paths.meta_file_name(model_id)
        dataset_path = lgb_paths.datasets_dir() / lgb_paths.dataset_file_name(model_id)

        result.model.save_model(str(model_path))

        meta: dict[str, Any] = {
            "model_id": model_id,
            "schema_version": SCHEMA_VERSION,
            "train_window": [start, end],
            "n_samples": ds.n_samples,
            "n_positive": ds.n_positive,
            "cv_auc_mean": result.cv_auc_mean,
            "cv_auc_std": result.cv_auc_std,
            "cv_logloss_mean": result.cv_logloss_mean,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "feature_importance_top20": result.feature_importance,
            "hyperparams": result.hyperparams,
            "label_threshold_pct": threshold,
            "label_source": source,
            "git_commit": git_commit,
            "plugin_version": _plugin_version(),
            "framework_version": _framework_version(),
            "lightgbm_version": _lightgbm_version(),
            "anomaly_dates_count": len(ds.anomaly_dates),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            import pandas as pd  # noqa: PLC0415

            full = pd.concat(
                [
                    ds.feature_matrix.reset_index(drop=True),
                    ds.labels.reset_index(drop=True).rename("label"),
                    ds.sample_index.reset_index(drop=True),
                ],
                axis=1,
            )
            full.to_parquet(dataset_path, index=False)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"⚠ dataset.parquet 写入失败: {e}")
            dataset_path = None  # type: ignore[assignment]

        rel_path = model_path.relative_to(lgb_paths.plugin_data_dir())
        record = lgb_registry.ModelRecord(
            model_id=model_id,
            schema_version=SCHEMA_VERSION,
            train_start_date=start,
            train_end_date=end,
            n_samples=ds.n_samples,
            n_positive=ds.n_positive,
            cv_auc_mean=result.cv_auc_mean,
            cv_auc_std=result.cv_auc_std,
            cv_logloss_mean=result.cv_logloss_mean,
            feature_count=len(FEATURE_NAMES),
            feature_list_json=json.dumps(FEATURE_NAMES),
            hyperparams_json=json.dumps(result.hyperparams),
            label_threshold_pct=threshold,
            label_source=source,
            framework_version=_framework_version(),
            plugin_version=_plugin_version(),
            git_commit=git_commit,
            file_path=str(rel_path).replace("\\", "/"),
        )
        lgb_registry.insert_model(db, record, activate=not no_activate)
        if not no_activate:
            lgb_paths.latest_pointer().write_text(
                str(rel_path).replace("\\", "/"), encoding="utf-8"
            )

        typer.echo(f"✔ Saved model: {model_id}")
        if result.cv_auc_mean is not None:
            typer.echo(
                f"  CV AUC = {result.cv_auc_mean:.4f} "
                f"± {(result.cv_auc_std or 0):.4f}"
            )
        if result.cv_logloss_mean is not None:
            typer.echo(f"  CV logloss = {result.cv_logloss_mean:.4f}")
        typer.echo("  Top-10 feature importance:")
        for name, score in result.top_features(10):
            typer.echo(f"    {name:<40} {score:.0f}")
        typer.echo(f"  file:    {model_path}")
        typer.echo(f"  meta:    {meta_path}")
        if dataset_path is not None:
            typer.echo(f"  dataset: {dataset_path}")
        typer.echo(f"  active:  {not no_activate}")
        if not no_activate:
            typer.echo(
                "\n👉 下一步：`deeptrade volume-anomaly analyze` "
                "（跑一轮策略以触发 LGB 评分；分数会落 va_lgb_predictions，"
                "之后可用 `lgb info --recent-N 5` 查看分布）"
            )
        else:
            typer.echo(
                f"\n👉 模型未激活；用 `lgb activate {model_id}` 切换为 active 后再 analyze。"
            )

        if not keep_checkpoint:
            ckpt.delete_checkpoint(digest)
    finally:
        db.close()


@lgb_app.command("evaluate")
def cmd_lgb_evaluate(
    start: str = typer.Option(..., "--start", help="评估窗口起始 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="评估窗口结束 YYYYMMDD"),
    model_id: str | None = typer.Option(
        None, "--model-id", help="目标模型 ID；不传则评估 active 模型"
    ),
    k: str = typer.Option(
        "5,10,20", "--k", help="逗号分隔的 Top-K 列表（默认 5,10,20）"
    ),
    force_sync: bool = typer.Option(False, "--force-sync"),
    drift: bool = typer.Option(
        False,
        "--drift",
        help="同时输出特征漂移报表（PSI vs baseline 的训练矩阵快照）",
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", help="drift baseline model_id；默认 = active 或 --model-id"
    ),
) -> None:
    """对指定窗口跑离线评估（AUC / logloss / Top-K 命中率 vs baseline）。

    模型在不指定 ``--model-id`` 时使用 active 行；``label_source`` /
    ``label_threshold_pct`` 取自 ``va_lgb_models`` 行（与训练口径一致）。
    JSON 报告落到 ``reports/lgb_evaluate_<model_id>_<window>.json``。

    ``--drift`` 附带 PSI 漂移检测：用同窗口的当前特征矩阵对 baseline 模型
    的 ``dataset.parquet`` 快照做 10-bin PSI，按 PSI 降序列出 top features。
    """
    import json  # noqa: PLC0415

    from deeptrade.core import paths as core_paths  # noqa: PLC0415

    from .calendar import TradeCalendar  # noqa: PLC0415
    from .lgb.dataset import collect_training_window  # noqa: PLC0415
    from .lgb.evaluate import (  # noqa: PLC0415
        compute_drift,
        evaluate_model,
        format_drift_table,
        format_evaluate_table,
        load_baseline_feature_matrix,
    )
    from .runtime import build_tushare_client  # noqa: PLC0415

    if start > end:
        typer.echo("✘ --start 必须 ≤ --end")
        raise typer.Exit(2)
    try:
        ks = tuple(int(x.strip()) for x in k.split(",") if x.strip())
    except ValueError as e:
        typer.echo(f"✘ --k 解析失败: {e}")
        raise typer.Exit(2) from e
    if not ks or any(kk < 1 for kk in ks):
        typer.echo(f"✘ --k 必须是 ≥1 的整数列表: {k!r}")
        raise typer.Exit(2)

    db, rt = _open_runtime()
    try:
        # Resolve the target model (default: active).
        target_id = model_id
        if target_id is None:
            active = _safe_get_active(db)
            if active is None:
                typer.echo(
                    "✘ no active model — 用 --model-id 指定，或先运行 `lgb train`。"
                )
                typer.echo(
                    "  注意：本命令是「模型 AUC/Top-K 评估」；"
                    "如果你想回填 T+N 实现收益（训练标签来源），"
                    "请用 `deeptrade volume-anomaly evaluate`。"
                )
                raise typer.Exit(2)
            target_id = active.model_id
        record = _safe_get_model(db, target_id)
        if record is None:
            typer.echo(f"✘ model_id not found: {target_id!r}")
            raise typer.Exit(2)

        tushare = build_tushare_client(rt)
        cal_df = tushare.call("trade_cal", force_sync=force_sync)
        calendar = TradeCalendar(cal_df)

        typer.echo(
            f"📊 评估 model={target_id} · window {start}..{end} · "
            f"label={record.label_source}≥{record.label_threshold_pct:g}%"
        )

        def _on_day(T: str, n: int, cum: int) -> None:
            if n < 0:
                typer.echo(f"  [{T}] (resumed)")
            else:
                typer.echo(f"  [{T}] +{n} samples (cum. {cum})")

        result = evaluate_model(
            tushare=tushare,
            calendar=calendar,
            db=db,
            start_date=start,
            end_date=end,
            model_id=target_id,
            k_values=ks,
            label_source=record.label_source,
            label_threshold_pct=record.label_threshold_pct,
            force_sync=force_sync,
            on_day=_on_day,
        )

        typer.echo("")
        typer.echo(format_evaluate_table(result))

        reports_root = core_paths.reports_dir()
        reports_root.mkdir(parents=True, exist_ok=True)
        out_path = (
            reports_root
            / f"lgb_evaluate_{result.model_id or 'none'}_{start}_{end}.json"
        )
        out_path.write_text(
            json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"\n📄 JSON 报告：{out_path}")

        if drift:
            baseline_id = baseline or target_id
            baseline_record = _safe_get_model(db, baseline_id)
            if baseline_record is None:
                typer.echo(f"⚠ drift baseline model_id 未找到: {baseline_id!r}")
            else:
                parquet_path = (
                    lgb_paths.datasets_dir()
                    / lgb_paths.dataset_file_name(baseline_id)
                )
                base_df = load_baseline_feature_matrix(parquet_path)
                if base_df is None:
                    typer.echo(
                        f"⚠ baseline dataset snapshot 缺失或不可读：{parquet_path}"
                    )
                else:
                    typer.echo(
                        f"\n🔬 计算 drift · baseline={baseline_id} → "
                        f"current {start}..{end} (PSI, 10 buckets)"
                    )
                    cur_ds = collect_training_window(
                        tushare=tushare,
                        db=db,
                        calendar=calendar,
                        start_date=start,
                        end_date=end,
                        label_source=record.label_source,
                        label_threshold_pct=record.label_threshold_pct,
                        force_sync=force_sync,
                    )
                    drift_result = compute_drift(
                        baseline_feature_matrix=base_df,
                        current_feature_matrix=cur_ds.feature_matrix,
                        baseline_model_id=baseline_id,
                        window_start=start,
                        window_end=end,
                    )
                    typer.echo("")
                    typer.echo(format_drift_table(drift_result))
                    drift_path = (
                        reports_root
                        / f"lgb_drift_{baseline_id}_{start}_{end}.json"
                    )
                    drift_path.write_text(
                        json.dumps(
                            drift_result.to_json_dict(),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    typer.echo(f"\n📄 drift JSON：{drift_path}")
                    typer.echo(
                        "提示：PSI>0.25 视为显著漂移；若同时 AUC 较训练阶段下降 "
                        ">5pt，建议扩窗 / 重训（lightgbm_iteration_plan §4.1 PR-3.3）。"
                    )
    finally:
        db.close()


@lgb_app.command("refresh-features")
def cmd_lgb_refresh_features(
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
) -> None:
    """仅拉历史数据 / 不训练（PR-3.x 周末预热工具，后续 PR 实现）。"""
    typer.echo(
        f"Not yet implemented in this iteration: lgb refresh-features "
        f"--start={start} --end={end}"
    )
    typer.echo(_LGB_PENDING_HINT)
    raise typer.Exit(2)


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


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
        sys.stderr.write(render_exception(e) + "\n")
        return 1


# Re-exported for tests that want to construct a VaLgbConfig without going
# through CLI plumbing.
__all__ = [
    "VaLgbConfig",
    "app",
    "load_config",
    "save_config",
    "main",
]
