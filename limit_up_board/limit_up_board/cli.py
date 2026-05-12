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

import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import questionary
import typer
from rich.console import Console
from rich.table import Table

from deeptrade.core import paths
from deeptrade.core.config import ConfigService
from deeptrade.core.db import Database
from deeptrade.core.llm_manager import LLMManager
from deeptrade.plugins_api import render_exception

from .calendar import TradeCalendar
from .config import LubConfig, list_for_show, load_config, save_config
from .lgb import checkpoint as lgb_checkpoint
from .lgb import paths as lgb_paths
from .lgb import registry as lgb_registry
from .lgb.checkpoint import CheckpointFingerprint, CheckpointMismatch
from .lgb.dataset import _enumerate_trade_dates, collect_training_window
from .lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from .lgb.registry import ModelRecord, ensure_unique_model_id, insert_model, mint_model_id
from .lgb.trainer import train_lightgbm
from .runner import LubRunner, PreconditionError, RunParams, render_finished_run
from .runtime import LubRuntime, build_tushare_client

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
        new_min_mv = _prompt_positive_float("流通市值下限（亿）", cur.min_float_mv_yi)
        new_mv = _prompt_positive_float("流通市值上限（亿）", cur.max_float_mv_yi)
        new_close = _prompt_positive_float("当前股价上限（元）", cur.max_close_yuan)
        if new_min_mv >= new_mv:
            typer.echo(
                f"✘ 流通市值下限（{new_min_mv}亿）必须小于上限（{new_mv}亿）"
            )
            raise typer.Exit(2)
        # 用 dataclasses.replace 保留所有未交互的字段（如 v0.5 的 lgb_* 配置）。
        new_cfg = replace(
            cur,
            min_float_mv_yi=new_min_mv,
            max_float_mv_yi=new_mv,
            max_close_yuan=new_close,
        )
        save_config(db, new_cfg)
        typer.echo(
            f"✔ Saved: {new_cfg.min_float_mv_yi}亿 < 流通市值 < "
            f"{new_cfg.max_float_mv_yi}亿、股价 < {new_cfg.max_close_yuan}元"
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
# lgb — LightGBM 评分模型生命周期
# ---------------------------------------------------------------------------


_LGB_PENDING_HINT = "（实现将在后续 PR 中补齐）"


def _safe_list_models(db: Database) -> list[ModelRecord]:
    """Wrap registry.list_models with a safety net for missing-table cases."""
    try:
        return lgb_registry.list_models(db)
    except Exception:  # noqa: BLE001 — duckdb CatalogException 不公开导出
        return []


def _safe_get_model(db: Database, model_id: str) -> ModelRecord | None:
    try:
        return lgb_registry.get_model(db, model_id)
    except Exception:  # noqa: BLE001
        return None


def _safe_get_active(db: Database) -> ModelRecord | None:
    try:
        return lgb_registry.get_active(db)
    except Exception:  # noqa: BLE001
        return None


def _safe_predictions_usage(db: Database, model_id: str) -> dict[str, int] | None:
    """Aggregate counts from ``lub_lgb_predictions`` for the given model.

    Returns ``{'n_rows', 'n_runs', 'n_trade_dates'}``; ``None`` 表示表本身
    不可访问（迁移未应用 / 旧 DB），让调用方静默降级。
    """
    try:
        row = db.fetchone(
            "SELECT COUNT(*), COUNT(DISTINCT run_id), COUNT(DISTINCT trade_date) "
            "FROM lub_lgb_predictions WHERE model_id = ?",
            (model_id,),
        )
    except Exception:  # noqa: BLE001 — table may not exist on legacy DBs
        return None
    if row is None:
        return {"n_rows": 0, "n_runs": 0, "n_trade_dates": 0}
    return {
        "n_rows": int(row[0] or 0),
        "n_runs": int(row[1] or 0),
        "n_trade_dates": int(row[2] or 0),
    }


def _safe_predictions_recent(
    db: Database, model_id: str, recent_n: int
) -> list[dict[str, Any]]:
    """Per-day score distribution snapshots for the most recent ``recent_n`` days.

    Returns rows ordered by trade_date desc; each row has ``trade_date``, ``n``,
    ``min``, ``p25``, ``median``, ``p75``, ``max``.

    Implementation note: DuckDB exposes ``approx_quantile`` / ``percentile_cont``;
    we use ``quantile_cont`` which is supported in current DuckDB releases.
    """
    if recent_n <= 0:
        return []
    try:
        rows = db.fetchall(
            "SELECT trade_date, COUNT(*) AS n, MIN(lgb_score) AS lo, "
            "quantile_cont(lgb_score, 0.25) AS q25, "
            "quantile_cont(lgb_score, 0.5)  AS med, "
            "quantile_cont(lgb_score, 0.75) AS q75, "
            "MAX(lgb_score) AS hi "
            "FROM lub_lgb_predictions WHERE model_id = ? "
            "GROUP BY trade_date ORDER BY trade_date DESC LIMIT ?",
            (model_id, recent_n),
        )
    except Exception:  # noqa: BLE001 — table missing or function unavailable
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "trade_date": str(r[0]),
                "n": int(r[1] or 0),
                "min": float(r[2] or 0.0),
                "p25": float(r[3] or 0.0),
                "median": float(r[4] or 0.0),
                "p75": float(r[5] or 0.0),
                "max": float(r[6] or 0.0),
            }
        )
    return out


def _git_short_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip() or None
    except Exception:  # noqa: BLE001
        return None


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
    for r in records:
        auc_repr = (
            f"{r.cv_auc_mean:.3f}±{(r.cv_auc_std or 0):.3f}"
            if r.cv_auc_mean is not None
            else "—"
        )
        table.add_row(
            "★" if r.is_active else "",
            r.model_id,
            str(r.schema_version),
            f"{r.train_start_date}..{r.train_end_date}",
            str(r.n_samples),
            str(r.n_positive),
            auc_repr,
            str(r.feature_count),
            r.plugin_version,
            str(r.created_at) if r.created_at is not None else "—",
        )
    console.print(table)


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
            "（数据来源：lub_lgb_predictions；由 `run` 命令在线上推理时写入，"
            "不含训练 / evaluate 的 in-sample 分数。刚训练完未跑过 run 时为空属正常现象）"
        ),
    ),
) -> None:
    """展示单个模型的详细信息（CV 指标 / 特征数 / 超参 / 文件路径 + 使用统计）。"""
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
    # PR-3.2 — usage stats from lub_lgb_predictions.
    if usage is not None:
        table.add_row(
            "predictions usage",
            (
                f"runs={usage['n_runs']}  trade_dates={usage['n_trade_dates']}  "
                f"rows={usage['n_rows']}"
                if usage["n_rows"]
                else (
                    "(尚无推理记录 — 此表由 `deeptrade limit-up-board run` 在线上推理时"
                    "写入；跑一次 run 后即可看到当日 candidate 的 LGB 分数分布)"
                )
            ),
        )
    console.print(table)

    if recent_n > 0:
        recent_table = Table(
            title=(
                f"recent {recent_n} trade dates · run-time score distribution "
                f"(from lub_lgb_predictions)"
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
                    f"{row['min'] * 100:.1f}",
                    f"{row['p25'] * 100:.1f}",
                    f"{row['median'] * 100:.1f}",
                    f"{row['p75'] * 100:.1f}",
                    f"{row['max'] * 100:.1f}",
                )
        console.print(recent_table)
        if not recent:
            typer.echo(
                "提示：此表数据由 `deeptrade limit-up-board run` 在线上推理时写入；"
                "刚训练完未跑过 run 时为空属正常现象。"
            )


@lgb_app.command("train")
def cmd_lgb_train(
    start: str = typer.Option(..., "--start", help="训练窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="训练窗口结束日期 YYYYMMDD"),
    folds: int = typer.Option(5, "--folds", help="GroupKFold 折数（≥2 才做 CV）"),
    no_activate: bool = typer.Option(
        False, "--no-activate", help="训练完成后不切换 active 模型"
    ),
    force_sync: bool = typer.Option(False, "--force-sync", help="强制刷新 tushare 缓存"),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="忽略已有 checkpoint，重新抓取所有交易日（默认遇到同窗口/同配置的"
        " checkpoint 会自动续训）",
    ),
) -> None:
    """训练新的 LightGBM 模型并落库。

    默认开启 Phase-1 checkpoint：按日把训练样本落到
    ``~/.deeptrade/limit_up_board/checkpoints/<digest>/days/<YYYYMMDD>.parquet``，
    崩溃/中断后再次以同窗口 + 同筛选参数运行将自动跳过已完成日；训练成功后
    checkpoint 自动清理。用 ``--fresh`` 丢弃现有 shard 重新抓取。
    """
    if start > end:
        typer.echo("✘ --start 必须 ≤ --end")
        raise typer.Exit(2)

    db, rt = _open_runtime()
    try:
        cfg = load_config(db)
        tushare = build_tushare_client(rt)
        cal_df = tushare.call("trade_cal", force_sync=force_sync)
        calendar = TradeCalendar(cal_df)

        # ---- checkpoint setup ----
        fingerprint = CheckpointFingerprint(
            start_date=start,
            end_date=end,
            schema_version=SCHEMA_VERSION,
            label_threshold_pct=cfg.lgb_label_threshold_pct,
            daily_lookback=30,
            moneyflow_lookback=5,
            min_float_mv_yi=cfg.min_float_mv_yi,
            max_float_mv_yi=cfg.max_float_mv_yi,
            max_close_yuan=cfg.max_close_yuan,
        )
        digest = fingerprint.digest()
        if fresh:
            lgb_checkpoint.delete_checkpoint(digest)
        try:
            state = lgb_checkpoint.open_or_create(
                fingerprint, plugin_version=_plugin_version()
            )
        except CheckpointMismatch as e:
            typer.echo(f"✘ checkpoint 状态损坏：{e}")
            raise typer.Exit(2) from e
        already_done = lgb_checkpoint.completed_dates(digest)
        # state 与磁盘自我修复：磁盘有但 state 没记的 → 补进 state
        if already_done - set(state.completed_dates):
            state.completed_dates = sorted(
                set(state.completed_dates) | already_done
            )
            lgb_checkpoint.save_state(state)

        typer.echo(
            f"📊 拉取训练数据 {start}..{end}  "
            f"({cfg.min_float_mv_yi}亿<float_mv<{cfg.max_float_mv_yi}亿, "
            f"max_close<{cfg.max_close_yuan}元, "
            f"label_threshold={cfg.lgb_label_threshold_pct}%)"
        )
        if already_done:
            typer.echo(
                f"  ↻ 续训：已落盘 {len(already_done)} 天 shard (digest={digest})，"
                f"本次仅补漏 (--fresh 可重抓)"
            )
        else:
            typer.echo(f"  📦 checkpoint digest={digest}")

        def _on_day(T: str, n: int, cum: int) -> None:
            if n < 0:
                typer.echo(f"  [{T}] (resumed)")
            else:
                typer.echo(f"  [{T}] +{n} samples (cum. {cum})")

        def _sink(T: str, shard_df: Any) -> None:
            lgb_checkpoint.save_day_shard(digest, T, shard_df)
            lgb_checkpoint.record_day_done(digest, T)

        collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date=start,
            end_date=end,
            max_float_mv_yi=cfg.max_float_mv_yi,
            max_close_yuan=cfg.max_close_yuan,
            min_float_mv_yi=cfg.min_float_mv_yi,
            label_threshold_pct=cfg.lgb_label_threshold_pct,
            force_sync=force_sync,
            on_day=_on_day,
            skip_dates=already_done,
            day_sink=_sink,
        )
        full_trade_dates = _enumerate_trade_dates(calendar, start, end)
        ds = lgb_checkpoint.assemble_full_dataset(
            digest,
            label_threshold_pct=cfg.lgb_label_threshold_pct,
            daily_lookback=30,
            moneyflow_lookback=5,
            trade_dates=full_trade_dates,
        )
        ds = ds.filter_labeled()
        if ds.n_samples < cfg.lgb_train_min_samples:
            typer.echo(
                f"✘ labeled samples = {ds.n_samples} < lgb_train_min_samples="
                f"{cfg.lgb_train_min_samples}（窗口太短或缺 T+1 数据）"
            )
            raise typer.Exit(2)
        if ds.n_positive == 0 or ds.n_positive == ds.n_samples:
            typer.echo(
                f"✘ 标签类别退化（n_positive={ds.n_positive}, n_samples={ds.n_samples}），"
                "无法训练二分类"
            )
            raise typer.Exit(2)

        typer.echo(
            f"🚂 训练（folds={folds}, n_samples={ds.n_samples}, "
            f"n_positive={ds.n_positive}）..."
        )
        result = train_lightgbm(ds, folds=folds)
        if result.cv_auc_mean is not None and result.cv_auc_mean < 0.55:
            typer.echo(f"⚠ CV AUC mean = {result.cv_auc_mean:.4f} < 0.55；质量较弱")

        # ---- mint + paths ----
        git_commit = _git_short_commit()
        base_id = mint_model_id(
            train_end_date=end, schema_version=SCHEMA_VERSION, git_commit=git_commit
        )
        model_id = ensure_unique_model_id(db, base_id)

        lgb_paths.ensure_layout()
        model_path = lgb_paths.models_dir() / lgb_paths.model_file_name(model_id)
        meta_path = lgb_paths.models_dir() / lgb_paths.meta_file_name(model_id)
        dataset_path = lgb_paths.datasets_dir() / lgb_paths.dataset_file_name(model_id)

        # ---- save model + meta + dataset snapshot ----
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
            "label_threshold_pct": cfg.lgb_label_threshold_pct,
            "git_commit": git_commit,
            "plugin_version": _plugin_version(),
            "framework_version": _framework_version(),
            "lightgbm_version": _lightgbm_version(),
            "trade_dates_count": len(ds.trade_dates),
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
        except Exception as e:  # noqa: BLE001 — parquet 缺 pyarrow/fastparquet 时降级
            logging.warning("failed to write dataset parquet snapshot: %s", e)
            dataset_path = None  # type: ignore[assignment]

        # ---- registry row ----
        rel_path = model_path.relative_to(lgb_paths.plugin_data_dir())
        record = ModelRecord(
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
            framework_version=_framework_version(),
            plugin_version=_plugin_version(),
            git_commit=git_commit,
            file_path=str(rel_path).replace("\\", "/"),
        )
        insert_model(db, record, activate=not no_activate)
        if not no_activate:
            lgb_paths.latest_pointer().write_text(
                str(rel_path).replace("\\", "/"), encoding="utf-8"
            )

        # ---- summary ----
        typer.echo(f"✔ Saved model: {model_id}")
        if result.cv_auc_mean is not None:
            typer.echo(
                f"  CV AUC = {result.cv_auc_mean:.4f} ± {(result.cv_auc_std or 0):.4f}"
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
                "\n👉 下一步：`deeptrade limit-up-board run` "
                "（跑一轮策略以触发 LGB 评分；分数会落 lub_lgb_predictions，"
                "之后可用 `lgb info --recent-N 5` 查看分布）"
            )
        else:
            typer.echo(
                "\n👉 模型未激活；用 `lgb activate "
                f"{model_id}` 切换为 active 后再跑 `run` 才会用上它。"
            )

        # 训练全程成功 → 清理 checkpoint 目录（失败时 shard 留盘，下次自动续）
        lgb_checkpoint.delete_checkpoint(digest)
    finally:
        db.close()


@lgb_app.command("evaluate")
def cmd_lgb_evaluate(
    start: str = typer.Option(..., "--start", help="评估窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="评估窗口结束日期 YYYYMMDD"),
    model_id: str | None = typer.Option(
        None, "--model-id", help="模型 ID；不指定则评估当前 active 模型"
    ),
    force_sync: bool = typer.Option(False, "--force-sync", help="强制刷新 tushare 缓存"),
    k_values: str = typer.Option(
        "5,10,20",
        "--k",
        help="逗号分隔的 Top-K 列表；默认 5,10,20",
    ),
    drift: bool = typer.Option(
        False, "--drift", help="附带输出特征 drift 报表（PSI vs 训练数据快照）"
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", help="drift baseline 模型 ID（默认 = active 或 --model-id）"
    ),
) -> None:
    """对指定窗口跑离线评估（AUC / logloss / Top-K 命中率 vs baseline）。"""
    if start > end:
        typer.echo("✘ --start 必须 ≤ --end")
        raise typer.Exit(2)
    try:
        ks = tuple(int(x.strip()) for x in k_values.split(",") if x.strip())
    except ValueError as e:
        typer.echo(f"✘ --k 解析失败: {e}")
        raise typer.Exit(2) from e
    if not ks or any(k < 1 for k in ks):
        typer.echo(f"✘ --k 必须是 ≥1 的整数列表: {k_values!r}")
        raise typer.Exit(2)

    from .lgb.evaluate import (  # noqa: PLC0415
        compute_drift,
        evaluate_model,
        format_drift_table,
        format_evaluate_table,
        load_baseline_feature_matrix,
    )
    from .lgb.dataset import collect_training_window as _collect_window  # noqa: PLC0415

    db, rt = _open_runtime()
    try:
        cfg = load_config(db)
        tushare = build_tushare_client(rt)
        cal_df = tushare.call("trade_cal", force_sync=force_sync)
        calendar = TradeCalendar(cal_df)

        # Pre-resolve the model so we can name the JSON output file even when
        # the user passed --model-id explicitly.
        target_id = model_id
        if target_id is None:
            active = _safe_get_active(db)
            if active is None:
                typer.echo("✘ no active model — 用 --model-id 指定，或先运行 lgb train")
                raise typer.Exit(2)
            target_id = active.model_id
        else:
            if _safe_get_model(db, target_id) is None:
                typer.echo(f"✘ model_id not found: {target_id!r}")
                raise typer.Exit(2)

        typer.echo(
            f"📊 评估 model={target_id} · window {start}..{end} · "
            f"thresh={cfg.lgb_label_threshold_pct}%"
        )

        def _on_day(T: str, n: int, cum: int) -> None:
            typer.echo(f"  [{T}] +{n} samples (cum. {cum})")

        result = evaluate_model(
            tushare=tushare,
            calendar=calendar,
            db=db,
            start_date=start,
            end_date=end,
            model_id=target_id,
            k_values=ks,
            label_threshold_pct=cfg.lgb_label_threshold_pct,
            max_float_mv_yi=cfg.max_float_mv_yi,
            max_close_yuan=cfg.max_close_yuan,
            min_float_mv_yi=cfg.min_float_mv_yi,
            force_sync=force_sync,
            on_day=_on_day,
        )

        typer.echo("")
        typer.echo(format_evaluate_table(result))

        # JSON dump alongside the existing reports/ tree.
        reports_root = paths.reports_dir()
        reports_root.mkdir(parents=True, exist_ok=True)
        out_path = reports_root / f"lgb_evaluate_{result.model_id or 'none'}_{start}_{end}.json"
        out_path.write_text(
            json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"\n📄 JSON 报告：{out_path}")

        # ---- PR-3.3: optional drift report --------------------------------
        if drift:
            baseline_id = baseline or target_id
            baseline_record = _safe_get_model(db, baseline_id)
            if baseline_record is None:
                typer.echo(f"⚠ drift baseline model_id not found: {baseline_id!r}")
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
                    cur_ds = _collect_window(
                        tushare=tushare,
                        calendar=calendar,
                        start_date=start,
                        end_date=end,
                        max_float_mv_yi=cfg.max_float_mv_yi,
                        max_close_yuan=cfg.max_close_yuan,
                        min_float_mv_yi=cfg.min_float_mv_yi,
                        label_threshold_pct=cfg.lgb_label_threshold_pct,
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
                        "提示: PSI>0.25 视为显著漂移；若同时 AUC 较训练阶段下降 >5pt，"
                        "建议增加训练窗口 / 重新训练（lightgbm_iteration_plan §4.1 PR-3.3）。"
                    )
    finally:
        db.close()


@lgb_app.command("activate")
def cmd_lgb_activate(
    model_id: str = typer.Argument(..., help="要激活的模型 ID"),
) -> None:
    """原子切换 active 模型。"""
    db = Database(paths.db_path())
    try:
        try:
            ok = lgb_registry.set_active(db, model_id)
        except Exception:  # noqa: BLE001 — 兼容 lub_lgb_models 表尚未迁移的场景
            ok = False
        if not ok:
            typer.echo(f"✘ model_id not found: {model_id!r}")
            raise typer.Exit(2)
        record = _safe_get_model(db, model_id)
        if record is not None:
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
        # 保留：最近 keep 个（无论 is_active）+ 任何 is_active 行兜底
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
                    f.unlink()
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
        help="清空 models/ 落盘文件 + lub_lgb_models 注册行（active 模型也会被删）",
    ),
    predictions: bool = typer.Option(
        False, "--predictions", help="清空 lub_lgb_predictions 评分审计表"
    ),
    checkpoints: bool = typer.Option(
        False,
        "--checkpoints",
        help="清空 checkpoints/ 目录下所有 Phase-1 续训 shard"
        "（典型于上次训练崩在抓数阶段、想从头再来时）",
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

    Examples:

      lgb purge --datasets             # 只删训练矩阵快照（最大件磁盘）
      lgb purge --checkpoints --yes    # 删所有未完成训练的 shard，跳过确认
      lgb purge --models --yes         # 删除所有模型 + 注册行，跳过确认
      lgb purge --all                  # 彻底清空（交互确认）
    """
    from .lgb.cleanup import count_artifacts, purge_lgb_artifacts  # noqa: PLC0415

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
        preview = count_artifacts(db)
        typer.echo("将清空以下范围：")
        if models:
            typer.echo(
                f"  • models/ 文件: {preview.n_model_files} model + "
                f"{preview.n_meta_files} meta"
                + ("  + latest.txt" if preview.latest_pointer_removed else "")
            )
            typer.echo(f"  • lub_lgb_models 行: {preview.n_model_rows}")
        if datasets:
            typer.echo(
                f"  • datasets/*.parquet: {preview.n_dataset_files}"
            )
        if predictions:
            typer.echo(
                f"  • lub_lgb_predictions 行: {preview.n_prediction_rows}"
            )
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
            confirm = questionary.confirm(
                "确认清空？此操作不可恢复",
                default=False,
            ).ask()
            if not confirm:
                typer.echo("✘ 已取消")
                raise typer.Exit(1)

        report = purge_lgb_artifacts(
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


@lgb_app.command("refresh-features")
def cmd_lgb_refresh_features(
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
) -> None:
    """仅拉历史数据 / 不训练（PR-3.x 实现）。"""
    typer.echo(
        f"Not yet implemented in this iteration: lgb refresh-features "
        f"--start={start} --end={end}"
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
        # DEEPTRADE_DEBUG=1 makes render_exception emit the full traceback;
        # otherwise it returns "✘ {ExcType}: {msg}".
        sys.stderr.write(render_exception(e) + "\n")
        return 1
