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

from .calendar import TradeCalendar
from .config import LubConfig, list_for_show, load_config, save_config
from .lgb import paths as lgb_paths
from .lgb import registry as lgb_registry
from .lgb.dataset import collect_training_window
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
) -> None:
    """展示单个模型的详细信息（CV 指标 / 特征数 / 超参 / 文件路径）。"""
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
    console.print(table)


@lgb_app.command("train")
def cmd_lgb_train(
    start: str = typer.Option(..., "--start", help="训练窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="训练窗口结束日期 YYYYMMDD"),
    folds: int = typer.Option(5, "--folds", help="GroupKFold 折数（≥2 才做 CV）"),
    no_activate: bool = typer.Option(
        False, "--no-activate", help="训练完成后不切换 active 模型"
    ),
    force_sync: bool = typer.Option(False, "--force-sync", help="强制刷新 tushare 缓存"),
) -> None:
    """训练新的 LightGBM 模型并落库。"""
    if start > end:
        typer.echo("✘ --start 必须 ≤ --end")
        raise typer.Exit(2)

    db, rt = _open_runtime()
    try:
        cfg = load_config(db)
        tushare = build_tushare_client(rt)
        cal_df = tushare.call("trade_cal", force_sync=force_sync)
        calendar = TradeCalendar(cal_df)

        typer.echo(
            f"📊 拉取训练数据 {start}..{end}  "
            f"(max_float_mv<{cfg.max_float_mv_yi}亿, max_close<{cfg.max_close_yuan}元, "
            f"label_threshold={cfg.lgb_label_threshold_pct}%)"
        )

        def _on_day(T: str, n: int, cum: int) -> None:
            typer.echo(f"  [{T}] +{n} samples (cum. {cum})")

        ds = collect_training_window(
            tushare=tushare,
            calendar=calendar,
            start_date=start,
            end_date=end,
            max_float_mv_yi=cfg.max_float_mv_yi,
            max_close_yuan=cfg.max_close_yuan,
            label_threshold_pct=cfg.lgb_label_threshold_pct,
            force_sync=force_sync,
            on_day=_on_day,
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
    finally:
        db.close()


@lgb_app.command("evaluate")
def cmd_lgb_evaluate(
    start: str = typer.Option(..., "--start", help="评估窗口起始日期 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="评估窗口结束日期 YYYYMMDD"),
    model_id: str | None = typer.Option(None, "--model-id"),
    drift: bool = typer.Option(False, "--drift", help="顺便输出特征 drift 报表（PR-3.3）"),
) -> None:
    """对指定窗口跑离线评估（PR-3.1 实现）。"""
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
        sys.stderr.write(f"✘ {type(e).__name__}: {e}\n")
        return 1
