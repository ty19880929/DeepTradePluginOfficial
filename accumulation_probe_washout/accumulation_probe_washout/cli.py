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
from .runner import (
    AnalyzeParams,
    ApwRunner,
    BackfillHistoryParams,
    EvaluateParams,
    PruneParams,
    RunParams,
    ScreenParams,
)
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

lgb_app = typer.Typer(
    name="lgb",
    help="LightGBM 主升浪启动概率评分模型生命周期管理（v0.5.0+）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(lgb_app, name="lgb")


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
    backfill_history: bool = typer.Option(
        False,
        "--backfill-history",
        help=(
            "v0.3.0 — LLM-free batch replay over a date range. "
            "Writes apw_signal_history only; never touches apw_watchlist. "
            "需要 --start / --end。"
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
            "--backfill-history 模式下,对已存在 apw_signal_history 行的日期"
            "也重新筛选 (DELETE 后 INSERT)。默认跳过已有日期 (作为 resume)。"
        ),
    ),
    no_dashboard: bool = typer.Option(  # noqa: ARG001 — wired in M4
        False, "--no-dashboard",
        help="禁用动态仪表盘 (M4 起生效)。",
    ),
) -> None:
    """Apply local screening rules → write apw_signal_history + apw_watchlist (no LLM).

    With ``--backfill-history --start --end`` switches to LLM-free batch replay
    that iterates every open trade_date in ``[start, end]`` and writes hits to
    ``apw_signal_history`` only (does NOT touch ``apw_watchlist``).
    """
    from .ui import choose_renderer

    if backfill_history:
        if not (start and end):
            typer.echo("✘ --backfill-history requires both --start and --end")
            raise typer.Exit(2)
        db, rt = _open_runtime()
        try:
            params = BackfillHistoryParams(
                start=start,
                end=end,
                overwrite=overwrite,
                force_sync=force_sync,
            )
            # Multi-day loops mislead the single-day StageStack dashboard —
            # force legacy renderer regardless of TTY (matches §4.6 of the
            # migration plan).
            renderer = LegacyStreamRenderer()
            outcome = ApwRunner(rt, renderer=renderer).execute_backfill_history(params)
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
        return

    if start or end or overwrite:
        typer.echo(
            "✘ --start / --end / --overwrite are only valid with --backfill-history"
        )
        raise typer.Exit(2)

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
# stats (v0.3.0)
# ---------------------------------------------------------------------------


@app.command("stats")
def cmd_stats(
    from_date: Optional[str] = typer.Option(None, "--from", help="YYYYMMDD (inclusive)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="YYYYMMDD (inclusive)"),
    by: str = typer.Option(
        "phase",
        "--by",
        help=(
            "聚合维度：phase | prediction | main_pattern | "
            "launch_score_bin | accumulation_score_bin | "
            "probe_quality_score_bin | washout_score_bin | "
            "launch_setup_score_bin | dimension_scores | lgb_score_bin"
        ),
    ),
) -> None:
    """Read-only aggregates over apw_signal_history ⋈ apw_realized_returns."""
    from rich.console import Console
    from rich.table import Table

    from .stats import StatsQueryError, run_stats_query

    db, _rt = _open_runtime()
    try:
        try:
            rows, title = run_stats_query(
                db, from_date=from_date, to_date=to_date, by=by
            )
        except StatsQueryError as exc:
            typer.echo(f"✘ {exc}")
            raise typer.Exit(2) from None

        if not rows:
            typer.echo(f"(no rows for --by={by} in [{from_date or '*'}, {to_date or '*'}])")
            return

        console = Console()
        tbl = Table(title=title)
        # Column order keys off the first row so dimension_scores can override
        # the default schema with its own pearson_r_* columns.
        col_order = [k for k in rows[0].keys() if k != "bucket"]
        tbl.add_column("bucket")
        for c in col_order:
            tbl.add_column(c, justify="right")
        for r in rows:
            cells = [str(r["bucket"])]
            for c in col_order:
                v = r.get(c)
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(str(v))
            tbl.add_row(*cells)
        console.print(tbl)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# prune (v0.3.0)
# ---------------------------------------------------------------------------


@app.command("prune")
def cmd_prune(
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="只展示将删除哪些 watchlist 行，不实际写库。",
    ),
    trade_date: Optional[str] = typer.Option(
        None, "--date",
        help="参照 trade_date (YYYYMMDD)；默认取最新开盘日。",
    ),
) -> None:
    """Phase-aware watchlist cleanup (always legacy renderer)."""
    db, rt = _open_runtime()
    try:
        params = PruneParams(dry_run=dry_run, trade_date=trade_date)
        outcome = ApwRunner(rt, renderer=LegacyStreamRenderer()).execute_prune(params)
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


@settings_app.command("reset")
def cmd_settings_reset(
    key: Optional[str] = typer.Option(
        None, "--key",
        help="只删除指定 key 的 override；省略则清空所有 override。",
    ),
) -> None:
    """Drop apw_config overrides (one key or all keys)."""
    from .config import ALLOWED_KEYS

    db, _rt = _open_runtime()
    try:
        if key is not None:
            if key not in ALLOWED_KEYS:
                typer.echo(f"✘ unknown key: {key!r}")
                typer.echo(f"  allowed: {', '.join(sorted(ALLOWED_KEYS))}")
                raise typer.Exit(2)
            db.execute("DELETE FROM apw_config WHERE key = ?", [key])
            typer.echo(f"✓ reset override for key {key!r}")
        else:
            db.execute("DELETE FROM apw_config")
            typer.echo("✓ cleared all apw_config overrides (defaults restored)")
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


# ---------------------------------------------------------------------------
# lgb subcommands (v0.5.0)
# ---------------------------------------------------------------------------


_LGB_LABEL_SOURCES = ("label_launch_t5", "label_launch_t10", "custom_t5")


@lgb_app.command("train")
def cmd_lgb_train(
    start: str = typer.Option(..., "--start", help="训练窗口起始 trade_date (YYYYMMDD)"),
    end: str = typer.Option(..., "--end", help="训练窗口结束 trade_date (YYYYMMDD)"),
    label_source: Optional[str] = typer.Option(
        None, "--label-source",
        help=f"标签源；默认读 apw_config。可选：{', '.join(_LGB_LABEL_SOURCES)}",
    ),
    label_threshold: Optional[float] = typer.Option(
        None, "--label-threshold",
        help="custom_t5 模式的收益阈值（百分比）",
    ),
    label_drawdown_threshold: Optional[float] = typer.Option(
        None, "--label-drawdown-threshold",
        help="custom_t5 模式的最大回撤阈值（百分比）",
    ),
    folds: Optional[int] = typer.Option(
        None, "--folds", help="GroupKFold 分组数；默认读 apw_config (=5)",
    ),
    no_activate: bool = typer.Option(
        False, "--no-activate", help="训练完不自动 activate 新模型",
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="忽略既有 checkpoint，全量重跑 Phase-1",
    ),
    keep_checkpoint: bool = typer.Option(
        False, "--keep-checkpoint",
        help="训练成功后保留 checkpoint 目录（默认成功即清理）",
    ),
) -> None:
    """Train a new APW LGB booster from apw_signal_history + apw_realized_returns."""
    from .config import ApwConfigStore
    from .lgb import dataset as _dataset
    from .lgb import trainer as _trainer

    if label_source and label_source not in _LGB_LABEL_SOURCES:
        typer.echo(
            f"✘ unknown --label-source {label_source!r}; "
            f"choose from {', '.join(_LGB_LABEL_SOURCES)}"
        )
        raise typer.Exit(2)

    db, _rt = _open_runtime()
    try:
        cfg = ApwConfigStore(db).load()
        if folds is not None:
            cfg.lgb_train_folds = folds
        typer.echo(
            f"▶ lgb train [{start} .. {end}] "
            f"label={(label_source or cfg.lgb_label_source)} "
            f"folds={cfg.lgb_train_folds}"
        )
        ds, ckpt = _dataset.collect_training_window(
            db,
            start_date=start,
            end_date=end,
            cfg=cfg,
            label_source=label_source,
            label_threshold_pct=label_threshold,
            label_drawdown_threshold_pct=label_drawdown_threshold,
            fresh=fresh,
            on_progress=lambda d, i, n: typer.echo(
                f"  [phase-1] {d}  ({i}/{n})"
            ),
        )
        typer.echo(
            f"  phase-1 done: n_signal_dates={len(ds.signal_dates)} "
            f"n_samples={ds.n_samples} n_labeled={ds.n_labeled}"
        )
        plugin_version = _read_plugin_version()
        try:
            result = _trainer.train_lightgbm(
                db,
                dataset=ds,
                cfg=cfg,
                plugin_version=plugin_version,
                activate=not no_activate,
            )
        except _trainer.LgbTrainError as exc:
            typer.echo(f"✘ train aborted: {exc}")
            raise typer.Exit(1) from None
        if not keep_checkpoint:
            ckpt.discard()
        typer.echo(
            "\n✓ trained model_id={mid}\n"
            "  n_samples={ns}  n_positive={npos}\n"
            "  CV AUC={auc:.4f} ± {auc_sd:.4f}  logloss={ll:.4f}\n"
            "  booster: {bp}\n"
            "  dataset: {dp}".format(
                mid=result.model_id,
                ns=result.n_samples,
                npos=result.n_positive,
                auc=result.cv_auc_mean,
                auc_sd=result.cv_auc_std,
                ll=result.cv_logloss_mean,
                bp=result.booster_path,
                dp=result.dataset_path,
            )
        )
    finally:
        db.close()


@lgb_app.command("list")
def cmd_lgb_list() -> None:
    """List all registered models (★ = active)."""
    from rich.console import Console
    from rich.table import Table

    from .lgb import registry as _registry

    db, _rt = _open_runtime()
    try:
        models = _registry.list_models(db)
        if not models:
            typer.echo("(no registered models — run `lgb train` first)")
            return
        tbl = Table(title="apw_lgb_models")
        for c in ("active", "model_id", "label_source",
                  "train_window", "n_samples", "n_positive", "AUC", "created"):
            tbl.add_column(c, overflow="fold")
        for m in models:
            tbl.add_row(
                "★" if m.is_active else "",
                m.model_id,
                m.label_source,
                f"{m.train_start_date}..{m.train_end_date}",
                str(m.n_samples),
                str(m.n_positive),
                "—" if m.cv_auc_mean is None else f"{m.cv_auc_mean:.4f}",
                "" if m.created_at is None else str(m.created_at)[:19],
            )
        Console().print(tbl)
    finally:
        db.close()


@lgb_app.command("info")
def cmd_lgb_info(
    model_id: Optional[str] = typer.Option(
        None, "--model-id",
        help="目标模型 id；省略读 active model",
    ),
) -> None:
    """Print one model's full metadata + recent usage stats."""
    from .lgb import registry as _registry

    db, _rt = _open_runtime()
    try:
        m = (
            _registry.get_model(db, model_id) if model_id
            else _registry.get_active(db)
        )
        if m is None:
            typer.echo(
                f"✘ model not found: {'(no active model)' if not model_id else model_id}"
            )
            raise typer.Exit(1)
        typer.echo(f"model_id          {m.model_id}")
        typer.echo(f"active            {m.is_active}")
        typer.echo(f"schema_version    {m.schema_version}")
        typer.echo(f"label_source      {m.label_source}")
        typer.echo(
            f"label_threshold   "
            f"{'—' if m.label_threshold_pct is None else m.label_threshold_pct}"
        )
        typer.echo(f"train_window      {m.train_start_date}..{m.train_end_date}")
        typer.echo(f"n_samples         {m.n_samples}")
        typer.echo(f"n_positive        {m.n_positive}")
        typer.echo(
            f"CV AUC            "
            f"{'—' if m.cv_auc_mean is None else f'{m.cv_auc_mean:.4f} ± {m.cv_auc_std:.4f}'}"
        )
        typer.echo(
            f"CV logloss        "
            f"{'—' if m.cv_logloss_mean is None else f'{m.cv_logloss_mean:.4f}'}"
        )
        typer.echo(f"feature_count     {m.feature_count}")
        typer.echo(f"framework_version {m.framework_version}")
        typer.echo(f"plugin_version    {m.plugin_version}")
        typer.echo(f"git_commit        {m.git_commit}")
        typer.echo(f"file_path         {m.file_path}")
        typer.echo(f"created_at        {m.created_at}")
    finally:
        db.close()


@lgb_app.command("activate")
def cmd_lgb_activate(model_id: str = typer.Argument(...)) -> None:
    """Switch the active model atomically."""
    from .lgb import registry as _registry

    db, _rt = _open_runtime()
    try:
        if not _registry.set_active(db, model_id):
            typer.echo(f"✘ model not found: {model_id!r}")
            raise typer.Exit(1)
        typer.echo(f"✓ activated {model_id}")
    finally:
        db.close()


@lgb_app.command("prune")
def cmd_lgb_prune(
    keep: int = typer.Option(
        5, "--keep",
        help="保留多少条非 active 的模型行（active 永远保留）",
    ),
) -> None:
    """Delete old non-active models + their booster / dataset files."""
    from .lgb import cleanup as _cleanup

    db, _rt = _open_runtime()
    try:
        rep = _cleanup.prune_models(db, keep=keep)
        typer.echo(
            f"✓ kept {len(rep.kept)}  deleted {len(rep.deleted)}  "
            f"missing_files {len(rep.missing_files)}"
        )
        for mid in rep.kept:
            typer.echo(f"  kept     {mid}")
        for mid in rep.deleted:
            typer.echo(f"  deleted  {mid}")
        for fp in rep.missing_files:
            typer.echo(f"  missing  {fp}")
    finally:
        db.close()


@lgb_app.command("purge")
def cmd_lgb_purge(
    datasets: bool = typer.Option(False, "--datasets"),
    models: bool = typer.Option(False, "--models"),
    predictions: bool = typer.Option(False, "--predictions"),
    checkpoints: bool = typer.Option(False, "--checkpoints"),
    all_scopes: bool = typer.Option(False, "--all"),
    yes: bool = typer.Option(False, "--yes", help="跳过确认提示"),
) -> None:
    """Wholesale clear of LGB artefacts by scope (DESTRUCTIVE)."""
    from .lgb import cleanup as _cleanup

    if all_scopes:
        datasets = models = predictions = checkpoints = True
    if not any((datasets, models, predictions, checkpoints)):
        typer.echo(
            "✘ pick at least one scope: --datasets / --models / "
            "--predictions / --checkpoints / --all"
        )
        raise typer.Exit(2)
    if not yes:
        scopes = [n for n, v in (
            ("datasets", datasets), ("models", models),
            ("predictions", predictions), ("checkpoints", checkpoints),
        ) if v]
        typer.echo(
            f"⚠ this will DESTRUCTIVELY purge: {', '.join(scopes)}\n"
            "  re-run with --yes to confirm."
        )
        raise typer.Exit(2)

    db, _rt = _open_runtime()
    try:
        reports = _cleanup.purge(
            db,
            datasets=datasets,
            models=models,
            predictions=predictions,
            checkpoints=checkpoints,
        )
        for r in reports:
            typer.echo(
                f"  {r.scope:12s} files_removed={r.files_removed}  "
                f"rows_removed={r.rows_removed}"
            )
        typer.echo("✓ purge complete")
    finally:
        db.close()


def _read_plugin_version() -> str:
    """Read the version from the bundled deeptrade_plugin.yaml.

    Lightweight YAML parse (looking for the leading ``version:`` line) so we
    don't have to add a runtime PyYAML dependency just for this audit field.
    """
    from pathlib import Path

    yaml_path = (
        Path(__file__).resolve().parent.parent / "deeptrade_plugin.yaml"
    )
    try:
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "unknown"


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
