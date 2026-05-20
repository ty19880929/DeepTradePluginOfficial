"""Plugin-managed CLI for checkmate.

Iter-0 PR-0.3 ships **only the command surface** — every subcommand prints
``not yet implemented in Iter-0`` and exits with code 2 so the dispatch wiring
is exercised end-to-end before any real pipeline arrives. Subsequent iters
flesh out the bodies:

* Iter-1: ``sync``
* Iter-2: ``scan``
* Iter-3: ``signals`` / ``explain``
* Iter-4: ``backtest`` / ``report``
* Iter-1+: ``settings show`` / ``settings reset``

CLI surface mirrors development_plan §4 (v0.1.0 target).
"""

from __future__ import annotations

import sys

import typer

from .cancellation import install_sigint_marker

app = typer.Typer(
    name="checkmate",
    help="Checkmate 趋势跟踪策略 — A 股 long-only 中期趋势组合系统。",
    no_args_is_help=True,
    add_completion=False,
)

settings_app = typer.Typer(
    name="settings",
    help="读写本插件可调参数（ConfigService 命名空间 checkmate.*）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(settings_app, name="settings")


_NOT_IMPLEMENTED_MSG = "not yet implemented in Iter-0"


def _derive_include_reasons(features: dict) -> list[str]:
    """Generate human-readable 'why this stock made it' bullets from features.

    Used by ``cmd_explain --json`` to surface :py:attr:`include_reasons`
    alongside the universe-side :py:attr:`exclude_reasons`. Heuristics are
    cheap threshold checks on already-computed feature values — no DB
    re-reads, no recomputation.
    """
    out: list[str] = []
    close = features.get("close_qfq")
    ma20 = features.get("ma20")
    ma60 = features.get("ma60")
    ma120 = features.get("ma120")
    if close is not None and ma20 is not None and close > ma20:
        out.append("close > MA20 (上升趋势)")
    if ma20 is not None and ma60 is not None and ma20 > ma60:
        out.append("MA20 > MA60 (中期趋势成立)")
    if ma60 is not None and ma120 is not None and ma60 > ma120:
        out.append("MA60 > MA120 (长期趋势成立)")
    rs60 = features.get("rs60_pctile")
    if rs60 is not None and rs60 >= 0.6:
        out.append(f"rs60_pctile={rs60:.2f} (强于市场)")
    score = features.get("score")
    if score is not None and score >= 60:
        out.append(f"综合 score={score:.1f}")
    dd = features.get("drawdown_60d_high")
    if dd is not None and -0.10 <= dd <= -0.01:
        out.append(f"轻度回踩 ({dd*100:.1f}%)")
    above_ma20 = features.get("above_ma20_days")
    if above_ma20 is not None and above_ma20 >= 40:
        out.append(f"近 60 日有 {above_ma20} 日站上 MA20")
    return out


def _stub() -> None:
    """Shared body for every Iter-0 stub subcommand."""
    typer.echo(_NOT_IMPLEMENTED_MSG)
    raise typer.Exit(2)


# ---------------------------------------------------------------------------
# Iter-1 surface
# ---------------------------------------------------------------------------


@app.command("sync")
def cmd_sync(
    start: str = typer.Option("2014-01-01", "--start", help="YYYY-MM-DD; default 2014-01-01"),
    end: str | None = typer.Option(None, "--end", help="YYYY-MM-DD; default 今日"),
    symbols: str | None = typer.Option(None, "--symbols", help="逗号分隔 ts_code 子集 (dev 友好)"),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="忽略本地 parquet 缓存"),
) -> None:
    """预热历史数据：trade_cal / stock_basic+namechange survivorship / 每只 daily+daily_basic 缓存。

    设计来源：development_plan.md §4 + iteration_tasks.md §2 PR-1.2。
    Iter-1 单线程，无 dashboard；Iter-5 才接 EventRenderer。
    """
    from . import paths as _paths  # noqa: PLC0415
    from .runtime import build_tushare_client, open_runtime  # noqa: PLC0415
    from .sync import SyncParams, parse_symbols, run_sync  # noqa: PLC0415

    _paths.ensure_layout()
    db, rt = open_runtime()
    try:
        rt.tushare = build_tushare_client(rt)
        outcome = run_sync(
            rt,
            SyncParams(
                start=start,
                end=end,
                symbols=parse_symbols(symbols),
                force_refresh=force_refresh,
            ),
            echo=typer.echo,
        )
        if outcome.errors:
            typer.echo(f"[sync] {len(outcome.errors)} symbol(s) failed; first: {outcome.errors[0]}")
            raise typer.Exit(1)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Iter-2 surface
# ---------------------------------------------------------------------------


@app.command("scan")
def cmd_scan(
    date: str | None = typer.Option(None, "--date", help="YYYYMMDD; default prev_session(today)"),
    quiet: bool = typer.Option(False, "--quiet", help="抑制 stdout 表格"),
    no_dashboard: bool = typer.Option(  # noqa: ARG001 — Iter-5 wires this
        False, "--no-dashboard", help="禁用动态仪表盘 (Iter-5 起生效)"
    ),
    quiet_disclaimer: bool = typer.Option(  # noqa: ARG001 — Iter-4 wires this
        False, "--quiet-disclaimer", help="抑制免责声明输出 (CI 友好)"
    ),
) -> None:
    """单日盘后扫描：universe → features → regime → score。

    PR-2.3 完整化：调用 ``scan.run_scan`` 串通 4 个 step + 写 3 张表 + 注册
    ``checkmate_runs(mode='scan')`` + 推流到 ``checkmate_events``。CLI 仅做
    参数收集 / 输出格式化；编排逻辑见 ``checkmate.scan.run_scan``。
    """
    from . import paths as _paths  # noqa: PLC0415
    from .runtime import build_tushare_client, open_runtime  # noqa: PLC0415
    from .scan import ScanParams, run_scan  # noqa: PLC0415
    from .ui import choose_renderer  # noqa: PLC0415

    _paths.ensure_layout()
    db, rt = open_runtime()
    try:
        rt.tushare = build_tushare_client(rt)
        renderer = choose_renderer(no_dashboard=no_dashboard, mode="scan")
        outcome = run_scan(
            rt,
            ScanParams(trade_date=date, quiet=quiet),
            renderer=renderer,
        )
        typer.echo(
            f"[scan] run_id={outcome.run_id}  trade_date={outcome.trade_date}  "
            f"regime={outcome.regime}  exposure_cap={outcome.exposure_cap:.2f}"
        )
        typer.echo(
            f"[scan] universe_total={outcome.n_universe}  eligible={outcome.n_eligible}  "
            f"features_rows={outcome.n_features}"
        )
        if outcome.reason_breakdown:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(outcome.reason_breakdown.items()))
            typer.echo(f"[scan] reason_breakdown: {parts}")
        if not quiet and outcome.top_scored:
            typer.echo("[scan] top 30 by score:")
            for r in outcome.top_scored:
                score = r["score"] if r["score"] is not None else float("nan")
                close = r["close_qfq"] if r["close_qfq"] is not None else float("nan")
                typer.echo(
                    f"  {r['ts_code']}  score={score:>6.2f}  "
                    f"close={close:>9.2f}  ret_60={r['ret_60']}"
                )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Iter-3 surface
# ---------------------------------------------------------------------------


@app.command("signals")
def cmd_signals(
    date: str | None = typer.Option(None, "--date", help="YYYYMMDD"),
    portfolio_value: float = typer.Option(
        1_000_000.0, "--portfolio-value",
        help="Sizing 用的组合总值（v0.1 默认 100 万；backtest 自动注入实际值）",
    ),
    no_dashboard: bool = typer.Option(False, "--no-dashboard"),  # noqa: ARG001
    quiet_disclaimer: bool = typer.Option(False, "--quiet-disclaimer"),  # noqa: ARG001
) -> None:
    """在 scan 输出上叠加 entry/exit 判定 + risk 过滤，落 checkmate_signals。

    前置条件：本日的 ``scan`` 已经跑过（features / universe / regime 三表有数据）。
    """
    from . import paths as _paths  # noqa: PLC0415
    from .runtime import build_tushare_client, open_runtime  # noqa: PLC0415
    from .signals import SignalsParams, run_signals  # noqa: PLC0415
    from .ui import choose_renderer  # noqa: PLC0415

    _paths.ensure_layout()
    db, rt = open_runtime()
    try:
        rt.tushare = build_tushare_client(rt)
        renderer = choose_renderer(no_dashboard=no_dashboard, mode="signals")
        outcome = run_signals(
            rt,
            SignalsParams(trade_date=date, portfolio_value=portfolio_value),
            renderer=renderer,
        )
        typer.echo(
            f"[signals] run_id={outcome.run_id}  trade_date={outcome.trade_date}  "
            f"regime={outcome.regime}"
        )
        typer.echo(
            f"[signals] entry_proposals={outcome.n_entry_proposals}  "
            f"accepted={outcome.n_entry_accepted}  exits={outcome.n_exits}"
        )
        if outcome.n_entry_proposals == 0 and outcome.n_exits == 0:
            typer.echo("[signals] no signals today")
    finally:
        db.close()


@app.command("explain")
def cmd_explain(
    date: str = typer.Option(..., "--date", help="YYYYMMDD"),
    symbol: str = typer.Option(..., "--symbol", help="ts_code, e.g. 600519.SH"),
    as_json: bool = typer.Option(False, "--json", help="输出结构化 JSON"),
) -> None:
    """解释单只股票当日 include / exclude / signal / exit 原因。

    Reads ``checkmate_universe_daily`` + ``checkmate_features_daily`` +
    ``checkmate_signals`` for the given ``(date, symbol)`` and prints a
    human-readable summary (or structured JSON with ``--json``).
    """
    import json as _json  # noqa: PLC0415

    from . import paths as _paths  # noqa: PLC0415
    from .runtime import open_runtime  # noqa: PLC0415

    _paths.ensure_layout()
    db, rt = open_runtime()
    try:
        d = date.replace("-", "")

        universe_row = rt.db.execute(
            """
            SELECT eligible, reason_codes, liquidity_score, name, industry,
                   amount_20d_avg, turnover_20d_avg, is_st, list_status
            FROM checkmate_universe_daily
            WHERE trade_date = ? AND ts_code = ?
            """,
            [d, symbol],
        ).fetchone()

        features_row = rt.db.execute(
            """
            SELECT score, score_breakdown, close_qfq, ma20, ma60, ma120,
                   atr20, atr_pct, ret_60, ret_120, rs60_pctile, rs120_pctile,
                   amount_20d_avg, turnover_20d_avg, limit_freq_60d,
                   drawdown_60d_high, quiet_score, above_ma20_days
            FROM checkmate_features_daily
            WHERE trade_date = ? AND ts_code = ?
            """,
            [d, symbol],
        ).fetchone()

        signal_rows = rt.db.execute(
            """
            SELECT action, signal_type, score, explain
            FROM checkmate_signals
            WHERE signal_date = ? AND ts_code = ?
            ORDER BY action
            """,
            [d, symbol],
        ).fetchall()

        regime_row = rt.db.execute(
            "SELECT regime, exposure_cap FROM checkmate_regime_daily WHERE trade_date = ?",
            [d],
        ).fetchone()

        payload: dict[str, object] = {
            "trade_date": d,
            "symbol": symbol,
            # Legacy sections (kept for back-compat with the v0.1 explain JSON).
            "universe": None,
            "features": None,
            "signals": [],
            "regime": None,
            # PR-5.3 spec'd schema — 6 top-level explainability fields shared
            # with the dashboard cards.
            "include_reasons": [],
            "exclude_reasons": [],
            "score_breakdown": None,
            "entry_plan": None,
            "exit_plan": None,
            "risk_snapshot": None,
        }
        if universe_row:
            payload["universe"] = {
                "eligible": bool(universe_row[0]),
                "reason_codes": _json.loads(universe_row[1]) if universe_row[1] else [],
                "liquidity_score": universe_row[2],
                "name": universe_row[3],
                "industry": universe_row[4],
                "amount_20d_avg": universe_row[5],
                "turnover_20d_avg": universe_row[6],
                "is_st": bool(universe_row[7]) if universe_row[7] is not None else None,
                "list_status": universe_row[8],
            }
            payload["exclude_reasons"] = payload["universe"]["reason_codes"]
        if features_row:
            fr_cols = ("score", "score_breakdown", "close_qfq",
                       "ma20", "ma60", "ma120",
                       "atr20", "atr_pct", "ret_60", "ret_120",
                       "rs60_pctile", "rs120_pctile",
                       "amount_20d_avg", "turnover_20d_avg", "limit_freq_60d",
                       "drawdown_60d_high", "quiet_score", "above_ma20_days")
            features_dict = dict(zip(fr_cols, features_row))
            if features_dict["score_breakdown"]:
                features_dict["score_breakdown"] = _json.loads(features_dict["score_breakdown"])
            payload["features"] = features_dict
            payload["score_breakdown"] = features_dict.get("score_breakdown")
            payload["include_reasons"] = _derive_include_reasons(features_dict)
        for action, signal_type, score, explain_json in signal_rows:
            ex = _json.loads(explain_json) if explain_json else {}
            sig_record = {
                "action": action, "signal_type": signal_type, "score": score,
                "explain": ex,
            }
            payload["signals"].append(sig_record)  # type: ignore[union-attr]
            # PR-5.3 schema: surface accepted/rejected entries + exits as
            # dedicated top-level plans so the dashboard / consumer code
            # can find them without iterating `signals`.
            if action == "enter":
                payload["entry_plan"] = ex
            elif action == "exit":
                payload["exit_plan"] = ex
            elif action == "rejected":
                payload["risk_snapshot"] = {
                    "rejected": True,
                    "cancel_reason": ex.get("cancel_reason"),
                    "entry_price": ex.get("entry_price"),
                    "stop_price": ex.get("stop_price"),
                }
        # If no rejected entry was found but an accepted one exists, surface
        # the accepted sizing as risk_snapshot.
        if payload["risk_snapshot"] is None and payload["entry_plan"]:
            ep = payload["entry_plan"]  # type: ignore[index]
            if isinstance(ep, dict) and "shares" in ep:
                payload["risk_snapshot"] = {
                    "rejected": False,
                    "shares": ep.get("shares"),
                    "weight": ep.get("weight"),
                    "industry": ep.get("industry"),
                }
        if regime_row:
            payload["regime"] = {
                "regime": regime_row[0], "exposure_cap": regime_row[1],
            }

        if as_json:
            typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return

        # --- pretty stdout
        typer.echo(f"=== explain {symbol} @ {d} ===")
        if payload["regime"]:
            r = payload["regime"]  # type: ignore[index]
            typer.echo(f"  regime={r['regime']}  exposure_cap={r['exposure_cap']}")
        if payload["universe"]:
            u = payload["universe"]  # type: ignore[index]
            typer.echo(
                f"  universe: eligible={u['eligible']}  name={u['name']}  "
                f"industry={u['industry']}  list_status={u['list_status']}  is_st={u['is_st']}"
            )
            if u["reason_codes"]:
                typer.echo(f"           reason_codes={u['reason_codes']}")
        else:
            typer.echo("  universe: <no row>")
        if payload["features"]:
            f = payload["features"]  # type: ignore[index]
            typer.echo(
                f"  features: score={f.get('score')}  close_qfq={f.get('close_qfq')}  "
                f"ma20={f.get('ma20')}  ma60={f.get('ma60')}  atr_pct={f.get('atr_pct')}"
            )
            typer.echo(
                f"            ret_60={f.get('ret_60')}  ret_120={f.get('ret_120')}  "
                f"rs60_pctile={f.get('rs60_pctile')}"
            )
            if f.get("score_breakdown") and isinstance(f["score_breakdown"], dict):
                comps = f["score_breakdown"].get("components", {})
                if comps:
                    typer.echo(f"  score_breakdown: {comps}")
        else:
            typer.echo("  features: <no row>")
        if payload["signals"]:
            typer.echo("  signals:")
            for s in payload["signals"]:  # type: ignore[union-attr]
                typer.echo(
                    f"    action={s['action']}  signal_type={s['signal_type']}  "
                    f"score={s['score']}"
                )
                ex = s["explain"]
                if isinstance(ex, dict):
                    if ex.get("hit"):
                        typer.echo(f"      hit:    {ex['hit']}")
                    if ex.get("missed"):
                        typer.echo(f"      missed: {ex['missed']}")
                    if "cancel_reason" in ex and ex["cancel_reason"]:
                        typer.echo(f"      cancel_reason: {ex['cancel_reason']}")
        else:
            typer.echo("  signals: <none>")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Iter-4 surface
# ---------------------------------------------------------------------------


@app.command("backtest")
def cmd_backtest(
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD"),
    initial_cash: float = typer.Option(
        1_000_000.0, "--initial-cash", help="起始资金，默认 100 万",
    ),
    resume: bool = typer.Option(True, "--resume/--fresh", help="断点续算 / 重头算"),
    grid: str | None = typer.Option(
        None, "--grid",
        help="参数网格 YAML 文件路径；命中时跑笛卡尔展开的所有 cell 而非单次回测 (PR-6.1)",
    ),
    rank_by: str = typer.Option(
        "final_equity", "--rank-by",
        help="网格排名度量：final_equity / cagr / max_drawdown / n_fills (PR-6.1)",
    ),
    split: str | None = typer.Option(
        None, "--split",
        help=(
            "训练-验证-OOS 切分，三段独立跑回测："
            "'train=2014-2020 val=2021-2023 oos=2024-2026'。"
            "与 --grid 同用时拒绝在 OOS 窗口排名 (PR-6.2)。"
        ),
    ),
    no_dashboard: bool = typer.Option(
        False, "--no-dashboard", help="禁用动态仪表盘"
    ),
    quiet_disclaimer: bool = typer.Option(  # noqa: ARG001
        False, "--quiet-disclaimer", help="抑制免责声明输出 (CI 友好)"
    ),
) -> None:
    """区间回测：trade_cal 单日推进，落 checkmate_backtest_runs + trades + positions。

    `--grid params.yaml` 切到参数扫描模式：每个 cell 独立 config_hash + 独立
    checkpoint 目录，可单独 resume；末尾输出按 `--rank-by` 排序的 top-N。
    """
    from pathlib import Path  # noqa: PLC0415

    from . import paths as _paths  # noqa: PLC0415
    from .backtest import BacktestParams, run_backtest  # noqa: PLC0415
    from .runtime import build_tushare_client, open_runtime  # noqa: PLC0415
    from .ui import choose_renderer  # noqa: PLC0415

    _paths.ensure_layout()
    db, rt = open_runtime()
    try:
        rt.tushare = build_tushare_client(rt)

        # ----- Split mode (PR-6.2) -----
        if split:
            from .split import forbid_rank_on_oos, parse_split  # noqa: PLC0415

            try:
                split_obj = parse_split(split)
            except ValueError as exc:
                typer.echo(f"[backtest] invalid --split: {exc}")
                raise typer.Exit(2)
            if grid:
                try:
                    forbid_rank_on_oos(split_obj, grid_start=start, grid_end=end)
                except ValueError as exc:
                    typer.echo(f"[backtest] split guard: {exc}")
                    raise typer.Exit(2)

            typer.echo(
                f"[backtest] split mode: "
                + " | ".join(f"{s.name}={s.start}..{s.end}" for s in split_obj.segments)
            )
            segment_outcomes: list[tuple[str, "object"]] = []
            for seg in split_obj.segments:
                renderer = choose_renderer(no_dashboard=no_dashboard, mode="backtest")
                outcome = run_backtest(
                    rt,
                    BacktestParams(
                        start=seg.start, end=seg.end,
                        initial_cash=initial_cash, resume=resume,
                    ),
                    renderer=renderer,
                )
                segment_outcomes.append((seg.name, outcome))

            typer.echo("")
            typer.echo("=== split segment summary ===")
            for name, o in segment_outcomes:
                ret_pct = (o.final_equity / initial_cash - 1.0) * 100.0 if initial_cash > 0 else 0.0
                typer.echo(
                    f"  [{name:>5}] {o.start} → {o.end}  "
                    f"({o.n_days} sessions)  "
                    f"return={ret_pct:+.2f}%  "
                    f"max_dd={o.max_drawdown*100:.2f}%  "
                    f"fills={o.n_fills}  "
                    f"run_id={o.run_id[:8]}"
                )
            return

        if grid:
            # ----- Grid mode (PR-6.1) -----
            from .grid import load_grid_yaml, run_grid  # noqa: PLC0415

            grid_path = Path(grid).expanduser().resolve()
            if not grid_path.is_file():
                typer.echo(f"[backtest] grid file not found: {grid_path}")
                raise typer.Exit(2)
            try:
                grid_def = load_grid_yaml(grid_path)
            except ValueError as exc:
                typer.echo(f"[backtest] invalid grid yaml: {exc}")
                raise typer.Exit(2)
            base = BacktestParams(
                start=start, end=end,
                initial_cash=initial_cash, resume=resume,
            )
            aggregate = run_grid(rt, base, grid_def, rank_by=rank_by, echo=typer.echo)
            typer.echo("")
            typer.echo(f"=== grid summary (n_cells={aggregate['n_cells']}, rank_by={rank_by}) ===")
            top_n = min(10, len(aggregate["ranked"]))
            for i, r in enumerate(aggregate["ranked"][:top_n], start=1):
                ov_parts = []
                for section, fields in (r["overrides"] or {}).items():
                    for k, v in fields.items():
                        ov_parts.append(f"{section}.{k}={v}")
                ov_repr = "  ".join(ov_parts) or "(defaults)"
                typer.echo(
                    f"  [{i:>2}] {ov_repr}\n"
                    f"       final_equity={r['final_equity']:,.0f}  "
                    f"max_dd={r['max_drawdown']*100:.2f}%  "
                    f"n_fills={r['n_fills']}"
                )
            typer.echo("")
            typer.echo(f"  output: {aggregate.get('output_path')}")
            return

        # ----- Single backtest -----
        renderer = choose_renderer(no_dashboard=no_dashboard, mode="backtest")
        outcome = run_backtest(
            rt,
            BacktestParams(
                start=start, end=end,
                initial_cash=initial_cash, resume=resume,
            ),
            renderer=renderer,
        )
        typer.echo("")
        typer.echo("=== backtest summary ===")
        typer.echo(f"  run_id        : {outcome.run_id}")
        typer.echo(f"  config_hash   : {outcome.config_hash}")
        typer.echo(f"  window        : {outcome.start} → {outcome.end}  ({outcome.n_days} sessions)")
        typer.echo(f"  initial_cash  : {initial_cash:,.2f}")
        typer.echo(f"  final_equity  : {outcome.final_equity:,.2f}")
        typer.echo(f"  final_cash    : {outcome.final_cash:,.2f}")
        typer.echo(f"  max_drawdown  : {outcome.max_drawdown*100:.2f}%")
        typer.echo(f"  n_fills       : {outcome.n_fills}")
        typer.echo("")
        typer.echo(f"  tip: 跑 `deeptrade checkmate report --run-id {outcome.run_id}` 看完整指标。")
    finally:
        db.close()


@app.command("report")
def cmd_report(
    run_id: str = typer.Option(..., "--run-id"),
    as_json: bool = typer.Option(False, "--json", help="同时把 JSON 写到 reports/ 目录"),
    as_markdown: bool = typer.Option(False, "--markdown", help="同时把 Markdown 写到 reports/ 目录"),
    as_html: bool = typer.Option(
        False, "--html", help="渲染 HTML 报告（含权益曲线 / 月度热力图 / 退出原因饼图）"
    ),
) -> None:
    """把已有回测聚合成 UI 可消费 JSON / Markdown / HTML 片段，并打印摘要到 stdout。"""
    from . import paths as _paths  # noqa: PLC0415
    from . import report as _report  # noqa: PLC0415
    from .runtime import open_runtime  # noqa: PLC0415

    _paths.ensure_layout()
    db, rt = open_runtime()
    try:
        try:
            payload = _report.build_report(rt, run_id)
        except ValueError as exc:
            typer.echo(f"[report] {exc}")
            raise typer.Exit(2)

        # stdout — always show the Markdown summary so it's pipe-friendly.
        typer.echo(_report.to_markdown(payload))

        if as_json:
            path = _report.write_to_disk(payload, fmt="json")
            typer.echo(f"\n[report] wrote {path}")
        if as_markdown:
            path = _report.write_to_disk(payload, fmt="markdown")
            typer.echo(f"\n[report] wrote {path}")
        if as_html:
            path = _report.write_to_disk(payload, fmt="html")
            typer.echo(f"\n[report] wrote {path}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@settings_app.command("show")
def cmd_settings_show() -> None:
    """打印当前 checkmate.* 命名空间的全部配置项。"""
    _stub()


@settings_app.command("reset")
def cmd_settings_reset(
    yes: bool = typer.Option(False, "--yes", help="不再二次确认"),  # noqa: ARG001
) -> None:
    """重置所有 checkmate.* 配置项为默认值。"""
    _stub()


# ---------------------------------------------------------------------------
# Entry called by Plugin.dispatch()
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """Entry called by the framework via :meth:`CheckmatePlugin.dispatch`.

    With ``standalone_mode=False`` typer/click do NOT call sys.exit; they
    return the exit code from ``app(...)`` directly (and ALSO surface
    ``typer.Exit`` if raised before invoke completes).
    """
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
