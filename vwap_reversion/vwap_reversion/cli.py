"""Plugin-managed CLI for vwap-reversion.

Subcommands:
    run      — 模拟交易 daemon（实时采集 + VWAP 带回归 + Paper 撮合）  [P1/P2]
    backtest — 回放已采集 vwr_bars 复算指标                            [P2/P3]
    report   — 重新生成/查看某次 run 的双报告                          [P3]
    history  — 列出历史 run
    settings — show / set / reset 持久化参数

Invoked via the framework's pure pass-through dispatch:
    deeptrade vwap-reversion <subcommand> [...]
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from deeptrade.core import paths
from deeptrade.core.config import ConfigService
from deeptrade.core.db import Database
from deeptrade.plugins_api import PluginContext, render_exception

from .clock import MarketClock
from .config import list_for_show, load_config, reset_config, set_one
from .runtime import PLUGIN_ID, VwrRuntime

app = typer.Typer(
    name="vwap-reversion",
    help="VWAP 带回归日内策略 — T+0 ETF 实时采集 + 模拟交易（daemon 形态）。",
    no_args_is_help=True,
    add_completion=False,
)

settings_app = typer.Typer(
    name="settings",
    help="本插件可持久化的运行参数（VWAP 带 k 值 / 风控 / 模拟账户 / 时区等）。",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(settings_app, name="settings")


def _open_runtime() -> tuple[Database, VwrRuntime, PluginContext]:
    """Build the per-process services bundle（仿 limit_up_board._open_runtime）。

    ``MarketClock`` 用持久化的 ``market_timezone`` 构造 —— 这一步必须在任何
    交易日/会话判断之前完成（设计 §4：绝不读本机本地时区）。
    """
    db = Database(paths.db_path())
    cfg_service = ConfigService(db)
    ctx = PluginContext(db=db, config=cfg_service, plugin_id=PLUGIN_ID)
    vwr_cfg = load_config(db)
    rt = VwrRuntime(db=db, config=cfg_service, clock=MarketClock(vwr_cfg.market_timezone))
    return db, rt, ctx


# ---------------------------------------------------------------------------
# run / backtest / report — P1+ 占位（P0 仅保证 CLI 面孔与参数面定型）
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    code: str = typer.Option(..., "--code", help="ETF 代码（如 159518.SZ）"),
    date: str | None = typer.Option(
        None, "--date", help="交易日 YYYYMMDD；默认市场时区（上海）今日。"
        "实时轮询只对今日有意义，给其他日期会拒绝启动。"
    ),
    poll_interval: int | None = typer.Option(
        None, "--poll-interval", help="轮询间隔秒；默认读 settings（30s）。一次性覆盖。"
    ),
    position_mode: str | None = typer.Option(
        None, "--position-mode", help="round_trip / base_position_t；默认读 settings。"
    ),
    no_dashboard: bool = typer.Option(
        False, "--no-dashboard", help="禁用实时双面板，使用行式输出"
    ),
) -> None:
    """模拟交易 daemon：开盘前待机 → 实时采集 +（P2）信号/Paper 撮合 → 收盘收尾。"""
    from dataclasses import replace  # noqa: PLC0415

    from .config import validate_config  # noqa: PLC0415
    from .daemon import CollectDaemon, PreconditionError  # noqa: PLC0415
    from .feed.tushare_realtime import TushareRealtimeSource  # noqa: PLC0415
    from .persistence import TradeCalendarStore  # noqa: PLC0415
    from .runtime import build_tushare_client  # noqa: PLC0415
    from .ui import choose_renderer  # noqa: PLC0415

    code = code.strip()
    if not code:
        typer.echo("✘ --code 不能为空")
        raise typer.Exit(2)

    db, rt, _ctx = _open_runtime()
    try:
        cfg = load_config(db)
        overrides = {}
        if poll_interval is not None:
            overrides["poll_interval_seconds"] = poll_interval
        if position_mode is not None:
            overrides["position_mode"] = position_mode
        if overrides:
            cfg = replace(cfg, **overrides)
            try:
                validate_config(cfg)
            except ValueError as e:
                typer.echo(f"✘ {e}")
                raise typer.Exit(2) from e

        today = rt.clock.today_str()
        if date is not None and date.strip() != today:
            typer.echo(
                f"✘ --date {date} ≠ 市场时区今日 {today}：实时轮询只能跑当日。"
                "历史日请等 backtest（已采集日）或 backfill（后续）。"
            )
            raise typer.Exit(2)

        tushare = build_tushare_client(rt)
        rt.tushare = tushare
        calendar = TradeCalendarStore(db)
        if not calendar.covers(today):
            typer.echo("… 同步交易日历（trade_cal，首次一次性）")
            calendar.ensure_synced(tushare)

        daemon = CollectDaemon(
            rt, cfg,
            source=TushareRealtimeSource(tushare, code, rt.clock),
            calendar=calendar,
            renderer=choose_renderer(no_dashboard=no_dashboard),
        )
        try:
            outcome = daemon.execute(code=code)
        except PreconditionError as e:
            typer.echo(f"✘ {e}")
            raise typer.Exit(2) from e
        typer.echo(f"\nstatus: {outcome.status}  run_id: {outcome.run_id}")
        if outcome.message:
            typer.echo(f"message: {outcome.message}")
        if outcome.exit_code:
            raise typer.Exit(outcome.exit_code)
    finally:
        db.close()


@app.command("backtest")
def cmd_backtest(
    code: str = typer.Option(..., "--code", help="ETF 代码"),
    start: str = typer.Option(..., "--start", help="起始交易日 YYYYMMDD"),
    end: str = typer.Option(..., "--end", help="结束交易日 YYYYMMDD"),
    k_entry: float | None = typer.Option(None, "--k-entry", help="一次性覆盖 band_k_entry"),
    k_exit: float | None = typer.Option(None, "--k-exit", help="一次性覆盖 band_k_exit"),
    k_stop: float | None = typer.Option(None, "--k-stop", help="一次性覆盖 band_k_stop"),
    warmup_minutes: int | None = typer.Option(None, "--warmup-minutes"),
    position_mode: str | None = typer.Option(None, "--position-mode"),
) -> None:
    """回放已采集的 vwr_bars 复算信号与指标（零额外接口依赖；回放=实盘同款撮合）。"""
    from dataclasses import replace  # noqa: PLC0415

    from .backtest import BacktestError, run_backtest  # noqa: PLC0415
    from .config import validate_config  # noqa: PLC0415
    from .paths import backtest_report_dir  # noqa: PLC0415
    from .persistence import set_run_report_dir  # noqa: PLC0415
    from .reporting import build_backtest_report  # noqa: PLC0415

    db, _rt, _ctx = _open_runtime()
    try:
        cfg = load_config(db)
        overrides = {
            k: v for k, v in {
                "band_k_entry": k_entry, "band_k_exit": k_exit,
                "band_k_stop": k_stop, "warmup_minutes": warmup_minutes,
                "position_mode": position_mode,
            }.items() if v is not None
        }
        if overrides:
            cfg = replace(cfg, **overrides)
            try:
                validate_config(cfg)
            except ValueError as e:
                typer.echo(f"✘ {e}")
                raise typer.Exit(2) from e
        try:
            outcome = run_backtest(db, cfg, code=code.strip(), start=start, end=end)
        except BacktestError as e:
            typer.echo(f"✘ {e}")
            raise typer.Exit(2) from e

        # 报告：生成 markdown 落盘 + 终端摘要
        md = build_backtest_report(db, outcome.run_id)
        out_dir = backtest_report_dir(code.strip(), start, end)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "backtest_report.md"
        report_path.write_text(md, encoding="utf-8")
        set_run_report_dir(db, outcome.run_id, str(out_dir))

        agg = outcome.aggregate
        console = Console()
        table = Table(title=f"backtest {code} {start}..{end}（run {outcome.run_id[:8]}…）")
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")
        table.add_row("回放交易日数", str(agg["n_days"]))
        table.add_row("总成交笔数", str(agg["n_trades"]))
        table.add_row("总净盈亏", f"{agg['net_pnl_total']:+,.2f} 元")
        table.add_row(
            "按日胜率",
            f"{agg['day_win_rate']:.0%}" if agg["day_win_rate"] is not None else "—",
        )
        table.add_row(
            "日频 Sharpe", f"{agg['sharpe']:.2f}" if agg["sharpe"] is not None else "—"
        )
        table.add_row("总费用/滑点", f"{agg['total_fee']:.2f} / {agg['total_slippage']:.2f} 元")
        table.add_row("熔断天数", str(agg["n_circuit_days"]))
        console.print(table)
        typer.echo(f"report: {report_path}")
    finally:
        db.close()


@app.command("report")
def cmd_report(
    run_id: str | None = typer.Option(None, "--run-id", help="默认最近一次 run"),
    kind: str = typer.Option("both", "--kind", help="exec / trades / both（仅 paper run 有效）"),
) -> None:
    """重新生成 + 查看某次 run 的报告（paper → 双报告；backtest → 回放报告）。"""
    from rich.markdown import Markdown  # noqa: PLC0415

    from .persistence import get_run, latest_run_id, set_run_report_dir  # noqa: PLC0415
    from .reporting import (  # noqa: PLC0415
        build_backtest_report,
        build_execution_report,
        build_trades_report,
        generate_run_reports,
    )

    if kind not in ("exec", "trades", "both"):
        typer.echo("✘ --kind 必须为 exec / trades / both")
        raise typer.Exit(2)

    db = Database(paths.db_path())
    try:
        rid = run_id or latest_run_id(db)
        if rid is None:
            typer.echo("(no runs)")
            raise typer.Exit(2)
        run = get_run(db, rid)
        if run is None:
            typer.echo(f"✘ run_id 不存在: {rid!r}")
            raise typer.Exit(2)

        console = Console()
        if run["mode"] == "backtest":
            console.print(Markdown(build_backtest_report(db, rid)))
            return
        # paper：重新生成双报告落盘（幂等），再按 --kind 打印
        report_dir = generate_run_reports(db, rid)
        set_run_report_dir(db, rid, str(report_dir))
        if kind in ("exec", "both"):
            console.print(Markdown(build_execution_report(db, rid)))
        if kind in ("trades", "both"):
            console.print(Markdown(build_trades_report(db, rid)))
        typer.echo(f"\nreport dir: {report_dir}")
    finally:
        db.close()


@app.command("history")
def cmd_history(
    code: str | None = typer.Option(None, "--code", help="按标的过滤"),
    mode: str | None = typer.Option(None, "--mode", help="paper / backtest"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List recent runs of this plugin."""
    db = Database(paths.db_path())
    try:
        clauses: list[str] = []
        params: list[object] = []
        if code:
            clauses.append("code = ?")
            params.append(code)
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        rows = db.fetchall(
            "SELECT run_id, mode, code, trade_date, status, started_at, finished_at "
            f"FROM vwr_runs {where}ORDER BY started_at DESC LIMIT ?",
            (*params, limit),
        )
    finally:
        db.close()
    if not rows:
        typer.echo("(no runs)")
        return
    for r in rows:
        typer.echo(
            f"{r[0]}  {r[1]:<8}  {r[2]:<10}  {r[3]:<10}  {r[4]:<8}  {r[5]} → {r[6] or '-'}"
        )


# ---------------------------------------------------------------------------
# settings — plugin-local persisted knobs
# ---------------------------------------------------------------------------


@settings_app.command("show")
def cmd_settings_show() -> None:
    """展示当前生效的设置（来源 = persisted / default）。"""
    db = Database(paths.db_path())
    try:
        rows = list_for_show(db)
    finally:
        db.close()
    console = Console()
    table = Table(title="vwap-reversion settings")
    table.add_column("Key", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_column("Source", style="yellow")
    for key, value, source in rows:
        table.add_row(key, "" if value is None else str(value), source)
    console.print(table)


@settings_app.command("set")
def cmd_settings_set(
    key: str = typer.Argument(..., help="配置项名（不含 vwr. 前缀），如 band_k_entry"),
    value: str = typer.Argument(..., help="新值（JSON 或裸字符串），如 2.5 / false / 14:50"),
) -> None:
    """设置单个配置项并持久化（整体校验后写入）。"""
    db = Database(paths.db_path())
    try:
        try:
            new_cfg = set_one(db, key, value)
        except ValueError as e:
            typer.echo(f"✘ {e}")
            raise typer.Exit(2) from e
        typer.echo(f"✔ vwr.{key} = {getattr(new_cfg, key)!r}")
    finally:
        db.close()


@settings_app.command("reset")
def cmd_settings_reset(
    yes: bool = typer.Option(False, "--yes", help="跳过确认"),
) -> None:
    """清空所有持久化设置，回到 dataclass 默认值。"""
    if not yes and not typer.confirm("确认清空 vwap-reversion 全部持久化设置？"):
        raise typer.Exit(1)
    db = Database(paths.db_path())
    try:
        reset_config(db)
    finally:
        db.close()
    typer.echo("✔ 已重置为默认值")


def main(argv: list[str]) -> int:
    """Plugin's dispatch entry. Returns exit code（与 limit_up_board.cli.main 同款契约）。"""
    # 确保框架 logger 有 handler（幂等；框架定义 setup_logging 但不主动调用）。
    try:
        from deeptrade.core.logging_config import setup_logging  # noqa: PLC0415

        setup_logging()
    except Exception:  # noqa: BLE001 — logging setup never blocks a run
        pass

    try:
        # click 在 standalone_mode=False 下会把命令里 raise 的 typer.Exit 捕获并
        # 把 exit_code 作为返回值（而不是继续抛）—— 必须接住返回值，否则
        # `raise typer.Exit(2)` 的路径会被静默成 exit 0。
        result = app(argv, standalone_mode=False)
        return int(result) if isinstance(result, int) else 0
    except typer.Exit as e:
        return int(e.exit_code or 0)
    except SystemExit as e:
        try:
            return int(e.code or 0)
        except (TypeError, ValueError):
            return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n⏹ 用户手动中断，已停止当前策略执行。\n")
        return 130
    except Exception as e:  # noqa: BLE001 — reflect to framework as exit 1
        sys.stderr.write(render_exception(e) + "\n")
        return 1
