"""Plugin-managed CLI for market-review (PR-6 — full implementation).

Subcommands (design §7):

- ``run``      — full pipeline (Steps 0..5)
- ``sync``     — data-only path (Steps 0..1; no metrics / LLM / upload)
- ``history``  — list recent runs
- ``report``   — re-render a finished run's terminal summary from disk
- ``settings`` — show / set persisted :class:`MrConfig`

Invoked via the framework's pure pass-through dispatch:
    deeptrade market-review <subcommand> [...]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .config import MrConfig
from .render import render_terminal_summary
from .runner import MrRunner, PreconditionError, RunOutcome, RunParams
from .runtime import MrRuntime, build_tushare_client

app = typer.Typer(
    name="market-review",
    help=(
        "市场复盘 — A 股单日 / 区间复盘：板块轮动 / 情绪 / 资金 / 龙头 / "
        "风格 / 风险 / 展望 七章节 LLM 综合报告。"
    ),
    no_args_is_help=True,
    add_completion=False,
)

settings_app = typer.Typer(
    name="settings",
    help="本插件可持久化的运行参数（窗口上限、情绪权重、板块口径、LLM provider 等）。",
    no_args_is_help=False,
    add_completion=False,
    invoke_without_command=True,
)
app.add_typer(settings_app, name="settings")


# ---------------------------------------------------------------------------
# Runtime / context helpers
# ---------------------------------------------------------------------------


def _open_runtime() -> tuple[object, MrRuntime, object]:
    """Build the per-process services bundle.

    Returns ``(db, rt, ctx)`` where ``ctx`` is the framework
    :class:`PluginContext` (or ``None`` on older frameworks that don't
    expose it). Tests monkey-patch this function with a fake to bypass
    the real framework wiring entirely.
    """
    from deeptrade.core import paths  # noqa: PLC0415
    from deeptrade.core.config import ConfigService  # noqa: PLC0415
    from deeptrade.core.db import Database  # noqa: PLC0415
    from deeptrade.core.llm_manager import LLMManager  # noqa: PLC0415

    db = Database(paths.db_path())
    cfg = ConfigService(db)
    try:
        from deeptrade.plugins_api import PluginContext  # noqa: PLC0415
        ctx = PluginContext(db=db, config=cfg, plugin_id="market-review")
    except Exception:  # noqa: BLE001 — pre-v0.11 framework path
        ctx = None
    rt = MrRuntime(db=db, config=cfg, llms=LLMManager(db, cfg))
    try:
        rt.tushare = build_tushare_client(rt)
    except RuntimeError:
        # tushare.token missing — keep ``rt.tushare = None``. ``run`` / ``sync``
        # will raise PreconditionError at Step 0 with a clearer message.
        pass
    return db, rt, ctx


def _close_db(db) -> None:
    """Best-effort DB close — never raises."""
    try:
        db.close()
    except Exception:  # noqa: BLE001
        pass


def _reports_root() -> Path:
    """Default reports directory — kept as a function (not a constant) so
    :class:`MrRunner` and :func:`cmd_report` agree on the location AND tests
    can monkey-patch one symbol to redirect both to a tmp path."""
    return Path.home() / ".deeptrade" / "reports"


# ---------------------------------------------------------------------------
# run / sync
# ---------------------------------------------------------------------------


@app.command("run", help="单日 / 区间复盘的完整流水线（Step 0..5）。")
def cmd_run(
    trade_date: str | None = typer.Option(
        None, "--trade-date", help="YYYYMMDD；与 --start/--end 互斥。"
    ),
    start: str | None = typer.Option(None, "--start", help="区间起点 YYYYMMDD。"),
    end: str | None = typer.Option(None, "--end", help="区间末日 YYYYMMDD。"),
    force_sync: bool = typer.Option(False, "--force-sync"),
    llm: str | None = typer.Option(
        None, "--llm",
        help="本次 run 使用的 LLM provider（覆盖框架默认）。",
    ),
    no_upload: bool = typer.Option(
        False, "--no-upload",
        help="跳过 summary.json 上传步骤；离线 / 调试时有用。",
    ),
) -> None:
    params = RunParams(
        trade_date=trade_date, start=start, end=end,
        force_sync=force_sync, llm_provider=llm, no_upload=no_upload,
    )
    db, rt, ctx = _open_runtime()
    try:
        outcome = _run_with_runner(rt, ctx, params, full=True)
        _print_terminal_summary(outcome)
    finally:
        _close_db(db)
    raise typer.Exit(_exit_code_for(outcome))


@app.command("sync", help="仅落 Tushare 数据，不调 LLM（Step 0..1）。")
def cmd_sync(
    trade_date: str | None = typer.Option(None, "--trade-date"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    force_sync: bool = typer.Option(False, "--force-sync"),
) -> None:
    params = RunParams(
        trade_date=trade_date, start=start, end=end,
        force_sync=force_sync, no_llm=True, no_upload=True,
    )
    db, rt, ctx = _open_runtime()
    try:
        outcome = _run_with_runner(rt, ctx, params, full=False)
        console = Console()
        if outcome.status == "success":
            console.print(
                f"[green]✓[/] sync 完成 · run_id=[cyan]{outcome.run_id}[/] · "
                f"reports={outcome.report_dir}"
            )
        else:
            console.print(
                f"[red]✘[/] sync 失败 · run_id={outcome.run_id} · "
                f"error={outcome.error or '未知'}"
            )
    finally:
        _close_db(db)
    raise typer.Exit(_exit_code_for(outcome))


def _run_with_runner(
    rt: MrRuntime, ctx, params: RunParams, *, full: bool,
) -> RunOutcome:
    """Dispatch to the appropriate runner method.

    :class:`PreconditionError` (user-facing input errors like bad window
    specs) is NOT caught here — it propagates up to :func:`main` which
    renders it + exits 2. System errors are already captured by the runner
    into a ``failed`` :class:`RunOutcome`.
    """
    runner = MrRunner(rt, ctx=ctx)
    if full:
        return runner.execute(params)
    return runner.execute_sync_only(params)


def _exit_code_for(outcome: RunOutcome) -> int:
    if outcome.status == "success":
        return 0
    if outcome.status == "partial_failed":
        return 0  # partial still counts as success for shell semantics
    return 1


def _print_terminal_summary(outcome: RunOutcome) -> None:
    """Print the post-run summary to stdout (design §5.5.3)."""
    console = Console()
    if outcome.status == "failed" and not outcome.run_id:
        # PreconditionError path — error already printed to stderr.
        return
    if outcome.status == "failed":
        console.print(
            f"[red]✘[/] run 失败 · run_id={outcome.run_id} · "
            f"error={outcome.error or '未知'} · reports={outcome.report_dir}"
        )
        return
    summary_json = outcome.report_dir / "summary.json"
    if not summary_json.is_file():
        console.print(
            f"[yellow]⚠[/] summary.json 缺失，跳过终端摘要 · run_id={outcome.run_id}"
        )
        return
    try:
        from .report.schema import ReviewReportSchema  # noqa: PLC0415
        report = ReviewReportSchema.model_validate_json(
            summary_json.read_text(encoding="utf-8")
        )
        console.print(Markdown(render_terminal_summary(report)))
        console.print(f"\n📁 完整报告：[cyan]{outcome.report_dir}[/]")
    except Exception as exc:  # noqa: BLE001 — terminal summary failures shouldn't crash CLI
        console.print(
            f"[yellow]⚠[/] 终端摘要渲染失败 ({type(exc).__name__}: {exc})，"
            f"完整报告在 {outcome.report_dir}"
        )


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


@app.command("history", help="列出最近的复盘 run（按开始时间倒序）。")
def cmd_history(
    limit: int = typer.Option(20, "--limit", min=1, max=200),
    mode: str | None = typer.Option(
        None, "--mode",
        help="过滤模式：day | range；省略 = 全部。",
    ),
) -> None:
    db, rt, _ = _open_runtime()
    try:
        sql = (
            "SELECT run_id, mode, start_date, end_date, anchor, status, "
            "started_at, finished_at FROM mr_runs"
        )
        params: list = []
        if mode:
            sql += " WHERE mode = ?"
            params.append(mode)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = db.fetchall(sql, params)
    finally:
        _close_db(db)

    console = Console()
    if not rows:
        console.print("[yellow]暂无 run 历史。[/]")
        raise typer.Exit(0)
    table = Table(title="市场复盘历史")
    for col in ("run_id", "mode", "anchor", "status", "started_at", "finished_at"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r[0])[:8] + "…",  # truncate UUID prefix
            str(r[1]),
            str(r[4]),
            _status_styled(str(r[5])),
            _fmt_ts(r[6]),
            _fmt_ts(r[7]),
        )
    console.print(table)
    raise typer.Exit(0)


def _status_styled(status: str) -> str:
    color = {
        "success": "green", "partial_failed": "yellow",
        "failed": "red", "cancelled": "blue", "running": "cyan",
    }.get(status, "white")
    return f"[{color}]{status}[/]"


def _fmt_ts(value) -> str:
    if value is None:
        return "—"
    return str(value)[:19]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command("report", help="重渲历史 run 的终端摘要 / 完整 markdown。")
def cmd_report(
    run_id: str = typer.Argument(..., help="run_id（可填 UUID 前缀，至少 6 位）"),
    full: bool = typer.Option(False, "--full", help="打印完整 summary.md，不只摘要。"),
    section: str | None = typer.Option(
        None, "--section",
        help="仅打印某 section markdown（overview / sectors / sentiment / ...）",
    ),
) -> None:
    db, rt, _ = _open_runtime()
    try:
        full_run_id = _resolve_run_id_prefix(db, run_id)
        report_dir = _reports_root() / full_run_id
    finally:
        _close_db(db)

    console = Console()
    if section:
        section_path = report_dir / f"{section}.md"
        if not section_path.is_file():
            console.print(f"[red]✘[/] 找不到 {section_path}")
            raise typer.Exit(1)
        console.print(Markdown(section_path.read_text(encoding="utf-8")))
        raise typer.Exit(0)

    if full:
        summary_md = report_dir / "summary.md"
        if not summary_md.is_file():
            console.print(f"[red]✘[/] 找不到 {summary_md}")
            raise typer.Exit(1)
        console.print(Markdown(summary_md.read_text(encoding="utf-8")))
        raise typer.Exit(0)

    # Default: terminal summary from summary.json
    summary_json = report_dir / "summary.json"
    if not summary_json.is_file():
        console.print(f"[red]✘[/] 找不到 {summary_json}")
        raise typer.Exit(1)
    from .report.schema import ReviewReportSchema  # noqa: PLC0415
    report = ReviewReportSchema.model_validate_json(
        summary_json.read_text(encoding="utf-8")
    )
    console.print(Markdown(render_terminal_summary(report)))
    raise typer.Exit(0)


def _resolve_run_id_prefix(db, raw: str) -> str:
    """Map a (possibly partial) run_id to the full UUID stored in mr_runs."""
    if len(raw) < 6:
        raise PreconditionError(
            f"run_id 前缀至少 6 位，收到 {len(raw)} 位：{raw!r}"
        )
    rows = db.fetchall(
        # ``run_id`` is UUID-typed in mr_runs; DuckDB needs an explicit
        # cast for LIKE pattern matching (UUID has no ``~~`` overload).
        "SELECT run_id FROM mr_runs WHERE CAST(run_id AS VARCHAR) LIKE ? || '%' "
        "ORDER BY started_at DESC",
        [raw],
    )
    if not rows:
        raise PreconditionError(f"找不到 run_id 前缀匹配 {raw!r}")
    if len(rows) > 1:
        raise PreconditionError(
            f"run_id 前缀 {raw!r} 命中 {len(rows)} 条记录，请提供更长的前缀"
        )
    return str(rows[0][0])


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@settings_app.callback()
def settings_callback(ctx: typer.Context) -> None:
    """``settings`` with no subcommand → ``show``."""
    if ctx.invoked_subcommand is None:
        _settings_show()


@settings_app.command("show")
def cmd_settings_show() -> None:
    _settings_show()


def _settings_show() -> None:
    """Print the current :class:`MrConfig` (defaults + DB overrides if present)."""
    db, _, _ = _open_runtime()
    try:
        overrides = {}
        rows = db.fetchall("SELECT key, value_json FROM mr_config")
        for k, vj in rows:
            try:
                overrides[str(k)] = json.loads(vj) if vj else None
            except json.JSONDecodeError:
                overrides[str(k)] = vj
    finally:
        _close_db(db)

    cfg = MrConfig()
    base = asdict(cfg)
    # Apply overrides on top of defaults.
    for k, v in overrides.items():
        # Convention: key looks like "mr.<field>"; strip prefix for display.
        attr = k[3:] if k.startswith("mr.") else k
        if attr in base:
            base[attr] = v

    console = Console()
    table = Table(title="MrConfig")
    table.add_column("字段")
    table.add_column("值")
    table.add_column("来源")
    for field, value in base.items():
        source = "user" if f"mr.{field}" in overrides else "default"
        table.add_row(field, _format_config_value(value), source)
    console.print(table)
    raise typer.Exit(0)


def _format_config_value(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


@settings_app.command("set", help="持久化覆盖一个 MrConfig 字段（写入 mr_config）。")
def cmd_settings_set(
    key: str = typer.Argument(
        ..., help='字段名，如 "max_window_days" 或 "mr.max_window_days"。'
    ),
    value: str = typer.Argument(
        ..., help='值（JSON 形式；字符串 / 数字 / 布尔 / 列表 / 对象皆可）。'
    ),
) -> None:
    """Set a persisted override. Validation tier: cheap.

    - Key must name an existing :class:`MrConfig` field (with or without
      the ``mr.`` prefix).
    - Value is parsed via ``json.loads`` so the CLI accepts JSON syntax —
      bare strings need surrounding double quotes. We fall back to
      treating the raw arg as a string if JSON parse fails so the common
      ``settings set sector_provider ths`` case still works.
    - Type-check is light: ``bool`` field doesn't accept str, ``int`` field
      requires int, ``dict`` field requires JSON object. Business rules
      (sentiment_weights sum = 1, etc.) are enforced at use-sites, not here.
    """
    from dataclasses import fields  # noqa: PLC0415

    attr = key[3:] if key.startswith("mr.") else key
    known = {f.name: f for f in fields(MrConfig)}
    if attr not in known:
        raise PreconditionError(
            f"未知配置项：{key!r}；有效字段：{sorted(known)}"
        )

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Bare string fallback — saves the user from quoting "ths" / "dc".
        parsed = value

    _type_check_for_field(known[attr], parsed)

    db, _, _ = _open_runtime()
    try:
        db.execute(
            # DuckDB upsert. ``NOW()`` rather than bare ``CURRENT_TIMESTAMP``
            # in the UPDATE clause — DuckDB's parser binds the latter as a
            # column reference inside ``DO UPDATE SET`` and errors out.
            "INSERT INTO mr_config (key, value_json) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET "
            "value_json = excluded.value_json, updated_at = NOW()",
            [f"mr.{attr}", json.dumps(parsed, ensure_ascii=False)],
        )
    finally:
        _close_db(db)

    console = Console()
    console.print(
        f"[green]✓[/] 已保存 [cyan]mr.{attr}[/] = "
        f"{_format_config_value(parsed)}"
    )
    raise typer.Exit(0)


def _type_check_for_field(field_info, value) -> None:
    """Lightweight check — refuse the obvious mistakes.

    pydantic-style coercion would be heavier than warranted for v0.1; we
    just check the broad category matches.
    """
    expected = field_info.type
    name = field_info.name
    # ``field_info.type`` may be a string (``__future__ annotations``); use
    # the dataclass default as the truth source instead.
    sample = field_info.default if field_info.default is not MISSING else None
    if sample is not None:
        ok = isinstance(value, type(sample))
        # Bool is a subclass of int — refuse cross-typing both directions.
        if isinstance(sample, bool) and not isinstance(value, bool):
            ok = False
        if isinstance(value, bool) and not isinstance(sample, bool):
            ok = False
        if not ok:
            raise PreconditionError(
                f"{name!r} 期望类型 {type(sample).__name__}，"
                f"收到 {type(value).__name__}（值 {value!r}）"
            )


# ``dataclasses.MISSING`` sentinel — imported lazily to keep cli import tight.
from dataclasses import MISSING  # noqa: E402


@settings_app.command("reset", help="重置 MrConfig 覆盖（指定 key = 单字段；省略 = 全部）。")
def cmd_settings_reset(
    key: str | None = typer.Argument(
        None, help="字段名；省略 = 删除所有 mr_config 行。",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="不指定 key 时强制确认，跳过 prompt。",
    ),
) -> None:
    from dataclasses import fields  # noqa: PLC0415

    db, _, _ = _open_runtime()
    try:
        console = Console()
        if key:
            attr = key[3:] if key.startswith("mr.") else key
            known = {f.name for f in fields(MrConfig)}
            if attr not in known:
                raise PreconditionError(
                    f"未知配置项：{key!r}；有效字段：{sorted(known)}"
                )
            db.execute("DELETE FROM mr_config WHERE key = ?", [f"mr.{attr}"])
            console.print(f"[green]✓[/] 已重置 [cyan]mr.{attr}[/] 为默认值")
        else:
            if not yes:
                # Non-destructive default — show a count + how to confirm.
                row = db.fetchone("SELECT COUNT(*) FROM mr_config")
                count = int(row[0] or 0) if row else 0
                console.print(
                    f"[yellow]⚠[/] 即将清空 mr_config（当前 {count} 条覆盖）。"
                    "重跑命令并加 --yes 确认，或指定具体字段名只重置单项。"
                )
                raise typer.Exit(2)
            db.execute("DELETE FROM mr_config")
            console.print("[green]✓[/] 已重置 mr_config 全部覆盖")
    finally:
        _close_db(db)
    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """Framework dispatch entrypoint (signature stable since PR-1)."""
    try:
        rv = app(list(argv), standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    except click.exceptions.ClickException as exc:
        exc.show()
        return int(exc.exit_code or 1)
    except PreconditionError as exc:
        sys.stderr.write(f"✘ {exc}\n")
        return 2
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 1
    if isinstance(rv, int):
        return rv
    return 0
