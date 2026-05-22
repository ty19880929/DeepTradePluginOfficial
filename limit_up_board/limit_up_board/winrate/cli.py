"""``deeptrade limit-up-board winrate`` Typer 子应用。

PR #2 — ``summary``。
PR #3 — ``export`` (JSON / CSV) + ``purge`` (--yes 二次确认)。
PR #4 追加 ``llm-review``。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

logger = logging.getLogger(__name__)

winrate_app = typer.Typer(
    name="winrate",
    help=(
        "胜率分析 — 基于 lub_prediction_records 中的 T 日预测，"
        "即时解析 T+1 开盘价并出胜负摘要。"
    ),
    no_args_is_help=True,
    add_completion=False,
)


# 强制多 subcommand 分组 — Typer 在只有一个 ``@app.command`` 时会把 subapp
# 自动扁平化（直接把 subapp 当成单命令），导致 PR #2 阶段只能用 `winrate ...`
# 而不能用 `winrate summary ...`。PR #3 / #4 加 export / purge / llm-review
# 后此问题自然消失，但为保证 CLI 表面在所有 PR 间稳定，这里显式声明 callback。
@winrate_app.callback()
def _winrate_callback() -> None:
    """胜率分析子命令组。"""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


MAX_WINDOW_TRADE_DAYS = 10  # PR 实施计划修订项 #3 — 区间硬上限
VALID_PREDICTIONS = {"top_candidate", "watchlist", "avoid"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WinrateError(RuntimeError):
    """User-facing error rendered as ``✘ {message}`` without traceback."""


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    is_default: bool


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _trade_days_in_range(db: Database, start: str, end: str) -> list[str]:
    """Return distinct open trade dates in [start, end] from ``lub_trade_cal``.

    Empty list when the calendar table is empty (fresh install) — caller falls
    back to ``lub_prediction_records.trade_date`` distinct values.
    """
    rows = db.fetchall(
        "SELECT DISTINCT cal_date FROM lub_trade_cal "
        "WHERE is_open=1 AND cal_date>=? AND cal_date<=? ORDER BY cal_date ASC",
        (start, end),
    )
    return [str(r[0]) for r in rows]


def _record_trade_days_in_range(db: Database, start: str, end: str) -> list[str]:
    rows = db.fetchall(
        "SELECT DISTINCT trade_date FROM lub_prediction_records "
        "WHERE trade_date>=? AND trade_date<=? ORDER BY trade_date ASC",
        (start, end),
    )
    return [str(r[0]) for r in rows]


def _resolve_default_window(db: Database) -> Window:
    """Default window = T-1 (latest open trade date strictly before today).

    Resolution order:
      1) lub_trade_cal: latest cal_date with is_open=1 AND cal_date < today_local
      2) Fallback (cal empty): max(trade_date) from lub_prediction_records
      3) Both empty: WinrateError
    """
    today = _today_str()
    row = db.fetchone(
        "SELECT MAX(cal_date) FROM lub_trade_cal WHERE is_open=1 AND cal_date < ?",
        (today,),
    )
    if row is not None and row[0]:
        d = str(row[0])
        return Window(start=d, end=d, is_default=True)
    # Fallback: latest record's trade_date
    row2 = db.fetchone("SELECT MAX(trade_date) FROM lub_prediction_records")
    if row2 is not None and row2[0]:
        d = str(row2[0])
        return Window(start=d, end=d, is_default=True)
    raise WinrateError(
        "无可用的交易日历或预测记录；请先跑一次 `deeptrade limit-up-board run` "
        "（单 LLM 模式下会自动落预测记录）。"
    )


def resolve_window(
    db: Database, start: str | None, end: str | None
) -> Window:
    """Resolve the analysis window.

    - 都为空 → 默认 T-1 单日
    - 只给一个 → 报错
    - 都给 → 校验 [start, end] 内交易日数 ≤ MAX_WINDOW_TRADE_DAYS
    """
    if start is None and end is None:
        return _resolve_default_window(db)
    if start is None or end is None:
        raise WinrateError("--start 和 --end 必须同时指定")
    if start > end:
        raise WinrateError(f"--start ({start}) 必须 ≤ --end ({end})")

    trade_days = _trade_days_in_range(db, start, end)
    if not trade_days:
        # Calendar absent — fall back to record-level distinct dates for a
        # weaker but functional cap check
        trade_days = _record_trade_days_in_range(db, start, end)

    if len(trade_days) > MAX_WINDOW_TRADE_DAYS:
        raise WinrateError(
            f"winrate 区间最长 {MAX_WINDOW_TRADE_DAYS} 个交易日；"
            f"当前 {start}..{end} = {len(trade_days)} 个交易日。"
            f"请缩短区间，或分多次执行。"
        )
    return Window(start=start, end=end, is_default=False)


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "--"
    return f"{v * 100:.1f}%"


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "--"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def render_summary(
    console: Console,
    *,
    window: Window,
    summary,
    by_prediction,
    by_rank,
    unresolved_breakdown: dict[str, int],
) -> None:
    """Print the terminal summary in the style of 方案 §7.1."""
    from . import stats as _  # noqa: F401  — type-only ref kept implicit

    title = f"胜率分析 · {window.start}..{window.end}"
    if window.is_default:
        title += "  (T-1)"
    console.print()
    console.print(f"[bold]{title}[/bold]")
    console.print()

    console.print("[bold]样本:[/bold]")
    console.print(f"  预测记录: {summary.total} 只")
    console.print(f"  已解析:   {summary.resolved} 只")
    console.print(f"  待解析:   {summary.unresolved} 只")
    console.print()

    console.print("[bold]整体:[/bold]")
    console.print(f"  胜: {summary.win}  平: {summary.flat}  负: {summary.loss}")
    console.print(f"  严格胜率: {_fmt_pct(summary.strict_win_rate)}")
    console.print(f"  非亏比例: {_fmt_pct(summary.non_loss_rate)}")
    console.print(f"  平均开盘溢价: {_fmt_delta(summary.avg_open_vs_limit_pct)}")
    console.print()

    if by_prediction:
        tbl = Table(title="按预测类型", show_header=True, header_style="bold")
        tbl.add_column("prediction")
        tbl.add_column("胜/已解析", justify="right")
        tbl.add_column("胜率", justify="right")
        tbl.add_column("平均开盘溢价", justify="right")
        for g in by_prediction:
            tbl.add_row(
                g.key,
                f"{g.win}/{g.resolved}",
                _fmt_pct(g.strict_win_rate),
                _fmt_delta(g.avg_open_vs_limit_pct),
            )
        console.print(tbl)
        console.print()

    if by_rank:
        tbl = Table(title="按 rank 分桶", show_header=True, header_style="bold")
        tbl.add_column("rank bucket")
        tbl.add_column("胜/已解析", justify="right")
        tbl.add_column("胜率", justify="right")
        tbl.add_column("平均开盘溢价", justify="right")
        for g in by_rank:
            tbl.add_row(
                g.key,
                f"{g.win}/{g.resolved}",
                _fmt_pct(g.strict_win_rate),
                _fmt_delta(g.avg_open_vs_limit_pct),
            )
        console.print(tbl)
        console.print()

    if unresolved_breakdown:
        console.print("[bold]待解析:[/bold]")
        for d, n in sorted(unresolved_breakdown.items()):
            console.print(f"  {d}: {n} 只，T+1 行情尚不可用或未同步")
        console.print()


def _unresolved_breakdown(resolved_list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in resolved_list:
        if r.outcome == "unresolved":
            d = r.record.trade_date
            out[d] = out.get(d, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Shared pipeline: window → records → resolved → (summary, by_pred, by_rank)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pipeline:
    """Hold all derived state for one `summary` / `export` / `llm-review`
    invocation. Reused so each subcommand shares one resolver pass."""

    window: Window
    resolved: list  # list[ResolvedRecord] — fwd-decl to avoid late import here
    summary: object  # WinrateSummary
    by_prediction: list  # list[GroupStat]
    by_rank: list  # list[GroupStat]
    unresolved_breakdown: dict[str, int]


def _pre_validate_predictions(prediction: list[str]) -> list[str] | None:
    if not prediction:
        return None
    unknown = [p for p in prediction if p not in VALID_PREDICTIONS]
    if unknown:
        typer.echo(
            f"✘ --prediction 不识别: {', '.join(unknown)}；"
            f"允许: {', '.join(sorted(VALID_PREDICTIONS))}"
        )
        raise typer.Exit(2)
    return prediction


def _build_pipeline(
    db: Database,
    rt,
    *,
    start: str | None,
    end: str | None,
    prediction: list[str],
    force_sync: bool,
) -> _Pipeline:
    """Resolve window + records + outcomes + stats. Common path for summary,
    export, llm-review."""
    from .persistence import load_prediction_records  # noqa: PLC0415
    from .resolver import resolve_records  # noqa: PLC0415
    from .stats import group_by_prediction, group_by_rank_bucket, summarize  # noqa: PLC0415

    pred_filter = _pre_validate_predictions(prediction)

    try:
        window = resolve_window(db, start, end)
    except WinrateError as e:
        typer.echo(f"✘ {e}")
        raise typer.Exit(2) from e

    records = load_prediction_records(
        db, start=window.start, end=window.end, predictions=pred_filter,
    )

    tushare_client = None
    if force_sync:
        try:
            from ..runtime import build_tushare_client  # noqa: PLC0415
            tushare_client = build_tushare_client(rt)
        except Exception as exc:  # noqa: BLE001 — degrade to local-only
            typer.echo(f"warning: tushare 不可用，--force-sync 将退化为本地查询: {exc}")

    resolved = resolve_records(
        records, db=db, tushare=tushare_client, force_sync=force_sync,
    )

    return _Pipeline(
        window=window,
        resolved=resolved,
        summary=summarize(resolved),
        by_prediction=group_by_prediction(resolved),
        by_rank=group_by_rank_bucket(resolved),
        unresolved_breakdown=_unresolved_breakdown(resolved),
    )


# ---------------------------------------------------------------------------
# `summary` subcommand
# ---------------------------------------------------------------------------


@winrate_app.command("summary")
def cmd_summary(
    start: str | None = typer.Option(None, "--start", help="统计起始 T 日 YYYYMMDD"),
    end: str | None = typer.Option(None, "--end", help="统计结束 T 日 YYYYMMDD"),
    prediction: list[str] = typer.Option(
        [], "--prediction",
        help="过滤预测类型；可重复，例如 --prediction top_candidate --prediction watchlist",
    ),
    force_sync: bool = typer.Option(
        False, "--force-sync",
        help="本地 lub_daily 缺 T+1 行情时尝试 tushare 回源",
    ),
) -> None:
    """终端输出胜率摘要 + 按 prediction / rank 分组。

    不传 --start/--end 时默认仅分析 T-1 日；显式指定区间时最长 10 个交易日。
    """
    from ..cli import _open_runtime  # noqa: PLC0415

    db, rt = _open_runtime()
    try:
        pipe = _build_pipeline(
            db, rt, start=start, end=end, prediction=prediction, force_sync=force_sync,
        )
        console = Console()
        render_summary(
            console,
            window=pipe.window,
            summary=pipe.summary,
            by_prediction=pipe.by_prediction,
            by_rank=pipe.by_rank,
            unresolved_breakdown=pipe.unresolved_breakdown,
        )
        if pipe.summary.total == 0:
            typer.echo("（窗口内无预测记录）")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# `export` subcommand
# ---------------------------------------------------------------------------


@winrate_app.command("export")
def cmd_export(
    output: str = typer.Option(..., "--output", help="输出文件路径"),
    fmt: str | None = typer.Option(
        None, "--format",
        help="输出格式 json/csv；不指定时按 --output 扩展名推断（默认 json）",
    ),
    start: str | None = typer.Option(None, "--start", help="统计起始 T 日 YYYYMMDD"),
    end: str | None = typer.Option(None, "--end", help="统计结束 T 日 YYYYMMDD"),
    prediction: list[str] = typer.Option([], "--prediction"),
    force_sync: bool = typer.Option(False, "--force-sync"),
) -> None:
    """终端输出摘要，并把逐股明细写入文件（JSON / CSV）。

    与 ``summary`` 共用 window 解析与 T+1 行情回退逻辑；终端摘要总是会打印，
    文件写入失败也不影响摘要展示，但会以非零退出码退出。
    """
    from ..cli import _open_runtime  # noqa: PLC0415
    from .exporter import build_payload, infer_format, write_to_disk  # noqa: PLC0415

    try:
        resolved_fmt = infer_format(output, fmt)
    except ValueError as e:
        typer.echo(f"✘ {e}")
        raise typer.Exit(2) from e

    db, rt = _open_runtime()
    try:
        pipe = _build_pipeline(
            db, rt, start=start, end=end, prediction=prediction, force_sync=force_sync,
        )
        # Terminal summary first — file IO is the optional add-on, never block
        # the read-only summary on disk problems.
        console = Console()
        render_summary(
            console,
            window=pipe.window,
            summary=pipe.summary,
            by_prediction=pipe.by_prediction,
            by_rank=pipe.by_rank,
            unresolved_breakdown=pipe.unresolved_breakdown,
        )

        payload = build_payload(
            window_start=pipe.window.start,
            window_end=pipe.window.end,
            summary=pipe.summary,
            by_prediction=pipe.by_prediction,
            resolved=pipe.resolved,
        )

        try:
            # Create parent dir if needed (parents=True is friendly for
            # "reports/2026/winrate.json" style paths).
            out_path = Path(output)
            if out_path.parent and not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
            write_to_disk(payload, output, resolved_fmt)
        except OSError as e:
            typer.echo(f"✘ 写文件失败: {e}")
            raise typer.Exit(1) from e

        typer.echo(f"\n详细结果已写入: {Path(output).resolve()}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# `purge` subcommand
# ---------------------------------------------------------------------------


@winrate_app.command("purge")
def cmd_purge(
    before: str = typer.Option(
        ..., "--before",
        help="删除 trade_date <= before 的所有预测记录（YYYYMMDD）",
    ),
    yes: bool = typer.Option(
        False, "--yes",
        help="跳过交互确认；非 TTY 模式下必须传，否则报错退出",
    ),
) -> None:
    """删除指定日期及之前的预测留痕（高危：不可撤销）。"""
    from ..cli import _open_runtime  # noqa: PLC0415
    from .persistence import purge_prediction_records  # noqa: PLC0415

    if not (len(before) == 8 and before.isdigit()):
        typer.echo(f"✘ --before 必须是 YYYYMMDD 格式，收到: {before}")
        raise typer.Exit(2)

    db, _ = _open_runtime()
    try:
        # 1) 先 count + 列样本日期
        row = db.fetchone(
            "SELECT COUNT(*) FROM lub_prediction_records WHERE trade_date <= ?",
            (before,),
        )
        n = int(row[0]) if row else 0
        if n == 0:
            typer.echo(f"（trade_date <= {before} 无记录可删除）")
            return

        date_rows = db.fetchall(
            "SELECT DISTINCT trade_date FROM lub_prediction_records "
            "WHERE trade_date <= ? ORDER BY trade_date ASC",
            (before,),
        )
        dates = [r[0] for r in date_rows]
        typer.echo(f"将删除 {n} 条预测记录，覆盖 {len(dates)} 个 trade_date:")
        for d in dates[:5]:
            typer.echo(f"  - {d}")
        if len(dates) > 5:
            typer.echo(f"  ...（共 {len(dates)} 个日期）")

        # 2) 二次确认
        if not yes:
            if not sys.stdin.isatty():
                typer.echo("✘ 高危操作未确认；非交互终端请显式传 --yes")
                raise typer.Exit(2)
            ok = typer.confirm("确认删除？", default=False)
            if not ok:
                typer.echo("已取消")
                raise typer.Exit(1)

        # 3) 真删
        deleted = purge_prediction_records(db, before=before)
        typer.echo(f"已删除 {deleted} 条预测记录。")
    finally:
        db.close()
