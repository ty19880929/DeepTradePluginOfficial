"""``deeptrade limit-up-board winrate`` Typer 子应用。

PR #2 — 仅 ``summary`` 子命令。PR #3 追加 ``export`` / ``purge``，
PR #4 追加 ``llm-review``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
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
    # Late imports to keep `--help` snappy and isolate winrate deps.
    from ..cli import _open_runtime  # noqa: PLC0415
    from .persistence import load_prediction_records  # noqa: PLC0415
    from .resolver import resolve_records  # noqa: PLC0415
    from .stats import group_by_prediction, group_by_rank_bucket, summarize  # noqa: PLC0415

    # Pre-validate prediction filter values before opening DB.
    pred_filter: list[str] | None = None
    if prediction:
        unknown = [p for p in prediction if p not in VALID_PREDICTIONS]
        if unknown:
            typer.echo(
                f"✘ --prediction 不识别: {', '.join(unknown)}；"
                f"允许: {', '.join(sorted(VALID_PREDICTIONS))}"
            )
            raise typer.Exit(2)
        pred_filter = prediction

    db, rt = _open_runtime()
    try:
        try:
            window = resolve_window(db, start, end)
        except WinrateError as e:
            typer.echo(f"✘ {e}")
            raise typer.Exit(2) from e

        records = load_prediction_records(
            db, start=window.start, end=window.end, predictions=pred_filter,
        )

        # Tushare client is constructed lazily — only when the user opted into
        # network fallback. Avoids forcing a tushare token check on every
        # `summary` invocation.
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

        summary = summarize(resolved)
        by_prediction_stats = group_by_prediction(resolved)
        by_rank_stats = group_by_rank_bucket(resolved)
        unresolved_brk = _unresolved_breakdown(resolved)

        console = Console()
        render_summary(
            console,
            window=window,
            summary=summary,
            by_prediction=by_prediction_stats,
            by_rank=by_rank_stats,
            unresolved_breakdown=unresolved_brk,
        )

        if summary.total == 0:
            typer.echo("（窗口内无预测记录）")
    finally:
        db.close()
