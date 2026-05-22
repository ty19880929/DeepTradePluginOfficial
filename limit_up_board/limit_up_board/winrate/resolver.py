"""T+1 行情解析 + 胜负判定。

PR #2 — 从 ``lub_prediction_records`` 取出一组预测记录后，逐股解析其
``next_trade_date`` 对应的开盘价并判定 outcome。

行情来源回退链：
    1) ``lub_daily``    — 命中即用
    2) tushare ``daily`` — 仅当 ``--force-sync`` 启用时回源（避免默认走 summary
                          就静默触发网络调用）
    3) 仍缺失           — outcome 标记为 ``unresolved``

胜负函数： T+1 open vs T close (=涨停价口径，PR #1 写入字段 ``t_close_price``)。
    open > close → win
    open = close → flat
    open < close → loss
    任一缺失     → unresolved
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database
    from deeptrade.core.tushare_client import TushareClient

    from .persistence import PredictionRecord

logger = logging.getLogger(__name__)

Outcome = Literal["win", "flat", "loss", "unresolved"]


@dataclass(frozen=True)
class ResolvedRecord:
    """One ``PredictionRecord`` with T+1 open + outcome attached."""

    record: PredictionRecord
    t1_open_price: float | None
    open_vs_limit_pct: float | None
    outcome: Outcome


# ---------------------------------------------------------------------------
# T+1 行情查询
# ---------------------------------------------------------------------------


def _lookup_lub_daily(db: Database, ts_code: str, trade_date: str) -> float | None:
    """Return the ``open`` price for (ts_code, trade_date) from ``lub_daily``,
    or None when no row exists."""
    row = db.fetchone(
        "SELECT open FROM lub_daily WHERE ts_code=? AND trade_date=?",
        (ts_code, trade_date),
    )
    if row is None:
        return None
    return float(row[0]) if row[0] is not None else None


def _fetch_via_tushare(
    tushare: TushareClient, ts_code: str, trade_date: str
) -> float | None:
    """One-shot fallback fetch via tushare ``daily``. Writes nothing back —
    the framework's TushareClient cache layer handles row persistence."""
    try:
        df = tushare.call(
            "daily",
            params={"ts_code": ts_code, "start_date": trade_date, "end_date": trade_date},
            force_sync=True,
        )
    except Exception:  # noqa: BLE001 — degrade to unresolved
        logger.warning("tushare daily fetch failed for %s @ %s", ts_code, trade_date, exc_info=True)
        return None
    if df is None or df.empty or "open" not in df.columns:
        return None
    val = df["open"].iloc[0]
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:  # noqa: PLR0124 — NaN
        return None
    return f


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


def classify_outcome(t1_open: float | None, t_close: float | None) -> Outcome:
    """T+1 open vs T close. Either missing → unresolved."""
    if t1_open is None or t_close is None:
        return "unresolved"
    if t1_open > t_close:
        return "win"
    if t1_open == t_close:
        return "flat"
    return "loss"


def _pct(t1_open: float | None, t_close: float | None) -> float | None:
    if t1_open is None or t_close is None or t_close == 0:
        return None
    return (t1_open / t_close - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_records(
    records: list[PredictionRecord],
    *,
    db: Database,
    tushare: TushareClient | None = None,
    force_sync: bool = False,
) -> list[ResolvedRecord]:
    """Resolve every record's T+1 outcome.

    ``force_sync`` 仅在 ``tushare`` 也非 None 时生效——决定是否在 ``lub_daily``
    miss 时尝试 tushare 回源。默认 False，等价于"只读本地缓存"，保证
    ``summary`` 默认不会静默拉网络。
    """
    out: list[ResolvedRecord] = []
    for rec in records:
        t1 = _lookup_lub_daily(db, rec.ts_code, rec.next_trade_date)
        if t1 is None and force_sync and tushare is not None:
            t1 = _fetch_via_tushare(tushare, rec.ts_code, rec.next_trade_date)
            # NOTE: TushareClient 自己负责把行落 lub_daily，所以这里不重复 insert。

        outcome = classify_outcome(t1, rec.t_close_price)
        pct = _pct(t1, rec.t_close_price)
        out.append(
            ResolvedRecord(
                record=rec,
                t1_open_price=t1,
                open_vs_limit_pct=pct,
                outcome=outcome,
            )
        )
    return out
