"""reporting — 收盘双报告 + backtest 报告（设计 §12）.

* :func:`build_execution_report` — 执行报告（采样质量/信号统计/异常降级）
* :func:`build_trades_report`    — 交易汇总报告（汇总指标 + 成交明细）
* :func:`build_backtest_report`  — 回放报告（逐日 + 聚合）
* :func:`generate_run_reports`   — daemon 收盘调用：写双 md 落盘，返回目录
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..paths import daily_report_dir
from .execution_report import build_execution_report
from .trades_report import build_trades_report
from .backtest_report import build_backtest_report

if TYPE_CHECKING:  # pragma: no cover
    from deeptrade.core.db import Database

__all__ = [
    "build_execution_report",
    "build_trades_report",
    "build_backtest_report",
    "generate_run_reports",
]


def generate_run_reports(db: Database, run_id: str) -> Path:
    """生成 paper run 的双报告并落盘，返回报告目录（设计 §12）。

    幂等：同 run 重复调用直接覆盖（``report`` 子命令复用本函数重新生成）。
    """
    from ..persistence import get_run  # noqa: PLC0415

    run = get_run(db, run_id)
    if run is None:
        raise ValueError(f"run_id 不存在: {run_id!r}")
    out_dir = daily_report_dir(str(run["code"]), str(run["trade_date"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "execution_report.md").write_text(
        build_execution_report(db, run_id), encoding="utf-8"
    )
    (out_dir / "trades_report.md").write_text(
        build_trades_report(db, run_id), encoding="utf-8"
    )
    return out_dir
