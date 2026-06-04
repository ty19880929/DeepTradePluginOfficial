"""插件本地数据路径（仿 limit_up_board.lgb.paths 的同根目录约定）.

布局：
    ~/.deeptrade/vwap_reversion/
    └── reports/
        ├── <code>/<trade_date>/execution_report.md + trades_report.md   （paper 收盘双报告）
        └── <code>/backtest_<start>_<end>/backtest_report.md             （回放报告）

未来框架若暴露 ``paths.plugin_data_dir(plugin_id)``，切到该 API。
"""

from __future__ import annotations

from pathlib import Path

from deeptrade.core import paths


def plugin_data_dir() -> Path:
    """``~/.deeptrade/vwap_reversion/`` —— 插件持久化数据根目录。"""
    return paths.db_path().parent / "vwap_reversion"


def reports_root() -> Path:
    return plugin_data_dir() / "reports"


def daily_report_dir(code: str, trade_date: str) -> Path:
    """paper 收盘双报告目录（每日一目录，重跑覆盖）。"""
    return reports_root() / code / trade_date


def backtest_report_dir(code: str, start: str, end: str) -> Path:
    return reports_root() / code / f"backtest_{start}_{end}"
