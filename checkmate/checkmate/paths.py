"""Local filesystem layout for Checkmate.

Mirrors VA's ``lgb/paths.py`` convention — every persistent file lives under
``<paths.db_path().parent>/checkmate/`` so that uninstall can wipe a single
plugin tree without depending on framework support for per-plugin data dirs.

Layout::

    ~/.deeptrade/checkmate/
    ├── cache/                       # Iter-1+ Tushare 缓存
    │   ├── trade_cal.parquet
    │   ├── daily/                   # raw daily + adj_factor 合表，per ts_code
    │   │   └── <ts_code>.parquet
    │   ├── daily_basic/             # 成交额 / 换手率 / 总市值，per ts_code
    │   │   └── <ts_code>.parquet
    │   ├── stk_limit/               # 全市场涨跌停，per trade_date
    │   │   └── <YYYYMMDD>.parquet
    │   └── index_daily/             # 指数日线，per index_code
    │       └── <index_code>.parquet
    ├── backtests/                   # Iter-4 BacktestRunner 检查点根
    │   └── <config_hash>/days/<YYYYMMDD>.parquet
    └── reports/                     # Iter-4 report 子命令落盘根
        └── <run_id>.{json,md,html}

All helpers are pure (no I/O); call :func:`ensure_layout` once at plugin
init to materialise the directory tree if missing.
"""

from __future__ import annotations

from pathlib import Path


def _data_root() -> Path:
    # Lazy framework import so importing this module is cheap and safe in
    # unit-test environments that monkey-patch the data root.
    from deeptrade.core import paths  # noqa: PLC0415

    return paths.db_path().parent / "checkmate"


def plugin_data_dir() -> Path:
    return _data_root()


def cache_dir() -> Path:
    return _data_root() / "cache"


def trade_cal_cache_path() -> Path:
    return cache_dir() / "trade_cal.parquet"


def daily_cache_dir() -> Path:
    return cache_dir() / "daily"


def daily_basic_cache_dir() -> Path:
    return cache_dir() / "daily_basic"


def stk_limit_cache_dir() -> Path:
    return cache_dir() / "stk_limit"


def index_daily_cache_dir() -> Path:
    return cache_dir() / "index_daily"


def backtests_dir() -> Path:
    return _data_root() / "backtests"


def reports_dir() -> Path:
    return _data_root() / "reports"


def ensure_layout() -> None:
    """Idempotent: create every directory in the layout if missing."""
    for d in (
        cache_dir(),
        daily_cache_dir(),
        daily_basic_cache_dir(),
        stk_limit_cache_dir(),
        index_daily_cache_dir(),
        backtests_dir(),
        reports_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)
