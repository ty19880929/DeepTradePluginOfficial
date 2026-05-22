"""PR #2 — CLI summary 子命令：window 校验 + end-to-end render。

直接调用 ``resolve_window`` 验证窗口逻辑；CLI 端到端用 typer.testing 调
``winrate summary`` 验证默认 T-1、--start/--end 校验、10 天上限。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from deeptrade.core.db import Database
from typer.testing import CliRunner

from limit_up_board.winrate.cli import (
    MAX_WINDOW_TRADE_DAYS,
    WinrateError,
    resolve_window,
    winrate_app,
)

MIGRATION_FILES = [
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_002_prediction_records.sql",
]


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Database:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))

    from deeptrade.core import paths as core_paths

    database = Database(core_paths.db_path())
    for mig in MIGRATION_FILES:
        sql_text = mig.read_text(encoding="utf-8")
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                database.execute(stmt)
    return database


def _seed_cal(db: Database, days: list[tuple[str, int]]) -> None:
    """``days`` = [(cal_date, is_open), ...]."""
    for cal_date, is_open in days:
        db.execute(
            "INSERT INTO lub_trade_cal (exchange, cal_date, is_open, pretrade_date) "
            "VALUES (?, ?, ?, ?)",
            ("SSE", cal_date, is_open, None),
        )


def _seed_record(db: Database, trade_date: str, ts_code: str = "600001.SH") -> None:
    db.execute(
        "INSERT INTO lub_prediction_records "
        "(trade_date, next_trade_date, ts_code, name, run_id, prediction, rank, "
        " continuation_score, confidence, t_close_price, lgb_score, lgb_decile, raw_prediction_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (trade_date, trade_date, ts_code, "n", "r", "top_candidate",
         1, 80.0, "high", 10.0, 0.7, 9, None),
    )


# ---------------------------------------------------------------------------
# resolve_window
# ---------------------------------------------------------------------------


def test_default_window_uses_t_minus_1_from_calendar(db: Database) -> None:
    """默认窗口 = lub_trade_cal 中 cal_date < today 的最新 is_open=1 日。"""
    # Today = 20260522 → T-1 should be 20260521 (open) skipping 20260522/today.
    _seed_cal(db, [
        ("20260520", 1),
        ("20260521", 1),
        ("20260522", 1),  # today — excluded
        ("20260523", 0),  # not open
    ])
    with patch("limit_up_board.winrate.cli._today_str", return_value="20260522"):
        w = resolve_window(db, None, None)
    assert w.start == w.end == "20260521"
    assert w.is_default is True


def test_default_window_skips_non_open_days(db: Database) -> None:
    _seed_cal(db, [
        ("20260518", 1),
        ("20260519", 0),  # weekend / holiday
        ("20260520", 0),
        ("20260521", 1),
        ("20260522", 0),  # today is_open=0
    ])
    with patch("limit_up_board.winrate.cli._today_str", return_value="20260522"):
        w = resolve_window(db, None, None)
    assert w.start == w.end == "20260521"


def test_default_window_falls_back_to_max_record(db: Database) -> None:
    """lub_trade_cal 空 → 退化用 max(trade_date) from records。"""
    _seed_record(db, "20260520")
    with patch("limit_up_board.winrate.cli._today_str", return_value="20260530"):
        w = resolve_window(db, None, None)
    assert w.start == w.end == "20260520"


def test_default_window_no_data_raises(db: Database) -> None:
    with patch("limit_up_board.winrate.cli._today_str", return_value="20260522"):
        with pytest.raises(WinrateError):
            resolve_window(db, None, None)


def test_explicit_window_only_one_endpoint_raises(db: Database) -> None:
    with pytest.raises(WinrateError):
        resolve_window(db, "20260520", None)
    with pytest.raises(WinrateError):
        resolve_window(db, None, "20260520")


def test_explicit_window_start_gt_end_raises(db: Database) -> None:
    _seed_cal(db, [("20260520", 1), ("20260521", 1)])
    with pytest.raises(WinrateError):
        resolve_window(db, "20260521", "20260520")


def test_explicit_window_within_cap(db: Database) -> None:
    """10 个交易日：刚好不超 cap。"""
    days = [(f"202605{i:02d}", 1) for i in range(11, 21)]  # 10 open days
    _seed_cal(db, days)
    w = resolve_window(db, "20260511", "20260520")
    assert w.start == "20260511"
    assert w.end == "20260520"
    assert w.is_default is False


def test_explicit_window_exceeds_cap_raises(db: Database) -> None:
    """11 个交易日：超过 cap，必须报错。"""
    days = [(f"202605{i:02d}", 1) for i in range(11, 22)]  # 11 open days
    _seed_cal(db, days)
    with pytest.raises(WinrateError) as exc:
        resolve_window(db, "20260511", "20260521")
    assert str(MAX_WINDOW_TRADE_DAYS) in str(exc.value)


def test_explicit_window_calendar_absent_uses_record_dates(db: Database) -> None:
    """无 lub_trade_cal → 用 records 里的 distinct trade_date 数计算 cap。"""
    for d in [f"202605{i:02d}" for i in range(11, 22)]:
        _seed_record(db, d, ts_code=f"60000{int(d[-2:])}.SH")
    with pytest.raises(WinrateError):
        resolve_window(db, "20260511", "20260521")


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------


def test_cli_summary_default_renders(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """跑通 summary，输出包含标题 + 摘要数字。"""
    _seed_cal(db, [("20260520", 1), ("20260521", 1), ("20260522", 0)])
    _seed_record(db, "20260521")

    monkeypatch.setattr(
        "limit_up_board.winrate.cli._today_str", lambda: "20260522",
    )
    # CLI opens a fresh runtime via the configured DEEPTRADE_HOME (set by `db`
    # fixture); reuse that for the CliRunner subprocess.
    runner = CliRunner()
    result = runner.invoke(winrate_app, ["summary"])
    assert result.exit_code == 0, result.stdout
    assert "胜率分析" in result.stdout
    assert "20260521" in result.stdout
    # No T+1 data seeded → 1 unresolved
    assert "待解析:" in result.stdout


def test_cli_summary_only_start_errors(db: Database) -> None:
    runner = CliRunner()
    result = runner.invoke(winrate_app, ["summary", "--start", "20260520"])
    assert result.exit_code == 2
    assert "--start 和 --end" in result.stdout


def test_cli_summary_eleven_days_errors(db: Database) -> None:
    days = [(f"202605{i:02d}", 1) for i in range(11, 22)]
    _seed_cal(db, days)
    runner = CliRunner()
    result = runner.invoke(
        winrate_app, ["summary", "--start", "20260511", "--end", "20260521"],
    )
    assert result.exit_code == 2
    assert "10" in result.stdout
    assert "请缩短区间" in result.stdout


def test_cli_summary_unknown_prediction_errors(db: Database) -> None:
    runner = CliRunner()
    result = runner.invoke(
        winrate_app, ["summary", "--prediction", "nonsense"],
    )
    assert result.exit_code == 2
    assert "nonsense" in result.stdout
