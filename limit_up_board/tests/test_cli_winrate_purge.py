"""PR #3 — purge & export CLI 行为。

覆盖：
    - export 写出 JSON / CSV，扩展名推断
    - purge --before 删除指定日期及之前；不动后续日期
    - purge 不带 --yes 在非 TTY 模式下报错退出
    - purge 不识别的日期格式 → 报错
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from deeptrade.core.db import Database
from typer.testing import CliRunner

from limit_up_board.winrate.cli import winrate_app

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


def _seed_record(db: Database, trade_date: str, ts_code: str = "600001.SH") -> None:
    db.execute(
        "INSERT INTO lub_prediction_records "
        "(trade_date, next_trade_date, ts_code, name, run_id, prediction, rank, "
        " continuation_score, confidence, t_close_price, lgb_score, lgb_decile, raw_prediction_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (trade_date, trade_date, ts_code, "n", "r", "top_candidate",
         1, 80.0, "high", 10.0, 0.7, 9, None),
    )


def _seed_cal(db: Database, days: list[tuple[str, int]]) -> None:
    for cal_date, is_open in days:
        db.execute(
            "INSERT INTO lub_trade_cal (exchange, cal_date, is_open, pretrade_date) "
            "VALUES (?, ?, ?, ?)",
            ("SSE", cal_date, is_open, None),
        )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_json(db: Database, tmp_path: Path) -> None:
    _seed_cal(db, [("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")
    out = tmp_path / "winrate.json"
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["export", "--output", str(out), "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["window"] == {"start": "20260521", "end": "20260521"}
    assert payload["summary"]["total"] == 1
    assert payload["records"][0]["ts_code"] == "600001.SH"


def test_export_csv_by_extension(db: Database, tmp_path: Path) -> None:
    _seed_cal(db, [("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")
    out = tmp_path / "winrate.csv"
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["export", "--output", str(out), "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "trade_date"  # header present
    assert any("600001.SH" in row for row in rows[1:])


def test_export_creates_parent_dir(db: Database, tmp_path: Path) -> None:
    _seed_cal(db, [("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")
    nested = tmp_path / "reports" / "2026" / "winrate.json"
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["export", "--output", str(nested), "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 0, result.output
    assert nested.exists()


def test_export_invalid_format_errors(db: Database, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["export", "--output", str(tmp_path / "x.json"), "--format", "yaml"],
    )
    assert result.exit_code == 2
    assert "unsupported" in result.output or "yaml" in result.output


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------


def test_purge_before_inclusive_with_yes(db: Database) -> None:
    for d, code in [
        ("20260518", "600001.SH"),
        ("20260519", "600002.SH"),
        ("20260520", "600003.SH"),
        ("20260521", "600004.SH"),
    ]:
        _seed_record(db, d, code)

    runner = CliRunner()
    result = runner.invoke(
        winrate_app, ["purge", "--before", "20260519", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "已删除 2 条" in result.output

    rows = db.fetchall(
        "SELECT trade_date FROM lub_prediction_records ORDER BY trade_date"
    )
    assert {r[0] for r in rows} == {"20260520", "20260521"}


def test_purge_no_records_to_delete(db: Database) -> None:
    runner = CliRunner()
    result = runner.invoke(
        winrate_app, ["purge", "--before", "20260101", "--yes"],
    )
    assert result.exit_code == 0
    assert "无记录可删除" in result.output


def test_purge_without_yes_in_non_tty_errors(db: Database) -> None:
    """CliRunner 默认 stdin 不是 TTY；不传 --yes 必须报错。"""
    _seed_record(db, "20260518", "600001.SH")
    runner = CliRunner()
    result = runner.invoke(winrate_app, ["purge", "--before", "20260518"])
    assert result.exit_code == 2
    assert "未确认" in result.output


def test_purge_invalid_date_format_errors(db: Database) -> None:
    runner = CliRunner()
    result = runner.invoke(
        winrate_app, ["purge", "--before", "2026-05-21", "--yes"],
    )
    assert result.exit_code == 2
    assert "YYYYMMDD" in result.output


def test_purge_lists_distinct_dates_in_preview(db: Database) -> None:
    """删除前的预览应列出涉及的 trade_date。"""
    for d, c in [("20260518", "a.SH"), ("20260518", "b.SH"), ("20260519", "c.SH")]:
        _seed_record(db, d, c)
    runner = CliRunner()
    result = runner.invoke(
        winrate_app, ["purge", "--before", "20260519", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "20260518" in result.output
    assert "20260519" in result.output
    assert "3 条" in result.output
