"""PR #4 — `winrate llm-review` end-to-end CLI 测试。

用 ``unittest.mock`` 替换 ``LLMManager.get_client`` 返回的客户端的
``complete_json``，避免真实网络调用。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deeptrade.core.db import Database
from typer.testing import CliRunner

from limit_up_board.winrate.cli import winrate_app
from limit_up_board.winrate.llm_review import (
    LandedSuggestion,
    WinrateLlmReview,
)

MIGRATION_FILES = [
    Path(__file__).resolve().parents[1] / "migrations" / "20260509_001_init.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_001_lgb_tables.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_002_prediction_records.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_003_winrate_reviews.sql",
    Path(__file__).resolve().parents[1] / "migrations" / "20260516_001_evidence_validation.sql",
    # v0.13.1：lub_lgb_models.calibration_* 列
    Path(__file__).resolve().parents[1] / "migrations" / "20260524_001_lgb_calibration.sql",
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


def _seed_record(db: Database, trade_date: str, ts_code: str) -> None:
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


def _sample_response() -> WinrateLlmReview:
    return WinrateLlmReview(
        diagnosis="样本中 top_candidate 非亏比例偏低，分类边界宽于评分本身的区分度。",
        prompt_adjustments=[
            LandedSuggestion(landing="prompt_weighting", text="提升封单强度权重描述"),
        ],
        validation_plan="再积累 5 个交易日并对比 strict_win_rate。",
        caveats=["样本量较小"],
    )


def _mock_llm_client() -> MagicMock:
    mock = MagicMock()
    mock.complete_json.return_value = (_sample_response(), {"audit": "stub"})
    mock.model_name = "deepseek-reasoner-stub"
    return mock


@pytest.fixture
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace LLMManager.get_client with a mock that returns _mock_llm_client()."""
    mock_client = _mock_llm_client()
    monkeypatch.setattr(
        "deeptrade.core.llm_manager.LLMManager.get_client",
        lambda self, *args, **kwargs: mock_client,
    )
    return mock_client


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_llm_review_writes_db_and_no_file(db: Database, patch_llm: MagicMock) -> None:
    _seed_cal(db, [("20260520", 1), ("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")

    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 0, result.output
    patch_llm.complete_json.assert_called_once()

    row = db.fetchone(
        "SELECT review_id, window_start, window_end, llm_provider, sample_total, "
        "sample_resolved, report_path "
        "FROM lub_winrate_reviews"
    )
    assert row is not None
    assert row[1] == "20260521"
    assert row[4] == 1
    assert row[6] is None  # 未传 --output
    assert "LLM 诊断" in result.output


def test_llm_review_writes_db_and_markdown(
    db: Database, patch_llm: MagicMock, tmp_path: Path
) -> None:
    _seed_cal(db, [("20260520", 1), ("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")
    out = tmp_path / "review.md"

    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260521", "--end", "20260521",
         "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    md = out.read_text(encoding="utf-8")
    assert "## diagnosis" in md

    row = db.fetchone("SELECT report_path FROM lub_winrate_reviews")
    assert row is not None
    # Path is stored as a string; on Windows this is a backslash form.
    assert str(out.name) in (row[0] or "")


def test_llm_review_creates_parent_dir(
    db: Database, patch_llm: MagicMock, tmp_path: Path
) -> None:
    _seed_cal(db, [("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")
    nested = tmp_path / "reports" / "review.md"
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260521", "--end", "20260521",
         "--output", str(nested)],
    )
    assert result.exit_code == 0, result.output
    assert nested.exists()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_llm_review_empty_window_errors(db: Database, patch_llm: MagicMock) -> None:
    _seed_cal(db, [("20260521", 1)])
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 2
    assert "无预测记录" in result.output
    # LLM 不应被调用
    patch_llm.complete_json.assert_not_called()
    # DB 不应有行
    row = db.fetchone("SELECT COUNT(*) FROM lub_winrate_reviews")
    assert row[0] == 0


def test_llm_review_window_exceeds_cap(db: Database, patch_llm: MagicMock) -> None:
    days = [(f"202605{i:02d}", 1) for i in range(11, 22)]
    _seed_cal(db, days)
    for d in [f"202605{i:02d}" for i in range(11, 22)]:
        _seed_record(db, d, ts_code=f"60000{int(d[-2:])}.SH")
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260511", "--end", "20260521"],
    )
    assert result.exit_code == 2
    assert "10" in result.output


def test_llm_review_llm_failure_no_db_write(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM 抛异常 → 用户看到错误 + DB 无行（落库在 call_llm 之后）。"""
    _seed_cal(db, [("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")

    mock = MagicMock()
    mock.complete_json.side_effect = RuntimeError("LLM transport down")
    monkeypatch.setattr(
        "deeptrade.core.llm_manager.LLMManager.get_client",
        lambda self, *args, **kwargs: mock,
    )
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 1
    assert "LLM 调用失败" in result.output

    row = db.fetchone("SELECT COUNT(*) FROM lub_winrate_reviews")
    assert row[0] == 0


def test_llm_review_llm_client_init_failure(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLMManager.get_client 抛异常（如 LLMNotConfiguredError） → 干净退出。"""
    _seed_cal(db, [("20260521", 1)])
    _seed_record(db, "20260521", "600001.SH")

    def _raise(self, *args, **kwargs):
        raise RuntimeError("no llm configured")

    monkeypatch.setattr(
        "deeptrade.core.llm_manager.LLMManager.get_client", _raise,
    )
    runner = CliRunner()
    result = runner.invoke(
        winrate_app,
        ["llm-review", "--start", "20260521", "--end", "20260521"],
    )
    assert result.exit_code == 2
    assert "LLM 客户端获取失败" in result.output
