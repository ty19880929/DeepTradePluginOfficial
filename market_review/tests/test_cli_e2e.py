"""CLI e2e — patch ``_open_runtime`` to inject fake services + smoke each subcommand."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_review import cli
from market_review.plugin import MarketReviewPlugin
from market_review.runtime import MrRuntime

from conftest import FakeLLMManager, FakeTushare  # type: ignore[import-not-found]


_OPEN_DAYS = ["20260528", "20260529", "20260530"]


def _seed_tushare(fake: FakeTushare) -> None:
    fake.set_static("trade_cal", pd.DataFrame([
        {"exchange": "SSE", "cal_date": d, "is_open": 1, "pretrade_date": None}
        for d in _OPEN_DAYS
    ]))
    fake.set_response(
        "index_daily",
        lambda *, params=None, trade_date=None, **_: pd.DataFrame([
            {"ts_code": (params or {}).get("ts_code", "000001.SH"),
             "trade_date": d, "pct_chg": 0.5, "amount": 1_000_000.0, "close": 3000.0}
            for d in _OPEN_DAYS
        ]),
    )
    fake.set_static("stock_basic", pd.DataFrame([
        {"ts_code": "600001.SH", "symbol": "600001", "name": "A股代表",
         "industry": "光模块", "market": "主板", "exchange": "SSE",
         "list_status": "L"},
    ]))
    fake.set_response(
        "daily",
        lambda *, trade_date=None, **_: pd.DataFrame([
            {"ts_code": "600001.SH", "trade_date": trade_date,
             "pct_chg": 2.0, "amount": 100_000.0, "close": 10.0,
             "open": 9.8, "pre_close": 9.8},
        ]),
    )


@pytest.fixture
def patched_runtime(mr_db, monkeypatch, tmp_path):
    """Patch ``cli._open_runtime`` to return a fake-runtime tuple."""
    fake_tushare = FakeTushare()
    _seed_tushare(fake_tushare)
    for d in _OPEN_DAYS:
        mr_db.execute(
            "INSERT INTO mr_trade_cal (exchange, cal_date, is_open, pretrade_date) "
            "VALUES (?, ?, ?, ?)",
            ["SSE", d, 1, None],
        )
    fake_llms = FakeLLMManager()
    rt = MrRuntime(
        db=mr_db, config=None, llms=fake_llms,  # type: ignore[arg-type]
        tushare=fake_tushare,
    )
    # Override the per-run reports root so files land in tmp_path.
    monkeypatch.setattr(
        "market_review.runner.MrRunner.__init__",
        _make_patched_init(tmp_path / "reports"),
    )
    monkeypatch.setattr(cli, "_open_runtime", lambda: (mr_db, rt, None))
    # Tests dispatch multiple CLI invocations sharing one DB instance; the
    # CLI's per-command ``_close_db`` would break subsequent fetches in
    # this test session. Stub it to a no-op while the fixture is active.
    monkeypatch.setattr(cli, "_close_db", lambda db: None)
    # ``cmd_report`` resolves report files via ``_reports_root() / run_id``;
    # tests' runner writes into ``tmp_path / reports`` — redirect to match.
    monkeypatch.setattr(cli, "_reports_root", lambda: tmp_path / "reports")
    return rt


def _make_patched_init(reports_root: Path):
    """Build a replacement ``MrRunner.__init__`` that pins ``reports_root``.

    Tests need the runner to write into ``tmp_path``; CLI doesn't expose a
    way to override the default ``~/.deeptrade/reports``. We monkey-patch
    the constructor to inject the override.
    """
    from market_review.runner import MrRunner

    original = MrRunner.__init__

    def patched(self, rt, *, ctx=None, plugin_version="0.1.0", reports_root_=reports_root, **kw):
        kw.setdefault("reports_root", reports_root_)
        original(self, rt, ctx=ctx, plugin_version=plugin_version, **kw)

    return patched


# ---------------------------------------------------------------------------
# --help still works
# ---------------------------------------------------------------------------


def test_run_subcommand_help_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = MarketReviewPlugin().dispatch(["run", "--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "--trade-date" in captured.out
    assert "--start" in captured.out


def test_sync_subcommand_help_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = MarketReviewPlugin().dispatch(["sync", "--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "--force-sync" in captured.out


# ---------------------------------------------------------------------------
# sync (lighter than run — exercises Steps 0..1 path through CLI)
# ---------------------------------------------------------------------------


def test_cli_sync_completes(patched_runtime, capsys: pytest.CaptureFixture[str]) -> None:
    rc = MarketReviewPlugin().dispatch(["sync", "--trade-date", "20260530"])
    captured = capsys.readouterr()
    assert rc == 0, f"sync expected exit 0; got {rc}; out={captured.out!r}"
    assert "sync 完成" in captured.out


def test_cli_run_smoke(patched_runtime, capsys: pytest.CaptureFixture[str]) -> None:
    """Full run via CLI — terminal summary shows market_tone + section list."""
    rc = MarketReviewPlugin().dispatch([
        "run", "--trade-date", "20260530", "--no-upload",
    ])
    captured = capsys.readouterr()
    assert rc == 0, f"run expected exit 0; out={captured.out!r}"
    assert "震荡分化" in captured.out  # FakeLLM default market_tone surfaces
    assert "大盘整体" in captured.out  # overview section appears in roster


# ---------------------------------------------------------------------------
# history / settings
# ---------------------------------------------------------------------------


def test_cli_history_lists_recent_run(
    patched_runtime, capsys: pytest.CaptureFixture[str],
) -> None:
    # Seed one row via run first.
    MarketReviewPlugin().dispatch(["sync", "--trade-date", "20260530"])
    capsys.readouterr()
    rc = MarketReviewPlugin().dispatch(["history", "--limit", "5"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "success" in captured.out


def test_cli_history_empty_message(patched_runtime, capsys: pytest.CaptureFixture[str]) -> None:
    rc = MarketReviewPlugin().dispatch(["history"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "暂无 run 历史" in captured.out


def test_cli_settings_show(patched_runtime, capsys: pytest.CaptureFixture[str]) -> None:
    rc = MarketReviewPlugin().dispatch(["settings", "show"])
    captured = capsys.readouterr()
    assert rc == 0
    # MrConfig default fields surface in the table.
    assert "max_window_days" in captured.out
    assert "sentiment_weights" in captured.out


def test_cli_settings_with_no_subcommand_shows(
    patched_runtime, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = MarketReviewPlugin().dispatch(["settings"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "MrConfig" in captured.out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_cli_report_round_trips_summary_md(
    patched_runtime, capsys: pytest.CaptureFixture[str],
) -> None:
    # First do a full run so summary files exist.
    MarketReviewPlugin().dispatch(["run", "--trade-date", "20260530", "--no-upload"])
    capsys.readouterr()
    # Find the run_id from DB.
    rt = patched_runtime
    rows = rt.db.fetchall("SELECT run_id FROM mr_runs ORDER BY started_at DESC LIMIT 1")
    run_id = str(rows[0][0])
    # Default report — short summary.
    rc = MarketReviewPlugin().dispatch(["report", run_id[:8]])
    captured = capsys.readouterr()
    assert rc == 0
    assert "市场复盘" in captured.out


def test_cli_report_full_renders_summary_md(
    patched_runtime, capsys: pytest.CaptureFixture[str],
) -> None:
    MarketReviewPlugin().dispatch(["run", "--trade-date", "20260530", "--no-upload"])
    capsys.readouterr()
    rt = patched_runtime
    rows = rt.db.fetchall("SELECT run_id FROM mr_runs ORDER BY started_at DESC LIMIT 1")
    run_id = str(rows[0][0])
    rc = MarketReviewPlugin().dispatch(["report", run_id[:8], "--full"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "章节" in captured.out


def test_cli_report_unknown_run_id_exits_two(
    patched_runtime, capsys: pytest.CaptureFixture[str],
) -> None:
    rc = MarketReviewPlugin().dispatch(["report", "deadbeef"])
    cap = capsys.readouterr()
    assert rc == 2
    assert "找不到" in (cap.out + cap.err)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_cli_run_with_mutex_flags_exits_two(
    patched_runtime, capsys: pytest.CaptureFixture[str],
) -> None:
    """--trade-date AND --start are mutually exclusive — PreconditionError → 2."""
    rc = MarketReviewPlugin().dispatch([
        "run", "--trade-date", "20260530", "--start", "20260520",
    ])
    cap = capsys.readouterr()
    assert rc == 2
    assert "互斥" in (cap.out + cap.err)
