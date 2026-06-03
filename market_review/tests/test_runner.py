"""MrRunner — end-to-end pipeline with FakeTushare + FakeLLM + tmp DB."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_review.runner import (
    MrRunner,
    PreconditionError,
    RunOutcome,
    RunParams,
    _compute_input_fingerprint,
)
from market_review.runtime import MrRuntime

from conftest import FakeLLMClient, FakeLLMManager, FakeTushare  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Tushare fixture data — minimal-but-syncable
# ---------------------------------------------------------------------------


_OPEN_DAYS = ["20260528", "20260529", "20260530"]


def _seed_tushare(fake: FakeTushare) -> None:
    """Configure FakeTushare to respond plausibly for sync_window's APIs."""

    # trade_cal returns a small calendar with the 3 open days marked open.
    cal_rows = [
        {"exchange": "SSE", "cal_date": d, "is_open": 1, "pretrade_date": None}
        for d in _OPEN_DAYS
    ]
    fake.set_static("trade_cal", pd.DataFrame(cal_rows))

    # index_daily returns rows for 000001.SH (latest-date probe + window).
    fake.set_response(
        "index_daily",
        lambda *, params=None, trade_date=None, **_: pd.DataFrame([
            {"ts_code": (params or {}).get("ts_code", "000001.SH"),
             "trade_date": d, "pct_chg": 0.5, "amount": 10_000_000.0,
             "close": 3000.0}
            for d in _OPEN_DAYS
        ]),
    )

    # stock_basic — small universe with one stock per market.
    fake.set_static(
        "stock_basic",
        pd.DataFrame([
            {"ts_code": "600001.SH", "symbol": "600001", "name": "A股代表",
             "industry": "光模块", "market": "主板", "exchange": "SSE",
             "list_status": "L"},
            {"ts_code": "300001.SZ", "symbol": "300001", "name": "创业板代表",
             "industry": "AI", "market": "创业板", "exchange": "SZSE",
             "list_status": "L"},
        ]),
    )

    # daily returns one row per (ts_code, trade_date).
    fake.set_response(
        "daily",
        lambda *, trade_date=None, **_: pd.DataFrame([
            {"ts_code": "600001.SH", "trade_date": trade_date,
             "pct_chg": 2.0, "amount": 100_000.0, "close": 10.0,
             "open": 9.8, "pre_close": 9.8},
            {"ts_code": "300001.SZ", "trade_date": trade_date,
             "pct_chg": -1.0, "amount": 50_000.0, "close": 20.0,
             "open": 20.2, "pre_close": 20.2},
        ]),
    )
    # Other per-day APIs return empty — sync_window records them in empty_days.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def primed_tushare(fake_tushare: FakeTushare) -> FakeTushare:
    _seed_tushare(fake_tushare)
    return fake_tushare


def _seed_mr_trade_cal(db) -> None:
    """FakeTushare.materialize is a no-op test fake, so the runner's
    "fetch trade_cal → materialize → re-fetch" bootstrap can't actually
    populate mr_trade_cal. We pre-seed the table here so _resolve_window
    finds rows on its first DB read and skips the Tushare fetch entirely.
    """
    for d in _OPEN_DAYS:
        db.execute(
            "INSERT INTO mr_trade_cal (exchange, cal_date, is_open, pretrade_date) "
            "VALUES (?, ?, ?, ?)",
            ["SSE", d, 1, None],
        )


def _seed_leaders_signal(db) -> None:
    """Seed a single ``mr_limit_step`` row + ``mr_stock_basic`` + ``mr_daily``
    so :func:`compute_leaders` returns at least one candidate ≥ 30 score and
    the v0.1.14 empty-pool short-circuit in ``pipeline.run_sections`` does
    NOT fire. Without this, full-run tests would only see 6 LLM calls
    (leaders short-circuited) — fine semantically, but it would break the
    "all 7 stages" contract these tests were originally written for.

    A 4-step 连板 on 600001.SH on the anchor day yields
    ladder_score ≈ 25 * log2(5)/log2(8) ≈ 19.34 — well over 30 even with
    return/capital/theme all at zero, so the candidate clears the cutoff.
    """
    db.execute(
        "INSERT INTO mr_stock_basic (ts_code, symbol, name, industry, market, list_status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ["600001.SH", "600001", "A股代表", "光模块", "主板", "L"],
    )
    db.execute(
        "INSERT INTO mr_daily (ts_code, trade_date, pct_chg, close) VALUES (?, ?, ?, ?)",
        ["600001.SH", "20260530", 5.0, 10.0],
    )
    db.execute(
        "INSERT INTO mr_limit_step (trade_date, ts_code, nums) VALUES (?, ?, ?)",
        ["20260530", "600001.SH", 4],
    )


@pytest.fixture
def runner_rt(mr_db, primed_tushare, fake_llm_manager: FakeLLMManager) -> MrRuntime:
    _seed_mr_trade_cal(mr_db)
    _seed_leaders_signal(mr_db)
    return MrRuntime(
        db=mr_db,
        config=None,  # type: ignore[arg-type] — runner doesn't read config in tests
        llms=fake_llm_manager,  # type: ignore[arg-type]
        tushare=primed_tushare,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# sync-only path
# ---------------------------------------------------------------------------


def test_sync_only_completes_and_records_run_row(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute_sync_only(RunParams(trade_date="20260530"))
    assert isinstance(out, RunOutcome)
    assert out.status == "success"
    assert out.run_id

    rows = runner_rt.db.fetchall(
        "SELECT run_id, status, anchor FROM mr_runs WHERE run_id = ?",
        [out.run_id],
    )
    assert len(rows) == 1
    assert str(rows[0][1]) == "success"
    assert str(rows[0][2]) == "20260530"


def test_sync_only_emits_step_events_into_mr_events(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute_sync_only(RunParams(trade_date="20260530"))
    rows = runner_rt.db.fetchall(
        "SELECT event_type, message FROM mr_events WHERE run_id = ? ORDER BY seq",
        [out.run_id],
    )
    types = [str(r[0]) for r in rows]
    # At minimum: 1 Step 0 + 2 sync events (sync_window + sync_sector_quotes).
    assert types.count("STEP_STARTED") >= 2
    assert types.count("STEP_FINISHED") >= 2


def test_sync_only_does_not_call_llm(runner_rt, fake_llm_manager, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    runner.execute_sync_only(RunParams(trade_date="20260530"))
    assert fake_llm_manager.client.calls == []


# ---------------------------------------------------------------------------
# Full path
# ---------------------------------------------------------------------------


def test_full_run_completes_with_success_status(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute(RunParams(trade_date="20260530", no_upload=True))
    assert out.status == "success"
    assert out.failed_sections == []

    # mr_runs row updated to "success".
    rows = runner_rt.db.fetchall(
        "SELECT status, finished_at, input_fingerprint FROM mr_runs WHERE run_id = ?",
        [out.run_id],
    )
    assert str(rows[0][0]) == "success"
    assert rows[0][1] is not None
    assert rows[0][2] and len(str(rows[0][2])) == 64


def test_full_run_writes_summary_files_to_report_dir(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute(RunParams(trade_date="20260530", no_upload=True))
    # All 5 + section files present.
    assert (out.report_dir / "summary.json").is_file()
    assert (out.report_dir / "summary.md").is_file()
    assert (out.report_dir / "overview.md").is_file()
    assert (out.report_dir / "sectors.md").is_file()
    assert (out.report_dir / "metrics.json").is_file()
    assert (out.report_dir / "llm_calls.jsonl").is_file()


def test_summary_json_round_trips_through_review_report_schema(
    runner_rt, tmp_path,
) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute(RunParams(trade_date="20260530", no_upload=True))
    from market_review.report.schema import ReviewReportSchema
    raw = (out.report_dir / "summary.json").read_text(encoding="utf-8")
    report = ReviewReportSchema.model_validate_json(raw)
    assert report.meta.run_id == out.run_id
    assert report.meta.status == "success"
    # Headline carries the overview placeholder market_tone.
    assert report.headline.market_tone == "震荡分化"


def test_full_run_calls_llm_for_all_seven_sections(
    runner_rt, fake_llm_manager, tmp_path,
) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    runner.execute(RunParams(trade_date="20260530", no_upload=True))
    stages = [c["stage"] for c in fake_llm_manager.client.calls]
    assert stages == [
        "overview", "sectors", "sentiment", "capital",
        "leaders", "style", "risk_outlook",
    ]


def test_input_fingerprint_propagated_to_every_llm_call(
    runner_rt, fake_llm_manager, tmp_path,
) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    runner.execute(RunParams(trade_date="20260530", no_upload=True))
    fps = {c["input_fingerprint"] for c in fake_llm_manager.client.calls}
    # All 7 LLM calls share the same 64-char fingerprint.
    assert len(fps) == 1
    only_fp = next(iter(fps))
    assert len(only_fp) == 64
    int(only_fp, 16)  # raises if not hex


def test_full_run_persists_stage_results(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute(RunParams(trade_date="20260530", no_upload=True))
    rows = runner_rt.db.fetchall(
        "SELECT section FROM mr_stage_results WHERE run_id = ?",
        [out.run_id],
    )
    sections_persisted = {str(r[0]) for r in rows}
    assert sections_persisted == {
        "overview", "sectors", "sentiment", "capital",
        "leaders", "style", "risk_outlook",
    }


def test_no_upload_flag_skips_uploader_path(
    runner_rt, fake_llm_manager, tmp_path,
) -> None:
    """``no_upload=True`` → no upload events even when ctx is set."""
    runner = MrRunner(runner_rt, ctx=object(), reports_root=tmp_path / "reports")
    out = runner.execute(RunParams(trade_date="20260530", no_upload=True))
    rows = runner_rt.db.fetchall(
        "SELECT message FROM mr_events WHERE run_id = ? AND event_type = 'LOG'",
        [out.run_id],
    )
    msgs = [str(r[0]) for r in rows]
    assert not any("上传" in m for m in msgs)


# ---------------------------------------------------------------------------
# partial_failed path
# ---------------------------------------------------------------------------


def test_failed_section_yields_partial_failed_status(
    runner_rt, fake_llm_manager, tmp_path,
) -> None:
    def responder(stage, schema_cls, *, user=None):  # noqa: ARG001
        if stage == "capital":
            raise RuntimeError("LLM timeout")
        from conftest import _default_llm_responder  # type: ignore
        return _default_llm_responder(stage, schema_cls, user=user)

    fake_llm_manager.client = FakeLLMClient(responder=responder)
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out = runner.execute(RunParams(trade_date="20260530", no_upload=True))
    assert out.status == "partial_failed"
    assert out.failed_sections == ["capital"]

    # mr_runs reflects status.
    rows = runner_rt.db.fetchall(
        "SELECT status FROM mr_runs WHERE run_id = ?", [out.run_id],
    )
    assert str(rows[0][0]) == "partial_failed"

    # Capital section's persisted schema has error string non-empty.
    rows = runner_rt.db.fetchall(
        "SELECT response_json FROM mr_stage_results "
        "WHERE run_id = ? AND section = 'capital'",
        [out.run_id],
    )
    payload = json.loads(rows[0][0])
    assert payload["error"] and "LLM timeout" in payload["error"]


# ---------------------------------------------------------------------------
# failed path
# ---------------------------------------------------------------------------


def test_window_spec_error_raises_precondition(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    # --trade-date AND --start are mutually exclusive (PR-2 design §3.2).
    with pytest.raises(PreconditionError, match="互斥"):
        runner.execute(RunParams(trade_date="20260530", start="20260520"))


def test_missing_tushare_raises_precondition_for_implicit_window(
    mr_db, fake_llm_manager, tmp_path,
) -> None:
    """No trade_cal in DB AND no tushare client → can't bootstrap calendar."""
    rt = MrRuntime(
        db=mr_db, config=None, llms=fake_llm_manager,  # type: ignore[arg-type]
        tushare=None,
    )
    runner = MrRunner(rt, reports_root=tmp_path / "reports")
    with pytest.raises(PreconditionError):
        runner.execute(RunParams(trade_date="20260530"))


# ---------------------------------------------------------------------------
# Fingerprint determinism
# ---------------------------------------------------------------------------


def test_input_fingerprint_is_deterministic_for_same_bundle() -> None:
    from market_review.metrics.breadth import BreadthReview
    from market_review.metrics.capital import CapitalReview
    from market_review.metrics.leaders import LeaderReview
    from market_review.metrics.risk import RiskReview
    from market_review.metrics.sectors import SectorReview
    from market_review.metrics.sentiment import SentimentReview
    from market_review.metrics.style import StyleReview
    from market_review.pipeline import MetricsBundle
    from market_review.windows import Window

    bundle = MetricsBundle(
        window=Window(mode="day", start="20260530", end="20260530",
                      trade_dates=("20260530",), anchor="20260530"),
        breadth=BreadthReview(), sentiment=SentimentReview(),
        capital=CapitalReview(), sectors=SectorReview(),
        leaders=LeaderReview(), style=StyleReview(), risk=RiskReview(),
    )
    a = _compute_input_fingerprint(bundle, plugin_version="0.1.0")
    b = _compute_input_fingerprint(bundle, plugin_version="0.1.0")
    c = _compute_input_fingerprint(bundle, plugin_version="0.2.0")
    assert a == b
    assert len(a) == 64
    int(a, 16)
    # Different plugin version produces different hash.
    assert a != c


# ---------------------------------------------------------------------------
# Reports root configurability (tests use tmp_path; production uses ~/.deeptrade)
# ---------------------------------------------------------------------------


def test_replay_policy_builder_tolerates_none_config(runner_rt, tmp_path) -> None:
    """rt.config=None (the test fixture default) → _build_replay_policy=None.

    Prevents a regression where the policy build accidentally requires a
    real ConfigService and breaks the test-runner setup we use everywhere.
    """
    from market_review.runner import MrRunner  # noqa: PLC0415
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    # rt.config is None per the runner_rt fixture, so policy must be None.
    assert runner._build_replay_policy() is None


def test_replay_policy_passed_to_complete_json(
    runner_rt, fake_llm_manager, tmp_path,
) -> None:
    """The replay kw lands in every complete_json call; tests pass disabled-by-
    default (None) since rt.config=None."""
    import inspect  # noqa: PLC0415
    # Re-instrument complete_json to capture replay kw.
    seen_replays = []
    original = fake_llm_manager.client.complete_json

    def spy(*args, **kw):
        seen_replays.append(kw.get("replay"))
        return original(*args, **kw)

    fake_llm_manager.client.complete_json = spy  # type: ignore[assignment]

    from market_review.runner import MrRunner  # noqa: PLC0415
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    runner.execute(RunParams(trade_date="20260530", no_upload=True))
    # 7 sections × replay kw — all None in this fixture (config=None).
    assert len(seen_replays) == 7
    assert all(r is None for r in seen_replays)


def test_reports_dir_is_per_run(runner_rt, tmp_path) -> None:
    runner = MrRunner(runner_rt, reports_root=tmp_path / "reports")
    out1 = runner.execute_sync_only(RunParams(trade_date="20260530"))
    out2 = runner.execute_sync_only(RunParams(trade_date="20260530"))
    assert out1.run_id != out2.run_id
    assert out1.report_dir != out2.report_dir
    assert out1.report_dir.parent == out2.report_dir.parent == tmp_path / "reports"
