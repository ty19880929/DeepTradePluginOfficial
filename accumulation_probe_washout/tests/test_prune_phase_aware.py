"""v0.3.0 — phase-aware ``prune`` regression tests.

The four trigger rules (see §3.1.5 of the migration plan):
  1. launch_ready idle ≥ prune_idle_days_launch_ready trade days → delete
  2. washing_after_probe past washout_max_trade_days → delete
  3. close on T < probe_low → delete
  4. close on T < MA60 → delete

We exercise _prune_reason directly so the test stays hermetic (no Tushare
network calls). A separate small integration test invokes execute_prune
with a fake Tushare to verify the event/run plumbing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.calendar import TradeCalendar
from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.runner import ApwRunner, PruneParams
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import NullRenderer


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def fresh_db(tmp_path):
    from deeptrade.core.db import Database
    db = Database(tmp_path / "apw.duckdb")
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for stmt in path.read_text(encoding="utf-8").split(";"):
            if stmt.strip():
                db.execute(stmt.strip())
    yield db
    db.close()


@pytest.fixture
def calendar() -> TradeCalendar:
    base = pd.date_range("2024-01-01", "2026-12-31", freq="D")
    return TradeCalendar(
        pd.DataFrame(
            {
                "cal_date": base.strftime("%Y%m%d"),
                "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                "pretrade_date": [None] * len(base),
            }
        )
    )


@pytest.fixture
def runner(fresh_db) -> ApwRunner:
    rt = ApwRuntime(
        db=fresh_db,
        config=None,  # type: ignore[arg-type]
        llms=None,  # type: ignore[arg-type]
        run_id=None,
    )
    return ApwRunner(rt, renderer=NullRenderer())


def test_launch_ready_idle_too_long_triggers_delete(runner, calendar):
    cfg = ApwConfig(prune_idle_days_launch_ready=5)
    # last_seen 20260501; today 20260520 (≥ 5 trade days later)
    reason = runner._prune_reason(
        cfg=cfg, phase="launch_ready",
        last_seen="20260501", probe_date="20260420",
        raw_json="{}", today="20260520", calendar=calendar,
    )
    assert reason is not None
    assert "launch_ready idle" in reason


def test_launch_ready_recent_keeps_row(runner, calendar):
    cfg = ApwConfig(prune_idle_days_launch_ready=5)
    reason = runner._prune_reason(
        cfg=cfg, phase="launch_ready",
        last_seen="20260519", probe_date="20260420",
        raw_json="{}", today="20260520", calendar=calendar,
    )
    assert reason is None


def test_washout_past_max_window_triggers_delete(runner, calendar):
    cfg = ApwConfig(washout_max_trade_days=25)
    reason = runner._prune_reason(
        cfg=cfg, phase="washing_after_probe",
        last_seen="20260520", probe_date="20260401",
        raw_json="{}", today="20260520", calendar=calendar,
    )
    assert reason is not None
    assert "washout_after_probe elapsed" in reason


def test_washout_within_window_keeps_row(runner, calendar):
    cfg = ApwConfig(washout_max_trade_days=25)
    reason = runner._prune_reason(
        cfg=cfg, phase="washing_after_probe",
        last_seen="20260520", probe_date="20260510",
        raw_json="{}", today="20260520", calendar=calendar,
    )
    assert reason is None


def test_close_below_probe_low_triggers_delete(runner, calendar):
    cfg = ApwConfig()
    raw = json.dumps({"close": 8.0, "probe_low": 9.5, "ma60": 7.0})
    reason = runner._prune_reason(
        cfg=cfg, phase="washing_after_probe",
        last_seen="20260520", probe_date="20260518",
        raw_json=raw, today="20260520", calendar=calendar,
    )
    assert reason is not None
    assert "probe_low" in reason


def test_close_below_ma60_triggers_delete(runner, calendar):
    cfg = ApwConfig(prune_drop_on_probe_low_break=False)
    raw = json.dumps({"close": 8.0, "probe_low": 7.0, "ma60": 9.0})
    reason = runner._prune_reason(
        cfg=cfg, phase="washing_after_probe",
        last_seen="20260520", probe_date="20260518",
        raw_json=raw, today="20260520", calendar=calendar,
    )
    assert reason is not None
    assert "MA60" in reason


def test_break_rules_disabled_via_config_keep_row(runner, calendar):
    cfg = ApwConfig(prune_drop_on_probe_low_break=False, prune_drop_on_ma60_break=False)
    raw = json.dumps({"close": 8.0, "probe_low": 9.5, "ma60": 10.0})
    reason = runner._prune_reason(
        cfg=cfg, phase="washing_after_probe",
        last_seen="20260520", probe_date="20260518",
        raw_json=raw, today="20260520", calendar=calendar,
    )
    assert reason is None


def test_malformed_raw_candidate_json_does_not_explode(runner, calendar):
    """A corrupt raw_candidate_json must not break the prune loop."""
    cfg = ApwConfig()
    reason = runner._prune_reason(
        cfg=cfg, phase="washing_after_probe",
        last_seen="20260520", probe_date="20260518",
        raw_json="{not valid json", today="20260520", calendar=calendar,
    )
    # No phase rule trips, no break rule trips because parse failed → None.
    assert reason is None


def test_dry_run_does_not_delete_rows(fresh_db, runner):
    """End-to-end: seed two stale launch_ready rows, run dry-run, confirm
    nothing got deleted but the would-delete count surfaces in the summary."""
    now = datetime.now()
    for code in ("600000.SH", "600001.SH"):
        fresh_db.execute(
            """
            INSERT INTO apw_watchlist
            (ts_code, name, first_seen_date, last_seen_date, phase,
             accumulation_score, probe_quality_score, washout_score,
             launch_setup_score, latest_launch_score, latest_prediction,
             latest_confidence, raw_candidate_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
            """,
            [
                code, "测试", "20260401", "20260401",
                "launch_ready", 70.0, 80.0, 70.0, 75.0,
                json.dumps({}), now,
            ],
        )

    # Patch the runner's tushare-dependent path: prune wants a calendar +
    # latest_trade_date. Inject both as a tiny shim by monkey-patching the
    # module-level imports through a side-channel.
    class _FakeTushare:
        def call(self, api: str, **kw: Any) -> pd.DataFrame:
            if api == "trade_cal":
                base = pd.date_range("2024-01-01", "2026-12-31", freq="D")
                return pd.DataFrame(
                    {
                        "cal_date": base.strftime("%Y%m%d"),
                        "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                        "pretrade_date": [None] * len(base),
                    }
                )
            return pd.DataFrame()

    runner.rt.tushare = _FakeTushare()  # type: ignore[assignment]
    # Pin "today" — also avoids needing fetch_latest_trade_date.
    outcome = runner.execute_prune(PruneParams(dry_run=True, trade_date="20260520"))
    assert outcome.status.value == "success"
    assert outcome.summary["dry_run"] is True
    assert outcome.summary["n_would_delete"] >= 2
    assert outcome.summary["n_deleted"] == 0
    # Still present in the DB.
    assert fresh_db.fetchone(
        "SELECT COUNT(*) FROM apw_watchlist"
    )[0] == 2
