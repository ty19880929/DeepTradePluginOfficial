"""v0.3.0 — pure-SQL ``stats`` aggregation over apw_signal_history /
apw_stage_results / apw_realized_returns.

Validates the routing logic in :mod:`accumulation_probe_washout.stats`:
* unknown --by → StatsQueryError
* lgb_score_bin before PR-4 → friendly error
* categorical (phase / prediction / main_pattern) → group-by buckets
* score_bin axes → 4-bucket buckets even when some are empty
* dimension_scores → 6 dim columns × Pearson r
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from accumulation_probe_washout.stats import (
    ALLOWED_BY,
    DIMENSION_COLS,
    StatsQueryError,
    run_stats_query,
)


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


def _seed_signal_history(
    db,
    *,
    trade_date: str,
    ts_code: str,
    phase: str,
    accumulation: float,
    probe_quality: float,
    washout: float,
    launch_setup: float,
) -> None:
    cand = {
        "trade_date": trade_date,
        "ts_code": ts_code,
        "phase": phase,
        "accumulation_score": accumulation,
        "probe_quality_score": probe_quality,
        "washout_score": washout,
        "launch_setup_score": launch_setup,
    }
    db.execute(
        """
        INSERT INTO apw_signal_history
        (trade_date, ts_code, name, phase, accumulation_score,
         probe_quality_score, washout_score, launch_setup_score,
         raw_candidate_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            trade_date, ts_code, f"测试{ts_code[:6]}",
            phase, accumulation, probe_quality, washout, launch_setup,
            json.dumps(cand, ensure_ascii=False), datetime.now(),
        ],
    )


def _seed_stage_result(
    db,
    *,
    run_id: str,
    trade_date: str,
    ts_code: str,
    prediction: str,
    main_pattern: str,
    phase: str,
    launch_score: float,
    dims: dict[str, float],
) -> None:
    db.execute(
        """
        INSERT INTO apw_stage_results
        (run_id, trade_date, ts_code, candidate_id, rank, launch_score,
         confidence, prediction, main_pattern, phase,
         dimension_scores_json, key_evidence_json, rationale,
         next_session_watch_json, invalidation_triggers_json,
         risk_flags_json, missing_data_json, raw_response_json, created_at,
         dim_accumulation, dim_probe, dim_washout, dim_launch_timing,
         dim_capital_confirmation, dim_risk)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id, trade_date, ts_code, f"{trade_date}_{ts_code}", 1,
            launch_score, "high", prediction, main_pattern, phase,
            json.dumps(dims), "[]", "rationale", "[]", "[]", "[]", "[]", "{}",
            datetime.now(),
            dims["accumulation"], dims["probe"], dims["washout"],
            dims["launch_timing"], dims["capital_confirmation"], dims["risk"],
        ],
    )


def _seed_realized(db, *, signal_date: str, ts_code: str, ret_t5_pct: float,
                   label_t5: int) -> None:
    db.execute(
        """
        INSERT INTO apw_realized_returns
        (signal_date, ts_code, ret_t5_pct, max_high_t5_pct,
         max_drawdown_t5_pct, label_launch_t5, data_status)
        VALUES (?, ?, ?, ?, ?, ?, 'complete')
        """,
        [signal_date, ts_code, ret_t5_pct, ret_t5_pct + 3.0, 4.0, label_t5],
    )


def _seed_scenario(db) -> None:
    """Two trade_dates × two phases × known realized returns."""
    run_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    _seed_signal_history(db, trade_date="20260501", ts_code="600000.SH",
                         phase="launch_ready", accumulation=72, probe_quality=85,
                         washout=66, launch_setup=78)
    _seed_signal_history(db, trade_date="20260501", ts_code="600001.SH",
                         phase="washing_after_probe", accumulation=55, probe_quality=70,
                         washout=58, launch_setup=42)
    _seed_signal_history(db, trade_date="20260502", ts_code="600002.SH",
                         phase="launch_ready", accumulation=82, probe_quality=90,
                         washout=75, launch_setup=88)

    _seed_stage_result(db, run_id=run_id, trade_date="20260501",
                       ts_code="600000.SH", prediction="launch_ready",
                       main_pattern="probe_washout_breakout", phase="launch_ready",
                       launch_score=78.0,
                       dims={"accumulation": 72, "probe": 85, "washout": 66,
                             "launch_timing": 78, "capital_confirmation": 58,
                             "risk": 22})
    _seed_stage_result(db, run_id=run_id, trade_date="20260501",
                       ts_code="600001.SH", prediction="still_washing",
                       main_pattern="low_base_accumulation", phase="washing_after_probe",
                       launch_score=42.0,
                       dims={"accumulation": 55, "probe": 70, "washout": 58,
                             "launch_timing": 30, "capital_confirmation": 40,
                             "risk": 50})
    _seed_stage_result(db, run_id=run_id, trade_date="20260502",
                       ts_code="600002.SH", prediction="launch_ready",
                       main_pattern="probe_washout_breakout", phase="launch_ready",
                       launch_score=88.0,
                       dims={"accumulation": 82, "probe": 90, "washout": 75,
                             "launch_timing": 88, "capital_confirmation": 70,
                             "risk": 18})

    _seed_realized(db, signal_date="20260501", ts_code="600000.SH",
                   ret_t5_pct=6.5, label_t5=1)
    _seed_realized(db, signal_date="20260501", ts_code="600001.SH",
                   ret_t5_pct=-1.2, label_t5=0)
    _seed_realized(db, signal_date="20260502", ts_code="600002.SH",
                   ret_t5_pct=9.0, label_t5=1)


def test_unknown_by_raises(fresh_db):
    with pytest.raises(StatsQueryError):
        run_stats_query(fresh_db, from_date=None, to_date=None, by="not_a_dim")


def test_by_phase_returns_two_buckets(fresh_db):
    _seed_scenario(fresh_db)
    rows, title = run_stats_query(
        fresh_db, from_date="20260501", to_date="20260531", by="phase"
    )
    by_bucket = {r["bucket"]: r for r in rows}
    assert "launch_ready" in by_bucket
    assert "washing_after_probe" in by_bucket
    assert by_bucket["launch_ready"]["n_samples"] == 2
    # winrate over two launch_ready rows (both label=1) is 100.0
    assert by_bucket["launch_ready"]["label_launch_t5_winrate"] == pytest.approx(100.0)


def test_by_launch_score_bin_emits_four_buckets(fresh_db):
    _seed_scenario(fresh_db)
    rows, _ = run_stats_query(
        fresh_db, from_date="20260501", to_date="20260531", by="launch_score_bin"
    )
    assert [r["bucket"] for r in rows] == ["0-40", "40-60", "60-80", "80-100"]


def test_by_accumulation_score_bin_uses_signal_history(fresh_db):
    _seed_scenario(fresh_db)
    rows, _ = run_stats_query(
        fresh_db, from_date="20260501", to_date="20260531",
        by="accumulation_score_bin",
    )
    # 4 fixed buckets, even when some have 0 samples.
    assert len(rows) == 4
    assert sum(r["n_samples"] for r in rows) == 3  # 3 signal_history rows total


def test_by_dimension_scores_returns_six_rows(fresh_db):
    _seed_scenario(fresh_db)
    rows, _ = run_stats_query(
        fresh_db, from_date="20260501", to_date="20260531",
        by="dimension_scores",
    )
    assert [r["bucket"] for r in rows] == list(DIMENSION_COLS)
    # All rows should have n_samples == 3 (all stage_results joined to realized).
    assert all(r["n_samples"] == 3 for r in rows)


def test_lgb_score_bin_returns_empty_buckets_when_no_predictions(fresh_db):
    """PR-4 creates apw_lgb_predictions; with the table empty, lgb_score_bin
    must still return the 4 canonical buckets (n_samples=0 each)."""
    rows, _ = run_stats_query(
        fresh_db, from_date=None, to_date=None, by="lgb_score_bin"
    )
    assert [r["bucket"] for r in rows] == ["0-30", "30-50", "50-70", "70-100"]
    assert all(r["n_samples"] == 0 for r in rows)


def test_allowed_by_axes_are_stable():
    """Locked in so docs / CLI help don't drift from the dispatch list."""
    assert set(ALLOWED_BY) == {
        "phase", "prediction", "main_pattern",
        "launch_score_bin",
        "accumulation_score_bin", "probe_quality_score_bin",
        "washout_score_bin", "launch_setup_score_bin",
        "dimension_scores", "lgb_score_bin",
    }
