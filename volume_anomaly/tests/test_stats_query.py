"""volume-anomaly v0.4.0 — `stats` SQL aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database
from volume_anomaly.stats import (
    LAUNCH_SCORE_BINS,
    run_stats_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_data(tmp_path: Path) -> Database:
    """Create the two tables and seed deterministic test data."""
    db = Database(tmp_path / "stats_test.duckdb")
    db.execute("""
        CREATE TABLE va_stage_results (
            run_id UUID,
            stage VARCHAR,
            batch_no INTEGER,
            trade_date VARCHAR NOT NULL,
            ts_code VARCHAR NOT NULL,
            name VARCHAR,
            rank INTEGER,
            launch_score DOUBLE,
            confidence VARCHAR,
            prediction VARCHAR,
            pattern VARCHAR,
            rationale VARCHAR,
            tracked_days INTEGER,
            evidence_json VARCHAR,
            risk_flags_json VARCHAR,
            raw_response_json VARCHAR
        )
    """)
    db.execute("""
        CREATE TABLE va_realized_returns (
            anomaly_date VARCHAR NOT NULL,
            ts_code VARCHAR NOT NULL,
            t_close DOUBLE,
            t1_close DOUBLE,
            t3_close DOUBLE,
            t5_close DOUBLE,
            t10_close DOUBLE,
            ret_t1 DOUBLE,
            ret_t3 DOUBLE,
            ret_t5 DOUBLE,
            ret_t10 DOUBLE,
            max_close_5d DOUBLE,
            max_close_10d DOUBLE,
            max_ret_5d DOUBLE,
            max_ret_10d DOUBLE,
            max_dd_5d DOUBLE,
            computed_at TIMESTAMP,
            data_status VARCHAR NOT NULL
        )
    """)

    # Seed data — tuples (trade_date, ts_code, prediction, pattern, score, ret_t3, max_ret_5d)
    samples = [
        ("20260601", "000001.SZ", "imminent_launch", "breakout", 78, 5.0, 9.0),
        ("20260601", "000002.SZ", "imminent_launch", "breakout", 82, 7.0, 12.0),
        ("20260601", "000003.SZ", "watching", "consolidation_break", 55, 1.5, 4.0),
        ("20260602", "000004.SZ", "watching", "consolidation_break", 50, -0.5, 2.0),
        ("20260603", "000005.SZ", "not_yet", "unclear", 25, -2.0, 1.0),
        ("20260603", "000006.SZ", "not_yet", "unclear", 30, 0.0, 1.0),
    ]
    for trade_date, ts_code, pred, pat, score, ret_t3, max_ret in samples:
        db.execute(
            "INSERT INTO va_stage_results VALUES (?, 'analyze', 1, ?, ?, ?, 1, ?, "
            "'high', ?, ?, '...', 0, '[]', '[]', '{}')",
            ("00000000-0000-0000-0000-000000000000", trade_date, ts_code,
             "测试", float(score), pred, pat),
        )
        db.execute(
            "INSERT INTO va_realized_returns VALUES (?, ?, 100.0, 101.0, ?, 102.0, "
            "100.0, 1.0, ?, 2.0, 0.0, 110.0, 110.0, ?, ?, -1.0, "
            "CURRENT_TIMESTAMP, 'complete')",
            (trade_date, ts_code, 100 + ret_t3, ret_t3, max_ret, max_ret),
        )
    yield db
    db.close()


# ---------------------------------------------------------------------------
# stats by prediction
# ---------------------------------------------------------------------------


def test_by_prediction(db_with_data: Database) -> None:
    rows, _title = run_stats_query(
        db_with_data, from_date=None, to_date=None, by="prediction"
    )
    by_bucket = {r["bucket"]: r for r in rows}
    assert "imminent_launch" in by_bucket
    assert by_bucket["imminent_launch"]["n_samples"] == 2
    # imminent_launch: ret_t3 = 5.0, 7.0 → mean 6.0
    assert by_bucket["imminent_launch"]["t3_mean"] == pytest.approx(6.0)
    # both positive → 100% winrate
    assert by_bucket["imminent_launch"]["t3_winrate"] == pytest.approx(100.0)


def test_by_pattern(db_with_data: Database) -> None:
    rows, _title = run_stats_query(
        db_with_data, from_date=None, to_date=None, by="pattern"
    )
    by_bucket = {r["bucket"]: r for r in rows}
    assert "breakout" in by_bucket
    assert by_bucket["breakout"]["n_samples"] == 2


# ---------------------------------------------------------------------------
# stats by launch_score_bin (G4)
# ---------------------------------------------------------------------------


def test_by_launch_score_bin(db_with_data: Database) -> None:
    rows, _title = run_stats_query(
        db_with_data, from_date=None, to_date=None, by="launch_score_bin"
    )
    by_bucket = {r["bucket"]: r for r in rows}
    # samples: 78 (60-80), 82 (80-100), 55 (40-60), 50 (40-60), 25 (0-40), 30 (0-40)
    assert by_bucket["80-100"]["n_samples"] == 1
    assert by_bucket["60-80"]["n_samples"] == 1
    assert by_bucket["40-60"]["n_samples"] == 2
    assert by_bucket["0-40"]["n_samples"] == 2


def test_bins_default() -> None:
    """The default bin labels & boundaries should match the design (G4)."""
    labels = [b[0] for b in LAUNCH_SCORE_BINS]
    assert labels == ["0-40", "40-60", "60-80", "80-100"]


# ---------------------------------------------------------------------------
# Unknown --by argument
# ---------------------------------------------------------------------------


def test_unknown_by_raises(db_with_data: Database) -> None:
    with pytest.raises(ValueError, match="unknown"):
        run_stats_query(db_with_data, from_date=None, to_date=None, by="bogus")


# ---------------------------------------------------------------------------
# Date-range filter
# ---------------------------------------------------------------------------


def test_date_range_filter(db_with_data: Database) -> None:
    rows, _title = run_stats_query(
        db_with_data, from_date="20260602", to_date="20260603", by="prediction"
    )
    by_bucket = {r["bucket"]: r for r in rows}
    # Only watching (06-02) and not_yet (06-03) remain
    assert "imminent_launch" not in by_bucket
    assert by_bucket["watching"]["n_samples"] == 1
    assert by_bucket["not_yet"]["n_samples"] == 2
