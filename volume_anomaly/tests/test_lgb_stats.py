"""PR-2.3 — ``stats --by lgb_score_bin`` aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database
from volume_anomaly.stats import LGB_SCORE_BINS, run_stats_query


@pytest.fixture
def db_with_predictions(tmp_path: Path) -> Database:
    db = Database(tmp_path / "lgb_stats.duckdb")
    db.execute(
        """
        CREATE TABLE va_lgb_predictions (
            run_id                UUID NOT NULL,
            trade_date            VARCHAR NOT NULL,
            ts_code               VARCHAR NOT NULL,
            model_id              VARCHAR NOT NULL,
            lgb_score             DOUBLE NOT NULL,
            lgb_decile            INTEGER,
            feature_hash          VARCHAR NOT NULL,
            feature_missing_json  VARCHAR,
            created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_id, ts_code)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE va_realized_returns (
            anomaly_date  VARCHAR NOT NULL,
            ts_code       VARCHAR NOT NULL,
            t_close       DOUBLE,
            ret_t1        DOUBLE,
            ret_t3        DOUBLE,
            ret_t5        DOUBLE,
            ret_t10       DOUBLE,
            max_ret_5d    DOUBLE,
            max_ret_10d   DOUBLE,
            max_dd_5d     DOUBLE,
            data_status   VARCHAR NOT NULL,
            PRIMARY KEY (anomaly_date, ts_code)
        )
        """
    )
    # Seed: 2 dates × 4 codes; scores spread across the four bins.
    samples = [
        # (trade_date, ts_code, lgb_score, ret_t3, max_ret_5d)
        ("20260601", "000001.SZ", 15.0, -1.0, 0.5),     # 0-30
        ("20260601", "000002.SZ", 40.0, 1.0, 4.0),      # 30-50
        ("20260601", "000003.SZ", 60.0, 5.0, 8.0),      # 50-70
        ("20260601", "000004.SZ", 85.0, 9.0, 12.0),     # 70-100
        ("20260602", "000005.SZ", 80.0, 6.0, 9.0),      # 70-100
    ]
    for i, (td, code, score, ret_t3, mr5) in enumerate(samples):
        db.execute(
            "INSERT INTO va_lgb_predictions "
            "(run_id, trade_date, ts_code, model_id, lgb_score, lgb_decile, feature_hash) "
            "VALUES (?, ?, ?, 'm', ?, NULL, 'h')",
            (f"00000000-0000-0000-0000-00000000000{i+1}", td, code, score),
        )
        db.execute(
            "INSERT INTO va_realized_returns "
            "(anomaly_date, ts_code, ret_t3, max_ret_5d, data_status) "
            "VALUES (?, ?, ?, ?, 'complete')",
            (td, code, ret_t3, mr5),
        )
    yield db
    db.close()


def test_by_lgb_score_bin_returns_one_row_per_populated_bucket(
    db_with_predictions: Database,
) -> None:
    rows, title = run_stats_query(
        db_with_predictions,
        from_date=None,
        to_date=None,
        by="lgb_score_bin",
    )
    assert "lgb_score_bin" in title
    buckets = sorted(r["bucket"] for r in rows)
    assert buckets == sorted({"0-30", "30-50", "50-70", "70-100"})


def test_by_lgb_score_bin_aggregates_t3_correctly(
    db_with_predictions: Database,
) -> None:
    rows, _ = run_stats_query(
        db_with_predictions, from_date=None, to_date=None, by="lgb_score_bin"
    )
    by_bucket = {r["bucket"]: r for r in rows}
    # 70-100 bucket has 2 samples (ret_t3 = 9.0 and 6.0).
    high = by_bucket["70-100"]
    assert high["n_samples"] == 2
    assert high["t3_mean"] == pytest.approx(7.5)
    # 0-30 bucket: a single -1.0 sample → 0% winrate.
    low = by_bucket["0-30"]
    assert low["n_samples"] == 1
    assert low["t3_winrate"] == pytest.approx(0.0)


def test_by_lgb_score_bin_respects_date_range(
    db_with_predictions: Database,
) -> None:
    rows, _ = run_stats_query(
        db_with_predictions,
        from_date="20260602",
        to_date="20260602",
        by="lgb_score_bin",
    )
    # Only one sample on 20260602 (the 80.0-score row).
    assert len(rows) == 1
    assert rows[0]["bucket"] == "70-100"
    assert rows[0]["n_samples"] == 1


def test_lgb_score_bins_definition_matches_design() -> None:
    labels = [b[0] for b in LGB_SCORE_BINS]
    assert labels == ["0-30", "30-50", "50-70", "70-100"]
