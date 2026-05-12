"""Tests for ``volume_anomaly.lgb.labels`` — design §5 / iteration PR-1.1."""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database
from volume_anomaly.lgb.labels import (
    LgbLabelError,
    VALID_LABEL_SOURCES,
    fetch_labels_for_date,
    fetch_labels_for_window,
)


# ---------------------------------------------------------------------------
# Fixture: a fresh DB pre-seeded with deterministic va_realized_returns rows.
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_realized_returns(tmp_path: Path) -> Database:
    db = Database(tmp_path / "lgb_labels.duckdb")
    db.execute(
        """
        CREATE TABLE va_realized_returns (
            anomaly_date  VARCHAR NOT NULL,
            ts_code       VARCHAR NOT NULL,
            t_close       DOUBLE,
            t1_close      DOUBLE,
            t3_close      DOUBLE,
            t5_close      DOUBLE,
            t10_close     DOUBLE,
            ret_t1        DOUBLE,
            ret_t3        DOUBLE,
            ret_t5        DOUBLE,
            ret_t10       DOUBLE,
            max_close_5d  DOUBLE,
            max_close_10d DOUBLE,
            max_ret_5d    DOUBLE,
            max_ret_10d   DOUBLE,
            max_dd_5d     DOUBLE,
            computed_at   TIMESTAMP,
            data_status   VARCHAR NOT NULL,
            PRIMARY KEY (anomaly_date, ts_code)
        )
        """
    )
    # (anomaly_date, ts_code, max_ret_5d, ret_t3, max_ret_10d, data_status)
    samples = [
        ("20260601", "000001.SZ", 6.2, 4.8, 8.0, "complete"),
        ("20260601", "000002.SZ", 4.99, 3.0, 5.5, "complete"),
        ("20260601", "000003.SZ", 5.00, 5.0, 6.0, "complete"),
        ("20260601", "000004.SZ", 5.01, 2.0, 9.0, "partial"),
        ("20260601", "000005.SZ", 2.0, 1.0, 3.5, "complete"),
        ("20260601", "000006.SZ", None, None, None, "pending"),
        ("20260602", "000007.SZ", 8.0, 7.5, 10.0, "complete"),
        ("20260602", "000008.SZ", 1.0, -1.0, 0.0, "complete"),
    ]
    for ad, code, mr5, ret_t3, mr10, status in samples:
        db.execute(
            "INSERT INTO va_realized_returns "
            "(anomaly_date, ts_code, max_ret_5d, ret_t3, max_ret_10d, data_status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ad, code, mr5, ret_t3, mr10, status),
        )
    yield db
    db.close()


# ---------------------------------------------------------------------------
# fetch_labels_for_date — single anomaly_date
# ---------------------------------------------------------------------------


def test_default_threshold_5pct_max_ret_5d(db_with_realized_returns: Database) -> None:
    labels = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="20260601",
        source="max_ret_5d",
        threshold_pct=5.0,
    )
    # 4.99 < 5.0 → 0; 5.00 = 5.0 → 1; 5.01 > 5.0 → 1; pending row absent.
    assert dict(labels) == {
        "000001.SZ": 1,  # 6.2 ≥ 5
        "000002.SZ": 0,  # 4.99 < 5
        "000003.SZ": 1,  # 5.00 = 5
        "000004.SZ": 1,  # 5.01 (partial still counts)
        "000005.SZ": 0,  # 2.0 < 5
        # 000006 is pending → filtered out
    }
    assert labels.dtype.name == "int8"


def test_label_threshold_boundary_4_99_5_00_5_01(
    db_with_realized_returns: Database,
) -> None:
    labels = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="20260601",
        threshold_pct=5.0,
    )
    # The boundary triple from samples should map exactly: < → 0, = → 1, > → 1.
    assert labels.loc["000002.SZ"] == 0
    assert labels.loc["000003.SZ"] == 1
    assert labels.loc["000004.SZ"] == 1


def test_source_switch_changes_label_distribution(
    db_with_realized_returns: Database,
) -> None:
    labels_5d = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="20260601",
        source="max_ret_5d",
        threshold_pct=5.0,
    )
    labels_t3 = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="20260601",
        source="ret_t3",
        threshold_pct=5.0,
    )
    labels_10d = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="20260601",
        source="max_ret_10d",
        threshold_pct=5.0,
    )
    # ret_t3 only puts 000003 (=5.0) over the bar.
    assert labels_t3.loc["000001.SZ"] == 0  # 4.8 < 5
    assert labels_t3.loc["000003.SZ"] == 1  # 5.0 = 5
    # max_ret_10d is looser → more positives.
    pos_10d = sum(labels_10d == 1)
    pos_5d = sum(labels_5d == 1)
    assert pos_10d >= pos_5d


def test_pending_rows_excluded(db_with_realized_returns: Database) -> None:
    labels = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="20260601",
    )
    assert "000006.SZ" not in labels.index


def test_empty_anomaly_date_returns_empty(
    db_with_realized_returns: Database,
) -> None:
    labels = fetch_labels_for_date(
        db_with_realized_returns,
        anomaly_date="29991231",
    )
    assert labels.empty


def test_invalid_source_raises(db_with_realized_returns: Database) -> None:
    with pytest.raises(LgbLabelError):
        fetch_labels_for_date(
            db_with_realized_returns,
            anomaly_date="20260601",
            source="not_a_real_source",
        )


def test_invalid_threshold_raises(db_with_realized_returns: Database) -> None:
    with pytest.raises(LgbLabelError):
        fetch_labels_for_date(
            db_with_realized_returns,
            anomaly_date="20260601",
            threshold_pct=0.0,
        )


def test_valid_sources_contract_matches_design() -> None:
    assert set(VALID_LABEL_SOURCES) == {"max_ret_5d", "ret_t3", "max_ret_10d"}


# ---------------------------------------------------------------------------
# fetch_labels_for_window — bulk variant
# ---------------------------------------------------------------------------


def test_window_returns_all_dates_combined(
    db_with_realized_returns: Database,
) -> None:
    df = fetch_labels_for_window(
        db_with_realized_returns,
        start_date="20260601",
        end_date="20260602",
        source="max_ret_5d",
        threshold_pct=5.0,
    )
    # Two dates × 6 labeled rows total (000006 pending filtered out).
    assert len(df) == 7
    assert set(df.columns) == {"anomaly_date", "ts_code", "label"}
    # 20260602 → 000007 (8.0 ≥ 5) yes, 000008 (1.0) no
    by_code = df[df.anomaly_date == "20260602"].set_index("ts_code")["label"].to_dict()
    assert by_code == {"000007.SZ": 1, "000008.SZ": 0}


def test_window_threshold_8pct_drops_marginal_positives(
    db_with_realized_returns: Database,
) -> None:
    df = fetch_labels_for_window(
        db_with_realized_returns,
        start_date="20260601",
        end_date="20260602",
        threshold_pct=8.0,
    )
    # Only 000007 (8.0) clears the bar at 8%.
    positives = df[df.label == 1].ts_code.tolist()
    assert positives == ["000007.SZ"]


def test_idempotent_repeated_calls(db_with_realized_returns: Database) -> None:
    a = fetch_labels_for_date(db_with_realized_returns, anomaly_date="20260601")
    b = fetch_labels_for_date(db_with_realized_returns, anomaly_date="20260601")
    assert a.equals(b)
