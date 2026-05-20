"""v0.5.0 — labels.fetch_labels_for_window over apw_realized_returns."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from accumulation_probe_washout.lgb.labels import (
    LgbLabelError,
    VALID_LABEL_SOURCES,
    fetch_labels_for_window,
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


def _seed_realized(
    db, *, signal_date, ts_code, label_launch_t5=None, label_launch_t10=None,
    max_high_t5=None, max_dd_t5=None, data_status="complete",
):
    db.execute(
        """
        INSERT INTO apw_realized_returns
        (signal_date, ts_code, label_launch_t5, label_launch_t10,
         max_high_t5_pct, max_drawdown_t5_pct, data_status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [signal_date, ts_code, label_launch_t5, label_launch_t10,
         max_high_t5, max_dd_t5, data_status],
    )


def test_valid_label_sources_exact():
    assert set(VALID_LABEL_SOURCES) == {
        "label_launch_t5", "label_launch_t10", "custom_t5",
    }


def test_unknown_source_raises():
    with pytest.raises(LgbLabelError):
        fetch_labels_for_window(
            None, start_date="20240101", end_date="20240601", source="bogus"
        )


def test_custom_t5_requires_thresholds():
    with pytest.raises(LgbLabelError, match="custom_t5 requires"):
        fetch_labels_for_window(
            None, start_date="20240101", end_date="20240601",
            source="custom_t5",
        )


def test_label_launch_t5_pulls_pre_computed(fresh_db):
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600000.SH", label_launch_t5=1)
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600001.SH", label_launch_t5=0)
    df = fetch_labels_for_window(
        fresh_db, start_date="20240101", end_date="20240601",
        source="label_launch_t5",
    )
    assert len(df) == 2
    by_code = dict(zip(df["ts_code"], df["label"]))
    assert by_code["600000.SH"] == 1
    assert by_code["600001.SH"] == 0


def test_null_label_dropped(fresh_db):
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600000.SH", label_launch_t5=None)
    df = fetch_labels_for_window(
        fresh_db, start_date="20240101", end_date="20240601",
        source="label_launch_t5",
    )
    assert df.empty


def test_pending_status_dropped(fresh_db):
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600000.SH", label_launch_t5=1, data_status="pending")
    df = fetch_labels_for_window(
        fresh_db, start_date="20240101", end_date="20240601",
        source="label_launch_t5",
    )
    assert df.empty


def test_custom_t5_derives_positive_from_thresholds(fresh_db):
    # Positive: high 10% with 5% dd, given thresholds (8.0, 8.0)
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600000.SH", max_high_t5=10.0, max_dd_t5=5.0)
    # Negative: high only 5%
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600001.SH", max_high_t5=5.0, max_dd_t5=3.0)
    # Negative: dd too big
    _seed_realized(fresh_db, signal_date="20240501",
                   ts_code="600002.SH", max_high_t5=12.0, max_dd_t5=10.0)
    df = fetch_labels_for_window(
        fresh_db, start_date="20240101", end_date="20240601",
        source="custom_t5", threshold_pct=8.0, drawdown_threshold_pct=8.0,
    )
    by_code = dict(zip(df["ts_code"], df["label"]))
    assert by_code["600000.SH"] == 1
    assert by_code["600001.SH"] == 0
    assert by_code["600002.SH"] == 0


def test_window_filter(fresh_db):
    _seed_realized(fresh_db, signal_date="20240101",
                   ts_code="600000.SH", label_launch_t5=1)
    _seed_realized(fresh_db, signal_date="20240601",
                   ts_code="600000.SH", label_launch_t5=0)
    df = fetch_labels_for_window(
        fresh_db, start_date="20240401", end_date="20240701",
        source="label_launch_t5",
    )
    assert len(df) == 1
    assert df.iloc[0]["signal_date"] == "20240601"
