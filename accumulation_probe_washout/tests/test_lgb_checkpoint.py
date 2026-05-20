"""v0.5.0 — checkpoint fingerprint stability + resume semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from accumulation_probe_washout.lgb.checkpoint import (
    CheckpointFingerprint,
    DayShard,
    META_COLUMNS,
    open_checkpoint,
)
from accumulation_probe_washout.lgb.features import FEATURE_NAMES


@pytest.fixture
def fingerprint() -> CheckpointFingerprint:
    return CheckpointFingerprint(
        start_date="20240101",
        end_date="20240601",
        label_source="label_launch_t5",
        label_threshold_pct=8.0,
        label_drawdown_threshold_pct=8.0,
        schema_version=1,
        baseline_index_code="000300.SH",
        volume_adjust_enabled=True,
        base_lookback_trade_days=120,
        probe_lookback_trade_days=40,
        accumulation_lookback_trade_days=60,
    )


@pytest.fixture(autouse=True)
def reroute_data_dir(tmp_path, monkeypatch):
    """Redirect ``db_path`` so each test gets a fresh on-disk root.

    The lgb.paths module anchors all sub-directories at
    ``paths.db_path().parent / 'accumulation_probe_washout'``; pointing
    ``db_path()`` at the per-test tmp dir isolates the test from any real
    ``~/.deeptrade`` state.
    """
    from deeptrade.core import paths as _paths
    monkeypatch.setattr(_paths, "db_path", lambda: tmp_path / "apw.duckdb")


def test_fingerprint_digest_deterministic(fingerprint):
    a = fingerprint.digest()
    b = fingerprint.digest()
    assert a == b
    assert len(a) == 16  # 8-byte BLAKE2b hex


def test_fingerprint_digest_changes_with_label_source(fingerprint):
    other = CheckpointFingerprint(**{**fingerprint.__dict__,
                                      "label_source": "label_launch_t10"})
    assert other.digest() != fingerprint.digest()


def test_fingerprint_digest_changes_with_schema_version(fingerprint):
    other = CheckpointFingerprint(**{**fingerprint.__dict__,
                                      "schema_version": 99})
    assert other.digest() != fingerprint.digest()


def test_open_checkpoint_creates_dir_layout(fingerprint, tmp_path):
    cp = open_checkpoint(fingerprint)
    assert cp.root.exists()
    assert (cp.root / "meta.json").exists()
    assert cp.days_dir.exists()


def test_write_and_read_day_shard_round_trip(fingerprint):
    cp = open_checkpoint(fingerprint)
    # 3 rows × 2 columns; reindex against FEATURE_NAMES so the parquet writer
    # gets the canonical column set.
    raw_feat = pd.DataFrame(
        {
            "f_acc_score": [70.0, 80.0, 60.0],
            "f_probe_quality": [85.0, 88.0, 72.0],
        },
        index=pd.Index(["A", "B", "C"], name="ts_code"),
    )
    feat = raw_feat.reindex(columns=FEATURE_NAMES)
    meta = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "signal_date": ["20240501"] * 3,
            "label": [1, 0, 1],
            "data_status": ["complete"] * 3,
        },
        columns=META_COLUMNS,
    )
    cp.write_day(DayShard(signal_date="20240501",
                          feature_matrix=feat, sample_meta=meta))
    assert cp.existing_dates() == {"20240501"}
    shards = cp.read_all()
    assert len(shards) == 1
    assert shards[0].signal_date == "20240501"
    assert list(shards[0].feature_matrix.index) == ["A", "B", "C"]
    # Round-trip preserves the metric columns.
    assert shards[0].feature_matrix.loc["A", "f_acc_score"] == 70.0


def test_existing_dates_supports_resume(fingerprint):
    cp = open_checkpoint(fingerprint)
    feat = pd.DataFrame(index=pd.Index(["A"], name="ts_code"),
                        columns=FEATURE_NAMES, dtype=float)
    meta = pd.DataFrame(
        [{"ts_code": "A", "signal_date": "20240501", "label": 1,
          "data_status": "complete"}],
        columns=META_COLUMNS,
    )
    cp.write_day(DayShard("20240501", feat, meta))
    cp.write_day(DayShard("20240502", feat, meta))
    assert cp.existing_dates() == {"20240501", "20240502"}


def test_discard_removes_directory(fingerprint):
    cp = open_checkpoint(fingerprint)
    feat = pd.DataFrame(index=pd.Index(["A"], name="ts_code"),
                        columns=FEATURE_NAMES, dtype=float)
    meta = pd.DataFrame(
        [{"ts_code": "A", "signal_date": "20240501", "label": 1,
          "data_status": "complete"}],
        columns=META_COLUMNS,
    )
    cp.write_day(DayShard("20240501", feat, meta))
    assert cp.root.exists()
    cp.discard()
    assert not cp.root.exists()
