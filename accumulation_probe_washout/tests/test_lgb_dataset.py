"""v0.5.0 — dataset.collect_training_window over apw_signal_history.

Verifies the all-DB path: no Tushare client involved. Seeds rows into
apw_signal_history with raw_candidate_json + apw_realized_returns; calls
collect_training_window; asserts shape + label join + GroupKFold-friendly
split_groups.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.lgb.dataset import (
    collect_training_window,
    enumerate_signal_dates,
    make_fingerprint,
)
from accumulation_probe_washout.lgb.features import FEATURE_NAMES, SCHEMA_VERSION


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture(autouse=True)
def reroute_data_dir(tmp_path, monkeypatch):
    from deeptrade.core import paths as _paths
    monkeypatch.setattr(_paths, "db_path", lambda: tmp_path / "apw.duckdb")


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


def _seed_signal(db, *, trade_date, ts_code, accumulation=70.0,
                  probe_quality=80.0, atr_pct=2.5):
    cand = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "phase": "launch_ready",
        "accumulation_score": accumulation,
        "probe_quality_score": probe_quality,
        "washout_score": 60.0,
        "launch_setup_score": 75.0,
        "atr_10d_pct": atr_pct,
    }
    db.execute(
        """
        INSERT INTO apw_signal_history
        (trade_date, ts_code, name, phase, accumulation_score,
         probe_quality_score, washout_score, launch_setup_score,
         raw_candidate_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [trade_date, ts_code, f"测试{ts_code[:6]}", "launch_ready",
         accumulation, probe_quality, 60.0, 75.0,
         json.dumps(cand, ensure_ascii=False), datetime.now()],
    )


def _seed_realized(db, *, signal_date, ts_code, label_t5=1, label_t10=1):
    db.execute(
        """
        INSERT INTO apw_realized_returns
        (signal_date, ts_code, label_launch_t5, label_launch_t10,
         max_high_t5_pct, max_drawdown_t5_pct, data_status)
        VALUES (?, ?, ?, ?, ?, ?, 'complete')
        """,
        [signal_date, ts_code, label_t5, label_t10, 9.0, 5.0],
    )


def test_enumerate_signal_dates_returns_distinct_sorted(fresh_db):
    # Same date appears twice for two stocks → DISTINCT shrinks to one row.
    _seed_signal(fresh_db, trade_date="20240501", ts_code="600000.SH")
    _seed_signal(fresh_db, trade_date="20240501", ts_code="600001.SH")
    _seed_signal(fresh_db, trade_date="20240502", ts_code="600000.SH")
    dates = enumerate_signal_dates(fresh_db, start_date="20240101", end_date="20240601")
    assert dates == ["20240501", "20240502"]


def test_collect_training_window_joins_labels_and_features(fresh_db):
    for d in ("20240501", "20240502"):
        _seed_signal(fresh_db, trade_date=d, ts_code="600000.SH",
                      accumulation=70 + (1 if d == "20240502" else 0))
        _seed_signal(fresh_db, trade_date=d, ts_code="600001.SH")
        _seed_realized(fresh_db, signal_date=d, ts_code="600000.SH", label_t5=1)
        _seed_realized(fresh_db, signal_date=d, ts_code="600001.SH", label_t5=0)

    cfg = ApwConfig()
    ds, _cp = collect_training_window(
        fresh_db, start_date="20240101", end_date="20240601", cfg=cfg,
    )
    assert ds.schema_version == SCHEMA_VERSION
    assert ds.n_samples == 4
    assert ds.n_labeled == 4
    assert ds.n_positive == 2
    # split_groups must be one signal_date per row, in same order as features.
    assert sorted(ds.split_groups.unique().tolist()) == ["20240501", "20240502"]
    assert list(ds.feature_matrix.columns) == FEATURE_NAMES


def test_unlabeled_rows_kept_with_na_label(fresh_db):
    _seed_signal(fresh_db, trade_date="20240501", ts_code="600000.SH")
    # No realized row → label should land as NA.
    cfg = ApwConfig()
    ds, _ = collect_training_window(
        fresh_db, start_date="20240101", end_date="20240601", cfg=cfg,
    )
    assert ds.n_samples == 1
    assert ds.n_labeled == 0


def test_fingerprint_drives_checkpoint_dir(fresh_db, tmp_path):
    _seed_signal(fresh_db, trade_date="20240501", ts_code="600000.SH")
    cfg = ApwConfig()
    fp = make_fingerprint(
        start_date="20240101",
        end_date="20240601",
        label_source="label_launch_t5",
        label_threshold_pct=8.0,
        label_drawdown_threshold_pct=8.0,
        cfg=cfg,
    )
    digest = fp.digest()
    # First call lays down the checkpoint dir for this digest.
    _ds, cp = collect_training_window(
        fresh_db, start_date="20240101", end_date="20240601", cfg=cfg,
    )
    assert cp.root.name == digest
    assert (cp.root / "days" / "20240501.parquet").exists()


def test_fresh_flag_wipes_checkpoint(fresh_db):
    _seed_signal(fresh_db, trade_date="20240501", ts_code="600000.SH")
    cfg = ApwConfig()
    _, cp = collect_training_window(
        fresh_db, start_date="20240101", end_date="20240601", cfg=cfg,
    )
    p = cp.root
    _, cp2 = collect_training_window(
        fresh_db, start_date="20240101", end_date="20240601", cfg=cfg,
        fresh=True,
    )
    assert cp2.root == p  # same digest
    # Either way the day shard re-materialised after the wipe.
    assert (cp2.root / "days" / "20240501.parquet").exists()
