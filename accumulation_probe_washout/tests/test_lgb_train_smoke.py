"""v0.5.0 — end-to-end training smoke test.

Seeds enough rows into apw_signal_history + apw_realized_returns to clear
the lgb_train_min_samples gate, runs ``train_lightgbm``, asserts a
``apw_lgb_models`` row + on-disk booster + dataset parquet show up. Uses a
tiny FEATURE_NAMES vector — we don't care about model quality, only that
the wiring works.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig
from accumulation_probe_washout.lgb.dataset import collect_training_window
from accumulation_probe_washout.lgb.registry import get_active, list_models
from accumulation_probe_washout.lgb.trainer import LgbTrainError, train_lightgbm


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


def _seed(db, *, n_dates=10, per_date=12, positive_rate=0.3, rng_seed=42):
    """Plant n_dates * per_date training samples across distinct trade_dates."""
    rng = random.Random(rng_seed)
    now = datetime.now()
    for dnum in range(n_dates):
        td = f"2024050{dnum}" if dnum < 10 else f"202405{dnum}"
        for k in range(per_date):
            ts = f"{600000 + k:06d}.SH"
            # Vary a few features so LightGBM has signal to work with.
            cand = {
                "ts_code": ts,
                "trade_date": td,
                "phase": "launch_ready",
                "accumulation_score": rng.uniform(50, 90),
                "probe_quality_score": rng.uniform(60, 90),
                "washout_score": rng.uniform(50, 80),
                "launch_setup_score": rng.uniform(55, 85),
                "atr_10d_pct": rng.uniform(1.5, 4.5),
                "alpha_20d_pct": rng.uniform(-5, 10),
            }
            db.execute(
                """
                INSERT INTO apw_signal_history
                (trade_date, ts_code, name, phase, accumulation_score,
                 probe_quality_score, washout_score, launch_setup_score,
                 raw_candidate_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [td, ts, "测试", "launch_ready",
                 cand["accumulation_score"], cand["probe_quality_score"],
                 cand["washout_score"], cand["launch_setup_score"],
                 json.dumps(cand, ensure_ascii=False), now],
            )
            # Label correlates loosely with launch_setup_score.
            is_positive = (
                cand["launch_setup_score"] > 70 and rng.random() < positive_rate * 2
            )
            db.execute(
                """
                INSERT INTO apw_realized_returns
                (signal_date, ts_code, label_launch_t5, label_launch_t10,
                 max_high_t5_pct, max_drawdown_t5_pct, data_status)
                VALUES (?, ?, ?, ?, ?, ?, 'complete')
                """,
                [td, ts, 1 if is_positive else 0,
                 1 if is_positive else 0,
                 rng.uniform(2, 14), rng.uniform(2, 8)],
            )


def test_train_writes_registry_row_and_artefacts(fresh_db):
    _seed(fresh_db, n_dates=10, per_date=15)
    cfg = ApwConfig(lgb_train_min_samples=50, lgb_train_folds=3)
    ds, cp = collect_training_window(
        fresh_db, start_date="20240500", end_date="20240509", cfg=cfg,
    )
    assert ds.n_labeled >= 50
    res = train_lightgbm(
        fresh_db, dataset=ds, cfg=cfg, plugin_version="0.5.0-test",
        activate=True,
    )
    assert res.booster_path.exists()
    assert res.dataset_path.exists()
    active = get_active(fresh_db)
    assert active is not None
    assert active.model_id == res.model_id
    assert active.n_samples == res.n_samples
    cp.discard()


def test_train_raises_on_undersized_dataset(fresh_db):
    _seed(fresh_db, n_dates=1, per_date=5)
    cfg = ApwConfig(lgb_train_min_samples=1000, lgb_train_folds=3)
    ds, _ = collect_training_window(
        fresh_db, start_date="20240500", end_date="20240509", cfg=cfg,
    )
    with pytest.raises(LgbTrainError, match="labeled samples"):
        train_lightgbm(
            fresh_db, dataset=ds, cfg=cfg, plugin_version="t", activate=False,
        )


def test_train_raises_on_degenerate_labels(fresh_db):
    # Force all labels to 1 → degenerate.
    now = datetime.now()
    for k in range(60):
        ts = f"{600000 + k:06d}.SH"
        td = "20240501"
        cand = {"ts_code": ts, "trade_date": td, "phase": "launch_ready",
                "accumulation_score": 70.0 + k * 0.1}
        fresh_db.execute(
            """
            INSERT INTO apw_signal_history
            (trade_date, ts_code, name, phase, accumulation_score,
             probe_quality_score, washout_score, launch_setup_score,
             raw_candidate_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [td, ts, "x", "launch_ready", 70.0, 80.0, 60.0, 75.0,
             json.dumps(cand), now],
        )
        fresh_db.execute(
            """
            INSERT INTO apw_realized_returns
            (signal_date, ts_code, label_launch_t5, label_launch_t10,
             max_high_t5_pct, max_drawdown_t5_pct, data_status)
            VALUES (?, ?, 1, 1, 12.0, 3.0, 'complete')
            """,
            [td, ts],
        )
    cfg = ApwConfig(lgb_train_min_samples=50, lgb_train_folds=2)
    ds, _ = collect_training_window(
        fresh_db, start_date="20240500", end_date="20240509", cfg=cfg,
    )
    with pytest.raises(LgbTrainError, match="degenerate"):
        train_lightgbm(
            fresh_db, dataset=ds, cfg=cfg, plugin_version="t", activate=False,
        )
