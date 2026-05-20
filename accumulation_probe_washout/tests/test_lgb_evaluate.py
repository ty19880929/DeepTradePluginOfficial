"""v0.6.0 — lgb evaluate (AUC / Top-K) + drift (PSI)."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accumulation_probe_washout.lgb import evaluate as _eval
from accumulation_probe_washout.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from accumulation_probe_washout.lgb.paths import datasets_dir
from accumulation_probe_washout.lgb.registry import ModelRecord, insert_model


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


def _seed_signal(db, *, trade_date, ts_code, acc=70):
    cand = {
        "ts_code": ts_code, "trade_date": trade_date, "phase": "launch_ready",
        "accumulation_score": acc, "probe_quality_score": 80.0,
        "washout_score": 60.0, "launch_setup_score": 75.0,
        "atr_10d_pct": 2.5,
    }
    db.execute(
        """
        INSERT INTO apw_signal_history
        (trade_date, ts_code, name, phase, accumulation_score,
         probe_quality_score, washout_score, launch_setup_score,
         raw_candidate_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [trade_date, ts_code, "测试", "launch_ready",
         acc, 80.0, 60.0, 75.0,
         json.dumps(cand, ensure_ascii=False), datetime.now()],
    )


def _seed_realized(db, *, signal_date, ts_code, label_t5):
    db.execute(
        """
        INSERT INTO apw_realized_returns
        (signal_date, ts_code, label_launch_t5, label_launch_t10,
         max_high_t5_pct, max_drawdown_t5_pct, data_status)
        VALUES (?, ?, ?, ?, 9.0, 3.0, 'complete')
        """,
        [signal_date, ts_code, label_t5, label_t5],
    )


def _train_real_booster(tmp_path) -> Path:
    import lightgbm as lgb
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(80, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    y = (df["f_acc_score"] > df["f_acc_score"].median()).astype(int).values
    clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    clf.fit(df, y)
    p = tmp_path / "real.txt"
    clf.booster_.save_model(str(p))
    return p


def _make_record(model_id, file_path, **overrides):
    base = dict(
        model_id=model_id, schema_version=SCHEMA_VERSION,
        train_start_date="20240101", train_end_date="20240601",
        n_samples=80, n_positive=40,
        feature_count=len(FEATURE_NAMES),
        feature_list_json=json.dumps(FEATURE_NAMES),
        hyperparams_json=json.dumps({}),
        label_source="label_launch_t5", label_threshold_pct=None,
        plugin_version="0.6.0", file_path=str(file_path),
    )
    base.update(overrides)
    return ModelRecord(**base)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_missing_raises(fresh_db):
    with pytest.raises(_eval.LgbEvalError):
        _eval.evaluate_model(fresh_db, start_date="20240501", end_date="20240601")


def test_evaluate_model_empty_window_raises(fresh_db, tmp_path):
    booster = _train_real_booster(tmp_path)
    insert_model(fresh_db, _make_record("M-1", booster), activate=True)
    with pytest.raises(_eval.LgbEvalError, match="no labeled rows"):
        _eval.evaluate_model(fresh_db, start_date="20240501", end_date="20240601")


def test_evaluate_model_writes_report(fresh_db, tmp_path):
    booster = _train_real_booster(tmp_path)
    insert_model(fresh_db, _make_record("M-1", booster), activate=True)
    # Seed at least one labeled (signal, realized) pair.
    for i in range(12):
        _seed_signal(fresh_db, trade_date="20240501",
                     ts_code=f"6000{i:02d}.SH", acc=50 + i * 2)
        _seed_realized(fresh_db, signal_date="20240501",
                       ts_code=f"6000{i:02d}.SH",
                       label_t5=1 if i % 2 else 0)

    result, path = _eval.evaluate_model(
        fresh_db, start_date="20240501", end_date="20240601", k=3,
    )
    assert result.model_id == "M-1"
    assert result.n_samples == 12
    assert path.exists()
    assert path.suffix == ".json"
    # Per-day Top-K rows present.
    assert len(result.per_day_topk) == 1
    assert result.per_day_topk[0]["topk"] == 3


# ---------------------------------------------------------------------------
# evaluate_drift — PSI
# ---------------------------------------------------------------------------


def _write_dataset_snapshot(model_id: str, *, n_rows: int, seed: int,
                             shift: float = 0.0) -> None:
    """Lay down a fake training-matrix parquet under datasets_dir()."""
    rng = np.random.RandomState(seed)
    data = {name: rng.rand(n_rows) + shift for name in FEATURE_NAMES}
    df = pd.DataFrame(data)
    df["__label__"] = rng.randint(0, 2, size=n_rows)
    df["__signal_date__"] = ["20240101"] * n_rows
    df.to_parquet(datasets_dir() / f"{model_id}.parquet", index=False)


def test_evaluate_drift_stable_label_when_distributions_identical(fresh_db, tmp_path):
    booster = _train_real_booster(tmp_path)
    insert_model(fresh_db, _make_record("base", booster), activate=False)
    insert_model(fresh_db, _make_record("cand", booster), activate=False)
    _write_dataset_snapshot("base", n_rows=200, seed=0)
    _write_dataset_snapshot("cand", n_rows=200, seed=0)
    result, path = _eval.evaluate_drift(
        fresh_db, baseline_model_id="base", candidate_model_id="cand",
    )
    assert path.exists()
    # Same seed → near-zero PSI → all stable.
    assert all(e.status in {"stable", "moderate"} for e in result.entries)
    assert result.entries[0].psi <= 0.25


def test_evaluate_drift_flags_shift_when_distributions_diverge(fresh_db, tmp_path):
    booster = _train_real_booster(tmp_path)
    insert_model(fresh_db, _make_record("base", booster), activate=False)
    insert_model(fresh_db, _make_record("cand", booster), activate=False)
    _write_dataset_snapshot("base", n_rows=400, seed=1, shift=0.0)
    _write_dataset_snapshot("cand", n_rows=400, seed=2, shift=2.0)  # +2 shift
    result, _ = _eval.evaluate_drift(
        fresh_db, baseline_model_id="base", candidate_model_id="cand",
    )
    # At least one feature should have shifted significantly.
    assert any(e.status == "shift" for e in result.entries)
    # Output sorted PSI desc.
    psis = [e.psi for e in result.entries]
    assert psis == sorted(psis, reverse=True)


def test_evaluate_drift_missing_snapshot_raises(fresh_db, tmp_path):
    booster = _train_real_booster(tmp_path)
    insert_model(fresh_db, _make_record("base", booster), activate=False)
    insert_model(fresh_db, _make_record("cand", booster), activate=False)
    # only base snapshot written
    _write_dataset_snapshot("base", n_rows=200, seed=0)
    with pytest.raises(_eval.LgbEvalError, match="snapshot missing"):
        _eval.evaluate_drift(
            fresh_db, baseline_model_id="base", candidate_model_id="cand",
        )
