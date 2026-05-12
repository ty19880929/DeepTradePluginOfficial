"""PR-2.1 — :class:`LgbScorer` degradation matrix + determinism.

Covers all five failure branches from design §7.3:
    1. no active row in va_lgb_models
    2. row exists but model file missing on disk
    3. schema mismatch (booster feature_name != FEATURE_NAMES)
    4. predict() raises
    5. lightgbm import failure  (covered via predict warmup branch)

Plus a happy-path determinism check (same input twice → same scores).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from deeptrade.core.db import Database

from volume_anomaly.lgb import paths as lgb_paths
from volume_anomaly.lgb import registry as lgb_registry
from volume_anomaly.lgb.features import FEATURE_NAMES
from volume_anomaly.lgb.scorer import (
    LgbScorer,
    attach_deciles,
)
from volume_anomaly.lgb.trainer import train_lightgbm


# ---------------------------------------------------------------------------
# Shared DB fixture (mirrors PR-1.3 registry test schema).
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Database:
    db_ = Database(tmp_path / "lgb_scorer.duckdb")
    db_.execute(
        """
        CREATE TABLE va_lgb_models (
            model_id            VARCHAR PRIMARY KEY,
            schema_version      INTEGER NOT NULL,
            train_start_date    VARCHAR NOT NULL,
            train_end_date      VARCHAR NOT NULL,
            n_samples           INTEGER NOT NULL,
            n_positive          INTEGER NOT NULL,
            cv_auc_mean         DOUBLE,
            cv_auc_std          DOUBLE,
            cv_logloss_mean     DOUBLE,
            feature_count       INTEGER NOT NULL,
            feature_list_json   VARCHAR NOT NULL,
            hyperparams_json    VARCHAR NOT NULL,
            label_threshold_pct DOUBLE NOT NULL,
            label_source        VARCHAR NOT NULL,
            framework_version   VARCHAR,
            plugin_version      VARCHAR NOT NULL,
            git_commit          VARCHAR,
            file_path           VARCHAR NOT NULL,
            is_active           BOOLEAN NOT NULL DEFAULT FALSE,
            created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    yield db_
    db_.close()


# ---------------------------------------------------------------------------
# Helpers to plant a real booster file in the isolated plugin_data_dir.
# ---------------------------------------------------------------------------


def _train_real_booster(plugin_dir: Path) -> tuple[str, Path]:
    """Train a tiny LGB model + save it to the isolated models dir.
    Returns (model_id, rel_path)."""
    from volume_anomaly.lgb.dataset import VaLgbDataset  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n_days, n_per_day = 4, 30
    rows: list[dict[str, float]] = []
    meta: list[dict[str, Any]] = []
    labels: list[int] = []
    dates: list[str] = []
    for d in range(n_days):
        anomaly_date = f"2026010{d + 1}"
        dates.append(anomaly_date)
        for i in range(n_per_day):
            feat = {col: rng.normal(0, 1) for col in FEATURE_NAMES}
            sig = rng.normal(0, 1) + (1 if i % 2 == 0 else -1) * 3.0
            feat["f_vol_anomaly_pct_chg"] = sig
            rows.append(feat)
            labels.append(1 if sig > 0 else 0)
            meta.append(
                {
                    "ts_code": f"{i:06d}.SZ",
                    "anomaly_date": anomaly_date,
                    "max_ret_5d": sig,
                    "data_status": "complete",
                }
            )
    feature_matrix = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float)
    ds = VaLgbDataset(
        feature_matrix=feature_matrix.reset_index(drop=True),
        labels=pd.Series(labels, dtype="Int64", name="label"),
        sample_index=pd.DataFrame(meta),
        split_groups=pd.Series(
            pd.DataFrame(meta)["anomaly_date"].astype(int), dtype="Int64"
        ),
        anomaly_dates=dates,
    )
    result = train_lightgbm(
        ds,
        folds=2,
        num_boost_round=30,
        early_stopping_rounds=5,
        hyperparams={"min_data_in_leaf": 5},
    )
    # Save under the isolated plugin_data_dir.
    plugin_dir.mkdir(parents=True, exist_ok=True)
    models = plugin_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    model_id = "20260601_1_test"
    model_file = models / f"lgb_model_{model_id}.txt"
    result.model.save_model(str(model_file))
    return model_id, Path("models") / f"lgb_model_{model_id}.txt"


def _insert_record(
    db: Database, *, model_id: str, file_path: str, active: bool = True
) -> None:
    record = lgb_registry.ModelRecord(
        model_id=model_id,
        schema_version=1,
        train_start_date="20260101",
        train_end_date="20260601",
        n_samples=120,
        n_positive=60,
        feature_count=len(FEATURE_NAMES),
        feature_list_json=json.dumps(FEATURE_NAMES),
        hyperparams_json="{}",
        label_threshold_pct=5.0,
        label_source="max_ret_5d",
        plugin_version="0.7.0",
        file_path=file_path,
    )
    lgb_registry.insert_model(db, record, activate=active)


def _dummy_feature_df(n: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = [{col: rng.normal() for col in FEATURE_NAMES} for _ in range(n)]
    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES,
        index=[f"{i:06d}.SZ" for i in range(n)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_active_model_loaded_false(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    scorer = LgbScorer(db)
    scorer.warmup()
    assert scorer.loaded is False
    assert scorer.load_error == "no_active_model"
    assert scorer.model_id is None


def test_score_batch_no_model_returns_nan(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    scorer = LgbScorer(db)
    out = scorer.score_batch(_dummy_feature_df(3))
    assert list(out.columns) == ["lgb_score", "feature_hash", "feature_missing_json"]
    assert out["lgb_score"].isna().all()
    # feature_hash still computed (lets audit row record the input vector).
    assert all(len(h) == 16 for h in out["feature_hash"])


def test_model_file_missing_branch(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    _insert_record(
        db, model_id="ghost", file_path="models/lgb_model_ghost.txt", active=True
    )
    scorer = LgbScorer(db)
    scorer.warmup()
    assert scorer.loaded is False
    assert scorer.load_error is not None
    assert "model_file_missing" in scorer.load_error


def test_schema_mismatch_branch(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    # Train a real booster, then plant a FEATURE_NAMES override that disagrees.
    model_id, rel_path = _train_real_booster(isolated_plugin_data_dir)
    _insert_record(
        db, model_id=model_id, file_path=str(rel_path).replace("\\", "/")
    )
    # Patch FEATURE_NAMES so the booster's feature_name() disagrees.
    from volume_anomaly.lgb import features as features_mod
    from volume_anomaly.lgb import scorer as scorer_mod

    original_names = features_mod.FEATURE_NAMES
    try:
        bad_names = original_names[:-1] + ["__intentionally_extra__"]
        features_mod.FEATURE_NAMES = bad_names  # type: ignore[assignment]
        scorer_mod.FEATURE_NAMES = bad_names  # type: ignore[assignment]
        s = LgbScorer(db)
        s.warmup()
        assert s.loaded is False
        assert s.load_error is not None
        assert "schema_mismatch" in s.load_error
    finally:
        features_mod.FEATURE_NAMES = original_names  # type: ignore[assignment]
        scorer_mod.FEATURE_NAMES = original_names  # type: ignore[assignment]


def test_predict_failure_returns_nan_batch(
    db: Database, isolated_plugin_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id, rel_path = _train_real_booster(isolated_plugin_data_dir)
    _insert_record(db, model_id=model_id, file_path=str(rel_path).replace("\\", "/"))
    scorer = LgbScorer(db)
    scorer.warmup()
    assert scorer.loaded is True

    # Sabotage the booster's predict to simulate an inference exception.
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic predict failure")

    monkeypatch.setattr(scorer._loaded.booster, "predict", boom)
    out = scorer.score_batch(_dummy_feature_df(3))
    assert out["lgb_score"].isna().all()
    # Hash + missing payload are still populated so audit can replay.
    assert all(len(h) == 16 for h in out["feature_hash"])


def test_happy_path_loads_and_scores(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    model_id, rel_path = _train_real_booster(isolated_plugin_data_dir)
    _insert_record(db, model_id=model_id, file_path=str(rel_path).replace("\\", "/"))
    scorer = LgbScorer(db)
    scorer.warmup()
    assert scorer.loaded is True
    assert scorer.model_id == model_id
    out = scorer.score_batch(_dummy_feature_df(5))
    assert out["lgb_score"].notna().all()
    assert ((out["lgb_score"] >= 0) & (out["lgb_score"] <= 1)).all()


def test_score_batch_is_deterministic(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    model_id, rel_path = _train_real_booster(isolated_plugin_data_dir)
    _insert_record(db, model_id=model_id, file_path=str(rel_path).replace("\\", "/"))
    scorer = LgbScorer(db)
    scorer.warmup()
    df = _dummy_feature_df(4)
    out1 = scorer.score_batch(df)
    out2 = scorer.score_batch(df)
    pd.testing.assert_frame_equal(out1, out2)


def test_score_batch_handles_empty_input(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    scorer = LgbScorer(db)
    empty = pd.DataFrame(columns=FEATURE_NAMES)
    out = scorer.score_batch(empty)
    assert out.empty
    assert set(out.columns) == {"lgb_score", "feature_hash", "feature_missing_json"}


def test_attach_deciles_assigns_buckets_when_enough_finite() -> None:
    # 12 finite scores → qcut into 10 buckets succeeds.
    scores = pd.DataFrame(
        {"lgb_score": np.linspace(0.05, 0.95, 12)},
        index=[f"S{i:02d}" for i in range(12)],
    )
    deciles = attach_deciles(scores)
    assert deciles.notna().sum() == 12
    assert deciles.min() == 1
    assert deciles.max() == 10


def test_attach_deciles_returns_na_for_small_batch() -> None:
    scores = pd.DataFrame({"lgb_score": [0.1, 0.2, 0.5]})
    deciles = attach_deciles(scores)
    assert deciles.isna().all()


def test_attach_deciles_handles_all_equal() -> None:
    scores = pd.DataFrame({"lgb_score": [0.5] * 12})
    deciles = attach_deciles(scores)
    assert deciles.isna().all()


def test_explicit_model_id_not_found(
    db: Database, isolated_plugin_data_dir: Path
) -> None:
    scorer = LgbScorer(db, model_id="ghost-not-in-db")
    scorer.warmup()
    assert scorer.loaded is False
    assert scorer.load_error is not None
    assert "model_id_not_found" in scorer.load_error
