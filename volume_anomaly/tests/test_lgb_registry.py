"""PR-1.3 — va_lgb_models CRUD (insert / list / activate / delete + mint id).

Plain DB tests; no Tushare / lightgbm required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptrade.core.db import Database
from volume_anomaly.lgb import registry


@pytest.fixture
def db(tmp_path: Path) -> Database:
    db_ = Database(tmp_path / "lgb_registry.duckdb")
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


def _make_record(model_id: str = "20260601_1_abc123", **overrides) -> registry.ModelRecord:
    base = registry.ModelRecord(
        model_id=model_id,
        schema_version=1,
        train_start_date="20260101",
        train_end_date="20260601",
        n_samples=1200,
        n_positive=320,
        feature_count=52,
        feature_list_json="[\"f_vol_anomaly_pct_chg\"]",
        hyperparams_json="{\"seed\": 42}",
        label_threshold_pct=5.0,
        label_source="max_ret_5d",
        plugin_version="0.7.0",
        file_path="models/lgb_model_20260601_1_abc123.txt",
        cv_auc_mean=0.65,
        cv_auc_std=0.02,
        cv_logloss_mean=0.6,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_insert_then_list_returns_one_row(db: Database) -> None:
    registry.insert_model(db, _make_record(), activate=True)
    rows = registry.list_models(db)
    assert len(rows) == 1
    assert rows[0].is_active
    assert rows[0].label_source == "max_ret_5d"


def test_insert_activates_exclusively(db: Database) -> None:
    registry.insert_model(db, _make_record("aaa"), activate=True)
    registry.insert_model(db, _make_record("bbb"), activate=True)
    active = registry.get_active(db)
    assert active is not None
    assert active.model_id == "bbb"
    # Only one active row at any time.
    by_id = {r.model_id: r.is_active for r in registry.list_models(db)}
    assert sum(1 for v in by_id.values() if v) == 1


def test_set_active_atomically_switches(db: Database) -> None:
    registry.insert_model(db, _make_record("aaa"), activate=True)
    registry.insert_model(db, _make_record("bbb"), activate=False)
    ok = registry.set_active(db, "aaa")
    assert ok is True
    active = registry.get_active(db)
    assert active is not None and active.model_id == "aaa"
    # Only one active row.
    actives = [r.model_id for r in registry.list_models(db) if r.is_active]
    assert actives == ["aaa"]


def test_set_active_returns_false_for_missing_id(db: Database) -> None:
    registry.insert_model(db, _make_record("aaa"), activate=True)
    assert registry.set_active(db, "nonexistent") is False
    # Original active row unchanged.
    active = registry.get_active(db)
    assert active is not None and active.model_id == "aaa"


def test_deactivate_all(db: Database) -> None:
    registry.insert_model(db, _make_record("aaa"), activate=True)
    registry.deactivate_all(db)
    assert registry.get_active(db) is None


def test_delete_model_removes_row(db: Database) -> None:
    registry.insert_model(db, _make_record("aaa"))
    assert registry.delete_model(db, "aaa") is True
    assert registry.list_models(db) == []
    # Idempotent on missing row.
    assert registry.delete_model(db, "aaa") is False


def test_mint_model_id_format() -> None:
    mid = registry.mint_model_id(
        train_end_date="20260601",
        schema_version=1,
        git_commit="a3f2c1",
    )
    assert mid == "20260601_1_a3f2c1"
    # No git commit → "nogit" sentinel.
    mid2 = registry.mint_model_id(
        train_end_date="20260601", schema_version=1, git_commit=None
    )
    assert mid2.endswith("_nogit")


def test_ensure_unique_model_id_appends_suffix(db: Database) -> None:
    registry.insert_model(db, _make_record("20260601_1_abc123"), activate=False)
    unique = registry.ensure_unique_model_id(db, "20260601_1_abc123")
    assert unique == "20260601_1_abc123-2"
    # Insert the -2 and request again → -3.
    registry.insert_model(db, _make_record("20260601_1_abc123-2"), activate=False)
    unique2 = registry.ensure_unique_model_id(db, "20260601_1_abc123")
    assert unique2 == "20260601_1_abc123-3"
