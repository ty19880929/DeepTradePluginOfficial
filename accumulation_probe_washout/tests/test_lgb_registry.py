"""v0.5.0 — apw_lgb_models registry CRUD + activation invariant."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accumulation_probe_washout.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from accumulation_probe_washout.lgb.registry import (
    ModelRecord,
    delete_model,
    ensure_unique_model_id,
    get_active,
    get_model,
    insert_model,
    list_models,
    mint_model_id,
    set_active,
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


def _make_record(model_id: str, **overrides) -> ModelRecord:
    defaults = dict(
        model_id=model_id,
        schema_version=SCHEMA_VERSION,
        train_start_date="20240101",
        train_end_date="20240601",
        n_samples=1000,
        n_positive=200,
        feature_count=len(FEATURE_NAMES),
        feature_list_json=json.dumps(FEATURE_NAMES),
        hyperparams_json=json.dumps({"learning_rate": 0.05}),
        label_source="label_launch_t5",
        label_threshold_pct=None,
        plugin_version="0.5.0",
        file_path="/tmp/booster.txt",
        cv_auc_mean=0.65,
        cv_auc_std=0.02,
        cv_logloss_mean=0.55,
    )
    defaults.update(overrides)
    return ModelRecord(**defaults)


def test_insert_with_activate_makes_row_active(fresh_db):
    insert_model(fresh_db, _make_record("M-1"), activate=True)
    m = get_active(fresh_db)
    assert m is not None
    assert m.model_id == "M-1"
    assert m.is_active is True


def test_at_most_one_active_invariant(fresh_db):
    insert_model(fresh_db, _make_record("M-1"), activate=True)
    insert_model(fresh_db, _make_record("M-2"), activate=True)
    # Only one is_active row allowed.
    n_active = fresh_db.fetchone(
        "SELECT COUNT(*) FROM apw_lgb_models WHERE is_active = TRUE"
    )[0]
    assert n_active == 1
    assert get_active(fresh_db).model_id == "M-2"


def test_set_active_atomic_switch(fresh_db):
    insert_model(fresh_db, _make_record("M-1"), activate=True)
    insert_model(fresh_db, _make_record("M-2"), activate=False)
    ok = set_active(fresh_db, "M-2")
    assert ok is True
    assert get_active(fresh_db).model_id == "M-2"


def test_set_active_unknown_returns_false(fresh_db):
    assert set_active(fresh_db, "missing") is False


def test_get_model_round_trips_label_threshold(fresh_db):
    insert_model(fresh_db, _make_record(
        "M-custom", label_source="custom_t5", label_threshold_pct=8.0,
    ))
    m = get_model(fresh_db, "M-custom")
    assert m is not None
    assert m.label_source == "custom_t5"
    assert m.label_threshold_pct == 8.0


def test_list_models_orders_desc_by_created_at(fresh_db):
    insert_model(fresh_db, _make_record("M-1"), activate=False)
    insert_model(fresh_db, _make_record("M-2"), activate=False)
    models = list_models(fresh_db)
    # Most recent first.
    assert [m.model_id for m in models[:2]] == ["M-2", "M-1"]


def test_delete_model_removes_row(fresh_db):
    insert_model(fresh_db, _make_record("M-1"))
    assert delete_model(fresh_db, "M-1") is True
    assert get_model(fresh_db, "M-1") is None
    assert delete_model(fresh_db, "M-1") is False  # idempotent


def test_mint_model_id_format():
    assert mint_model_id(
        train_end_date="20240601",
        schema_version=1,
        git_commit="abc123",
    ) == "20240601_1_abc123"
    assert mint_model_id(
        train_end_date="20240601",
        schema_version=1,
        git_commit=None,
    ) == "20240601_1_nogit"


def test_ensure_unique_model_id_appends_suffix(fresh_db):
    insert_model(fresh_db, _make_record("20240601_1_abc"))
    new_id = ensure_unique_model_id(fresh_db, "20240601_1_abc")
    assert new_id == "20240601_1_abc-2"
    insert_model(fresh_db, _make_record(new_id))
    assert ensure_unique_model_id(fresh_db, "20240601_1_abc") == "20240601_1_abc-3"
