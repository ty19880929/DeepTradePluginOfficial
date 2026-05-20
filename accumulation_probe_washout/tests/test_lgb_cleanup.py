"""v0.5.0 — lgb prune / purge cleanup semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accumulation_probe_washout.lgb.cleanup import prune_models, purge
from accumulation_probe_washout.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from accumulation_probe_washout.lgb.paths import datasets_dir, models_dir, checkpoints_dir
from accumulation_probe_washout.lgb.registry import (
    ModelRecord, get_active, get_model, insert_model, list_models,
)


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


def _make_model(model_id: str, **overrides) -> ModelRecord:
    base = dict(
        model_id=model_id,
        schema_version=SCHEMA_VERSION,
        train_start_date="20240101",
        train_end_date="20240601",
        n_samples=1000, n_positive=200,
        feature_count=len(FEATURE_NAMES),
        feature_list_json=json.dumps(FEATURE_NAMES),
        hyperparams_json=json.dumps({}),
        label_source="label_launch_t5", label_threshold_pct=None,
        plugin_version="t",
        file_path=str(models_dir() / f"{model_id}.txt"),
    )
    base.update(overrides)
    return ModelRecord(**base)


def _touch_file(p: Path, content: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_prune_negative_keep_raises(fresh_db):
    with pytest.raises(ValueError):
        prune_models(fresh_db, keep=-1)


def test_prune_preserves_active_even_when_outside_keep(fresh_db):
    """Active row must never be deleted by prune."""
    for i in range(5):
        rec = _make_model(f"M-{i}")
        # Touch on-disk file so prune has something to remove.
        _touch_file(Path(rec.file_path))
        insert_model(fresh_db, rec, activate=(i == 0))
    # keep=0 → expect active row kept, all four non-active deleted.
    rep = prune_models(fresh_db, keep=0)
    assert "M-0" in rep.kept
    assert len(rep.deleted) == 4
    # Active model still queryable.
    assert get_active(fresh_db).model_id == "M-0"


def test_prune_keeps_n_most_recent_non_active(fresh_db):
    for i in range(5):
        rec = _make_model(f"M-{i}")
        _touch_file(Path(rec.file_path))
        insert_model(fresh_db, rec, activate=False)
    # Reactivate first one so it's not considered "non-active".
    fresh_db.execute(
        "UPDATE apw_lgb_models SET is_active = TRUE WHERE model_id = ?",
        ["M-0"],
    )
    rep = prune_models(fresh_db, keep=2)
    # M-0 (active) + 2 most-recent non-active (M-4, M-3) kept.
    assert set(rep.kept) == {"M-0", "M-4", "M-3"}
    assert set(rep.deleted) == {"M-1", "M-2"}


def test_prune_removes_booster_file(fresh_db):
    rec = _make_model("M-1")
    p = Path(rec.file_path)
    _touch_file(p)
    insert_model(fresh_db, rec, activate=False)
    prune_models(fresh_db, keep=0)
    assert not p.exists()
    assert get_model(fresh_db, "M-1") is None


def test_purge_models_clears_dir_and_table(fresh_db):
    for i in range(3):
        rec = _make_model(f"M-{i}")
        _touch_file(Path(rec.file_path))
        insert_model(fresh_db, rec, activate=(i == 0))
    reports = purge(fresh_db, models=True)
    assert reports[0].scope == "models"
    assert reports[0].rows_removed == 3
    assert list_models(fresh_db) == []


def test_purge_datasets_only_clears_files_not_registry(fresh_db):
    rec = _make_model("M-1")
    insert_model(fresh_db, rec, activate=True)
    snap = datasets_dir() / "snap.parquet"
    _touch_file(snap, "snap")
    reports = purge(fresh_db, datasets=True)
    assert reports[0].scope == "datasets"
    assert reports[0].files_removed >= 1
    # Registry untouched.
    assert get_model(fresh_db, "M-1") is not None


def test_purge_checkpoints_clears_dir(fresh_db):
    f = checkpoints_dir() / "abc" / "days" / "20240501.parquet"
    _touch_file(f, "shard")
    reports = purge(fresh_db, checkpoints=True)
    assert reports[0].scope == "checkpoints"
    assert reports[0].files_removed >= 1


def test_purge_predictions_silent_until_pr4(fresh_db):
    """apw_lgb_predictions table doesn't exist in PR-3 — purge no-ops."""
    reports = purge(fresh_db, predictions=True)
    assert reports[0].scope == "predictions"
    assert reports[0].rows_removed == 0
