"""Tests for :mod:`limit_up_board.lgb.cleanup`.

Verifies that ``purge_lgb_artifacts`` correctly tears down the requested
artifact sets:

* ``--models`` removes model files + meta.json + ``latest.txt`` + the
  ``lub_lgb_models`` rows (including the currently active one).
* ``--datasets`` removes ``datasets/*.parquet`` only — models and DB rows
  stay intact.
* ``--predictions`` truncates ``lub_lgb_predictions``.
* No-op call (all flags False) returns an all-zero ``PurgeReport``.

These run against a temp ``~/.deeptrade`` via the same isolated_home /
lgb_db fixtures the other LGB tests use.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from deeptrade.core.db import Database

from limit_up_board.lgb import paths as lgb_paths
from limit_up_board.lgb.cleanup import (
    PurgeReport,
    count_artifacts,
    purge_lgb_artifacts,
)
from limit_up_board.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from limit_up_board.lgb.registry import ModelRecord, insert_model

MIGRATION_FILE = (
    Path(__file__).resolve().parents[1] / "migrations" / "20260601_001_lgb_tables.sql"
)


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_lgb_scorer.py style)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "deeptrade-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))
    return home


@pytest.fixture
def lgb_db(isolated_home: Path) -> Database:  # noqa: ARG001 — chaining
    from deeptrade.core import paths

    db = Database(paths.db_path())
    sql_text = MIGRATION_FILE.read_text(encoding="utf-8")
    for stmt in sql_text.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    return db


def _seed_artifacts(db: Database, model_id: str = "20260530_1_seed") -> dict[str, Path]:
    """Create matching files + registry row + audit rows for one model."""
    lgb_paths.ensure_layout()
    model_path = lgb_paths.models_dir() / lgb_paths.model_file_name(model_id)
    meta_path = lgb_paths.models_dir() / lgb_paths.meta_file_name(model_id)
    dataset_path = lgb_paths.datasets_dir() / lgb_paths.dataset_file_name(model_id)
    latest_path = lgb_paths.latest_pointer()

    model_path.write_bytes(b"fake-booster")
    meta_path.write_text(
        json.dumps({"model_id": model_id, "feature_names": FEATURE_NAMES}),
        encoding="utf-8",
    )
    dataset_path.write_bytes(b"fake-parquet")
    rel = model_path.relative_to(lgb_paths.plugin_data_dir())
    latest_path.write_text(str(rel).replace("\\", "/"), encoding="utf-8")

    insert_model(
        db,
        ModelRecord(
            model_id=model_id,
            schema_version=SCHEMA_VERSION,
            train_start_date="20260101",
            train_end_date="20260530",
            n_samples=100,
            n_positive=30,
            feature_count=len(FEATURE_NAMES),
            feature_list_json=json.dumps(list(FEATURE_NAMES)),
            hyperparams_json="{}",
            plugin_version="0.5.1",
            file_path=str(rel).replace("\\", "/"),
        ),
        activate=True,
    )
    # Seed two audit rows so we can assert truncation later.
    # Run_id 派生自 model_id，避免单测试内多次 seed 不同 model 撞 PK(run_id, ts_code)。
    run_id = str(uuid.uuid5(uuid.NAMESPACE_OID, model_id))
    for ts in ("600519.SH", "000001.SZ"):
        db.execute(
            "INSERT INTO lub_lgb_predictions("
            "run_id, trade_date, ts_code, model_id, lgb_score, lgb_decile, "
            "feature_hash, feature_missing_json) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                "20260530",
                ts,
                model_id,
                0.5,
                5,
                "abc",
                "[]",
            ),
        )
    return {
        "model": model_path,
        "meta": meta_path,
        "dataset": dataset_path,
        "latest": latest_path,
    }


# ---------------------------------------------------------------------------
# No-op / preview
# ---------------------------------------------------------------------------


def test_noop_returns_zero_report(lgb_db: Database) -> None:
    report = purge_lgb_artifacts(lgb_db)
    assert isinstance(report, PurgeReport)
    assert report.total_files_removed == 0
    assert report.n_model_rows == 0
    assert report.n_prediction_rows == 0
    assert report.errors == []


def test_count_artifacts_reports_seeded_state(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    _seed_artifacts(lgb_db)
    preview = count_artifacts(lgb_db)
    assert preview.n_model_files == 1
    assert preview.n_meta_files == 1
    assert preview.n_dataset_files == 1
    assert preview.latest_pointer_removed is True  # pointer file exists
    assert preview.n_model_rows == 1
    assert preview.n_prediction_rows == 2


# ---------------------------------------------------------------------------
# --datasets only
# ---------------------------------------------------------------------------


def test_purge_datasets_only(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    paths_ = _seed_artifacts(lgb_db)
    report = purge_lgb_artifacts(lgb_db, datasets=True)

    assert report.n_dataset_files == 1
    assert report.n_model_files == 0
    assert report.n_meta_files == 0
    assert report.n_model_rows == 0
    assert report.n_prediction_rows == 0
    # Files: dataset gone, model + meta + latest survive.
    assert not paths_["dataset"].exists()
    assert paths_["model"].exists()
    assert paths_["meta"].exists()
    assert paths_["latest"].exists()
    # DB: registry + predictions intact.
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_models")
    assert int(row[0]) == 1
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_predictions")
    assert int(row[0]) == 2


# ---------------------------------------------------------------------------
# --models
# ---------------------------------------------------------------------------


def test_purge_models_removes_files_and_rows(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    paths_ = _seed_artifacts(lgb_db)
    report = purge_lgb_artifacts(lgb_db, models=True)

    assert report.n_model_files == 1
    assert report.n_meta_files == 1
    assert report.latest_pointer_removed is True
    assert report.n_model_rows == 1
    # dataset + predictions untouched
    assert report.n_dataset_files == 0
    assert report.n_prediction_rows == 0
    assert paths_["dataset"].exists()
    assert not paths_["model"].exists()
    assert not paths_["meta"].exists()
    assert not paths_["latest"].exists()
    # DB: model rows gone, predictions remain
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_models")
    assert int(row[0]) == 0
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_predictions")
    assert int(row[0]) == 2


# ---------------------------------------------------------------------------
# --predictions
# ---------------------------------------------------------------------------


def test_purge_predictions_truncates_audit_only(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    paths_ = _seed_artifacts(lgb_db)
    report = purge_lgb_artifacts(lgb_db, predictions=True)

    assert report.n_prediction_rows == 2
    assert report.n_model_files == 0
    assert report.n_model_rows == 0
    # Everything else intact
    assert paths_["model"].exists()
    assert paths_["dataset"].exists()
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_models")
    assert int(row[0]) == 1
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_predictions")
    assert int(row[0]) == 0


# ---------------------------------------------------------------------------
# Full scorched-earth
# ---------------------------------------------------------------------------


def test_purge_all_clears_everything(
    isolated_home: Path, lgb_db: Database  # noqa: ARG001
) -> None:
    paths_ = _seed_artifacts(lgb_db, model_id="m1")
    _seed_artifacts(lgb_db, model_id="m2")  # second model on the same DB
    report = purge_lgb_artifacts(
        lgb_db, datasets=True, models=True, predictions=True
    )

    # insert_model(activate=True) flips the previous active to False but keeps
    # the row, so we expect 2 rows total → 2 truncated; 2 prediction rows per
    # seed call → 4 truncated.
    assert report.n_model_files == 2
    assert report.n_meta_files == 2
    assert report.n_dataset_files == 2
    assert report.latest_pointer_removed is True
    assert report.n_model_rows == 2
    assert report.n_prediction_rows == 4
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_models")
    assert int(row[0]) == 0
    row = lgb_db.fetchone("SELECT COUNT(*) FROM lub_lgb_predictions")
    assert int(row[0]) == 0
    assert not paths_["model"].exists()
    assert not paths_["dataset"].exists()


# ---------------------------------------------------------------------------
# Resilience: missing tables / dirs don't crash
# ---------------------------------------------------------------------------


def test_purge_when_models_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator manually deleted ~/.deeptrade/limit_up_board/ — purge should still run."""
    home = tmp_path / "fresh-home"
    home.mkdir()
    monkeypatch.setenv("DEEPTRADE_HOME", str(home))

    from deeptrade.core import paths as core_paths

    db = Database(core_paths.db_path())
    for stmt in MIGRATION_FILE.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())

    report = purge_lgb_artifacts(db, models=True, datasets=True, predictions=True)
    assert report.total_files_removed == 0
    assert report.n_model_rows == 0
    assert report.errors == []
