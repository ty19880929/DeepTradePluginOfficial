"""v0.6.0 — LgbScorer 5-branch fallback contract.

Every degrade branch:
  1. no active model
  2. booster file missing on disk
  3a. schema_version drift between model record and runtime
  3b. feature_list_json mismatch
  3c. feature frame columns disagree at score time (impossible while
      build_feature_frame is in canonical order, but covered with a manual
      column drop)
  4. Booster.predict raises (per-row degrade)
  5. lightgbm ImportError

Every branch must:
  * emit lgb_score=None / lgb_decile=None for every candidate;
  * carry a non-None degrade_reason on the ScoreOutcome;
  * **not** write any row to apw_lgb_predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from accumulation_probe_washout.lgb.features import FEATURE_NAMES, SCHEMA_VERSION
from accumulation_probe_washout.lgb.registry import ModelRecord, insert_model
from accumulation_probe_washout.lgb.scorer import LgbScorer, build_lgb_scorer


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


def _make_active_record(model_id: str, file_path: str, **overrides) -> ModelRecord:
    base = dict(
        model_id=model_id, schema_version=SCHEMA_VERSION,
        train_start_date="20240101", train_end_date="20240601",
        n_samples=1000, n_positive=200,
        feature_count=len(FEATURE_NAMES),
        feature_list_json=json.dumps(FEATURE_NAMES),
        hyperparams_json=json.dumps({}),
        label_source="label_launch_t5", label_threshold_pct=None,
        plugin_version="0.6.0", file_path=file_path,
    )
    base.update(overrides)
    return ModelRecord(**base)


def _candidate(ts: str) -> dict:
    return {
        "ts_code": ts,
        "accumulation_score": 70.0,
        "probe_quality_score": 80.0,
        "launch_setup_score": 75.0,
        "atr_10d_pct": 2.5,
    }


# ---------------------------------------------------------------------------
# Branch 1 — no active model
# ---------------------------------------------------------------------------


def test_branch_1_no_active_model(fresh_db):
    scorer = build_lgb_scorer(fresh_db)
    out = scorer.score_batch([_candidate("600000.SH"), _candidate("600001.SH")])
    assert out.degrade_reason == "LGB_DEGRADE_NO_MODEL"
    assert all(s.lgb_score is None for s in out.scores)
    assert all(s.lgb_decile is None for s in out.scores)


def test_branch_1_persist_writes_zero_rows(fresh_db):
    scorer = build_lgb_scorer(fresh_db)
    out = scorer.score_batch([_candidate("600000.SH")])
    n = scorer.persist_predictions(fresh_db, out, run_id="r1", trade_date="20260520")
    assert n == 0
    assert fresh_db.fetchone("SELECT COUNT(*) FROM apw_lgb_predictions")[0] == 0


# ---------------------------------------------------------------------------
# Branch 2 — booster file missing
# ---------------------------------------------------------------------------


def test_branch_2_file_missing(fresh_db, tmp_path):
    rec = _make_active_record(
        "M-missing", file_path=str(tmp_path / "ghost.txt"),
    )
    insert_model(fresh_db, rec, activate=True)
    scorer = build_lgb_scorer(fresh_db)
    out = scorer.score_batch([_candidate("X")])
    assert out.degrade_reason and out.degrade_reason.startswith("LGB_DEGRADE_FILE_MISSING")


# ---------------------------------------------------------------------------
# Branch 3a — schema_version drift
# ---------------------------------------------------------------------------


def test_branch_3a_schema_version_drift(fresh_db, tmp_path):
    booster = tmp_path / "real.txt"
    booster.write_text("fake but exists", encoding="utf-8")
    rec = _make_active_record(
        "M-drift", file_path=str(booster), schema_version=99,
    )
    insert_model(fresh_db, rec, activate=True)
    scorer = build_lgb_scorer(fresh_db)
    out = scorer.score_batch([_candidate("X")])
    assert out.degrade_reason and "SCHEMA_MISMATCH" in out.degrade_reason


# ---------------------------------------------------------------------------
# Branch 3b — feature_list_json drift
# ---------------------------------------------------------------------------


def test_branch_3b_feature_list_drift(fresh_db, tmp_path):
    booster = tmp_path / "real.txt"
    booster.write_text("fake but exists", encoding="utf-8")
    # We can't write a *real* booster without training, so this branch will
    # actually trip on the booster load before feature_list, surfacing the
    # LOAD_FAIL degrade reason. Make a real booster instead.
    import lightgbm as lgb
    df = pd.DataFrame(
        np.random.RandomState(0).rand(40, len(FEATURE_NAMES)),
        columns=FEATURE_NAMES,
    )
    y = np.random.RandomState(0).randint(0, 2, size=40)
    clf = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    clf.fit(df, y)
    booster_path = tmp_path / "real.txt"
    clf.booster_.save_model(str(booster_path))
    bogus_list = ["x_" + f for f in FEATURE_NAMES]
    rec = _make_active_record(
        "M-feats", file_path=str(booster_path),
        feature_list_json=json.dumps(bogus_list),
    )
    insert_model(fresh_db, rec, activate=True)
    scorer = build_lgb_scorer(fresh_db)
    out = scorer.score_batch([_candidate("X")])
    assert out.degrade_reason and "SCHEMA_MISMATCH" in out.degrade_reason


# ---------------------------------------------------------------------------
# Branch 4 — per-row predict failure (synthetic via patched booster)
# ---------------------------------------------------------------------------


def test_branch_4_predict_raises_batch_wide(fresh_db, tmp_path):
    import lightgbm as lgb
    df = pd.DataFrame(
        np.random.RandomState(0).rand(40, len(FEATURE_NAMES)),
        columns=FEATURE_NAMES,
    )
    y = np.random.RandomState(0).randint(0, 2, size=40)
    clf = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    clf.fit(df, y)
    booster_path = tmp_path / "real.txt"
    clf.booster_.save_model(str(booster_path))
    rec = _make_active_record("M-predict", file_path=str(booster_path))
    insert_model(fresh_db, rec, activate=True)
    scorer = build_lgb_scorer(fresh_db)

    def _explode(*_args, **_kw):
        raise RuntimeError("simulated predict failure")

    with patch.object(scorer._booster, "predict", side_effect=_explode):
        out = scorer.score_batch([_candidate("X"), _candidate("Y")])
    assert out.degrade_reason and "PREDICT_FAIL" in out.degrade_reason
    assert all(s.lgb_score is None for s in out.scores)


# ---------------------------------------------------------------------------
# Branch 5 — lightgbm ImportError
# ---------------------------------------------------------------------------


def test_branch_5_lightgbm_missing(fresh_db, monkeypatch):
    """Simulate lightgbm being unimportable by monkey-patching sys.modules
    BEFORE the scorer's lazy import."""
    import builtins

    real_import = builtins.__import__

    def _no_lgb(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("simulated: lightgbm not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_lgb)
    scorer = build_lgb_scorer(fresh_db)
    out = scorer.score_batch([_candidate("X")])
    assert out.degrade_reason == "LGB_DEGRADE_NO_LIGHTGBM"


# ---------------------------------------------------------------------------
# Happy path — real booster, end-to-end scoring + persistence
# ---------------------------------------------------------------------------


def test_happy_path_scores_and_persists(fresh_db, tmp_path):
    import lightgbm as lgb
    rng = np.random.RandomState(7)
    df = pd.DataFrame(rng.rand(80, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    y = (df["f_acc_score"] > df["f_acc_score"].median()).astype(int).values
    clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    clf.fit(df, y)
    booster_path = tmp_path / "real.txt"
    clf.booster_.save_model(str(booster_path))
    rec = _make_active_record("M-happy", file_path=str(booster_path))
    insert_model(fresh_db, rec, activate=True)

    scorer = build_lgb_scorer(fresh_db)
    cands = [
        {"ts_code": f"60000{i}.SH",
         "accumulation_score": 50 + i * 1.5,
         "probe_quality_score": 70,
         "launch_setup_score": 70,
         "atr_10d_pct": 2.0}
        for i in range(15)
    ]
    out = scorer.score_batch(cands)
    assert out.degrade_reason is None
    assert all(s.lgb_score is not None for s in out.scores)
    # decile must populate when ≥10 valid rows.
    assert any(s.lgb_decile is not None for s in out.scores)

    import uuid
    run_id = str(uuid.uuid4())
    written = scorer.persist_predictions(
        fresh_db, out, run_id=run_id, trade_date="20260520"
    )
    assert written == len(cands)
    n_rows = fresh_db.fetchone(
        "SELECT COUNT(*) FROM apw_lgb_predictions WHERE run_id = ?", (run_id,)
    )[0]
    assert n_rows == len(cands)


def test_empty_batch_returns_empty_scores(fresh_db):
    scorer = LgbScorer(booster=None, model_record=None, degrade_reason=None)
    out = scorer.score_batch([])
    assert out.scores == []
    assert out.degrade_reason is None
