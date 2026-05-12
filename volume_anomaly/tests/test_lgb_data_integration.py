"""PR-2.2 — ``_attach_lgb_scores`` + ``collect_analyze_bundle`` integration.

Three branches covered (design §7.3):
    1. scorer = None (LGB disabled)          → fields all None, no DB rows
    2. scorer present but unloaded            → same + load_error in
       data_unavailable
    3. scorer loaded                          → real scores, bundle.lgb_model_id
       set, deciles assigned when ≥10 candidates

Plus persistence: ``_persist_lgb_predictions`` writes exactly the candidates
with non-None ``lgb_score`` into ``va_lgb_predictions``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from deeptrade.core.db import Database

from volume_anomaly.data import AnalyzeBundle, _attach_lgb_scores
from volume_anomaly.lgb import paths as lgb_paths
from volume_anomaly.lgb import registry as lgb_registry
from volume_anomaly.lgb.features import FEATURE_NAMES


def _candidate(ts_code: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "candidate_id": ts_code,
        "ts_code": ts_code,
        "name": "TestCo",
        "industry": "测试",
        "tracked_since": "20260101",
        "tracked_days": 3,
        "last_close": 12.0,
        "ma5": 11.8,
        "ma10": 11.5,
        "ma20": 11.0,
        "ma60": 10.5,
        "above_ma20": True,
        "above_ma60": True,
        "anomaly_pct_chg": 5.0,
        "anomaly_body_ratio": 0.75,
        "anomaly_turnover_rate": 8.0,
        "anomaly_vol_ratio_5d": 2.5,
        "moneyflow_5d_summary": {"rows_used": 0},
    }
    base.update(overrides)
    return base


def _empty_bundle(candidates: list[dict[str, Any]]) -> AnalyzeBundle:
    return AnalyzeBundle(
        trade_date="20260601",
        next_trade_date="20260602",
        candidates=candidates,
        market_summary={},
        sector_strength_data={"top_sectors": []},
        sector_strength_source="industry_fallback",
    )


# ---------------------------------------------------------------------------
# Branch 1: scorer is None (LGB disabled)
# ---------------------------------------------------------------------------


def test_attach_lgb_scores_disabled_no_scorer() -> None:
    bundle = _empty_bundle([_candidate("000001.SZ")])
    _attach_lgb_scores(bundle, scorer=None)
    c = bundle.candidates[0]
    assert c["lgb_score"] is None
    assert c["lgb_decile"] is None
    assert c["lgb_feature_missing"] == []
    assert any("lgb_model (disabled)" in s for s in bundle.data_unavailable)
    assert bundle.lgb_model_id is None


# ---------------------------------------------------------------------------
# Branch 2: scorer present but unloaded (no active model on disk)
# ---------------------------------------------------------------------------


def test_attach_lgb_scores_scorer_not_loaded() -> None:
    scorer = MagicMock()
    scorer.loaded = False
    scorer.load_error = "no_active_model"
    bundle = _empty_bundle([_candidate("000001.SZ")])
    _attach_lgb_scores(bundle, scorer=scorer)
    c = bundle.candidates[0]
    assert c["lgb_score"] is None
    assert bundle.lgb_model_id is None
    assert any("no_active_model" in s for s in bundle.data_unavailable)


def test_attach_lgb_scores_scorer_load_error_propagated() -> None:
    scorer = MagicMock()
    scorer.loaded = False
    scorer.load_error = "schema_mismatch: model_features(50) != FEATURE_NAMES(52)"
    bundle = _empty_bundle([_candidate("000001.SZ")])
    _attach_lgb_scores(bundle, scorer=scorer)
    assert any("schema_mismatch" in s for s in bundle.data_unavailable)


# ---------------------------------------------------------------------------
# Branch 3: scorer loaded → real scoring
# ---------------------------------------------------------------------------


def _stub_loaded_scorer(scores: list[float], model_id: str = "test-model"):
    """Build a scorer whose loaded=True and score_batch returns `scores`."""
    scorer = MagicMock()
    scorer.loaded = True
    scorer.model_id = model_id
    scorer.load_error = None

    def fake_score(feature_df: pd.DataFrame) -> pd.DataFrame:
        assert list(feature_df.columns) == FEATURE_NAMES
        return pd.DataFrame(
            {
                "lgb_score": scores[: len(feature_df)],
                "feature_hash": ["abc"] * len(feature_df),
                "feature_missing_json": ["[]"] * len(feature_df),
            },
            index=feature_df.index,
        )

    scorer.score_batch = MagicMock(side_effect=fake_score)
    return scorer


def test_attach_lgb_scores_loaded_writes_score_and_model_id() -> None:
    candidates = [_candidate(f"{i:06d}.SZ") for i in range(3)]
    bundle = _empty_bundle(candidates)
    scorer = _stub_loaded_scorer([0.5, 0.7, 0.2], model_id="20260601_1_abc")
    _attach_lgb_scores(bundle, scorer=scorer)
    assert bundle.lgb_model_id == "20260601_1_abc"
    # 0.5 → 50.0, 0.7 → 70.0, 0.2 → 20.0 (×100 per design §7).
    expected = [50.0, 70.0, 20.0]
    for c, want in zip(bundle.candidates, expected, strict=False):
        assert c["lgb_score"] == pytest.approx(want)
    # < 10 candidates → all deciles are None (design §3.1).
    assert all(c["lgb_decile"] is None for c in bundle.candidates)


def test_attach_lgb_scores_loaded_assigns_deciles_when_batch_geq_10() -> None:
    candidates = [_candidate(f"{i:06d}.SZ") for i in range(12)]
    bundle = _empty_bundle(candidates)
    scores = list(np.linspace(0.05, 0.95, 12))
    scorer = _stub_loaded_scorer(scores)
    _attach_lgb_scores(bundle, scorer=scorer)
    deciles = [c["lgb_decile"] for c in bundle.candidates]
    # Every candidate now has an integer decile in [1, 10].
    assert all(d is not None for d in deciles)
    assert min(deciles) == 1
    assert max(deciles) == 10


def test_attach_lgb_scores_predict_failure_does_not_crash() -> None:
    candidates = [_candidate("000001.SZ")]
    bundle = _empty_bundle(candidates)
    scorer = MagicMock()
    scorer.loaded = True
    scorer.model_id = "m"
    scorer.load_error = None
    scorer.score_batch = MagicMock(side_effect=RuntimeError("synthetic"))
    _attach_lgb_scores(bundle, scorer=scorer)
    # Candidate fields remain default; bundle gets an explicit
    # "lgb_predict_failed" entry.
    assert bundle.candidates[0]["lgb_score"] is None
    assert any("lgb_predict_failed" in s for s in bundle.data_unavailable)


# ---------------------------------------------------------------------------
# _persist_lgb_predictions — the audit-table write
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_predictions_table(tmp_path: Path) -> Database:
    db = Database(tmp_path / "lgb_persist.duckdb")
    db.execute(
        """
        CREATE TABLE va_lgb_predictions (
            run_id                UUID NOT NULL,
            trade_date            VARCHAR NOT NULL,
            ts_code               VARCHAR NOT NULL,
            model_id              VARCHAR NOT NULL,
            lgb_score             DOUBLE NOT NULL,
            lgb_decile            INTEGER,
            feature_hash          VARCHAR NOT NULL,
            feature_missing_json  VARCHAR,
            created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_id, ts_code)
        )
        """
    )
    yield db
    db.close()


def test_persist_lgb_predictions_writes_scored_rows(
    db_with_predictions_table: Database,
) -> None:
    from volume_anomaly.runner import _persist_lgb_predictions
    from volume_anomaly.runtime import VaRuntime

    rt = MagicMock(spec=VaRuntime)
    rt.db = db_with_predictions_table
    rt.run_id = "00000000-0000-0000-0000-000000000001"

    candidates = [
        {"ts_code": "000001.SZ", "lgb_score": 70.0, "lgb_decile": 8,
         "lgb_feature_missing": [], "anomaly_pct_chg": 5.0},
        {"ts_code": "000002.SZ", "lgb_score": None, "lgb_decile": None,
         "lgb_feature_missing": [], "anomaly_pct_chg": 4.0},
    ]
    bundle = _empty_bundle(candidates)
    bundle.lgb_model_id = "model_v1"
    _persist_lgb_predictions(rt, bundle)

    rows = db_with_predictions_table.fetchall(
        "SELECT ts_code, lgb_score, lgb_decile, model_id FROM va_lgb_predictions"
    )
    # Only the candidate with a non-None lgb_score lands in the audit table.
    assert len(rows) == 1
    assert rows[0][0] == "000001.SZ"
    assert rows[0][1] == 70.0
    assert rows[0][2] == 8
    assert rows[0][3] == "model_v1"


def test_persist_lgb_predictions_no_op_when_model_id_none(
    db_with_predictions_table: Database,
) -> None:
    from volume_anomaly.runner import _persist_lgb_predictions
    from volume_anomaly.runtime import VaRuntime

    rt = MagicMock(spec=VaRuntime)
    rt.db = db_with_predictions_table
    rt.run_id = "00000000-0000-0000-0000-000000000002"

    bundle = _empty_bundle([{"ts_code": "000001.SZ", "lgb_score": 50.0}])
    bundle.lgb_model_id = None  # LGB disabled
    _persist_lgb_predictions(rt, bundle)

    rows = db_with_predictions_table.fetchall("SELECT * FROM va_lgb_predictions")
    assert rows == []


def test_persist_lgb_predictions_swallows_missing_table(
    tmp_path: Path,
) -> None:
    """Pre-migration DB (no va_lgb_predictions) must not crash the runner."""
    from volume_anomaly.runner import _persist_lgb_predictions
    from volume_anomaly.runtime import VaRuntime

    db = Database(tmp_path / "no_table.duckdb")
    try:
        rt = MagicMock(spec=VaRuntime)
        rt.db = db
        rt.run_id = "00000000-0000-0000-0000-000000000003"
        bundle = _empty_bundle([{"ts_code": "000001.SZ", "lgb_score": 50.0}])
        bundle.lgb_model_id = "m"
        # Should not raise.
        _persist_lgb_predictions(rt, bundle)
    finally:
        db.close()
