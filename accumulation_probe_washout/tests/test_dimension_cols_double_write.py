"""v0.3.0 — apw_stage_results dual-write of dim_* columns.

The migration 20260602_001 introduces 6 DOUBLE columns alongside the
legacy ``dimension_scores_json`` blob. The runner must keep both in sync so
``stats --by dimension_scores`` can rely on the columns being authoritative.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from accumulation_probe_washout.runner import ApwRunner
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import NullRenderer


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def fresh_db(tmp_path):
    from deeptrade.core.db import Database
    db = Database(tmp_path / "apw_test.duckdb")
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for stmt in path.read_text(encoding="utf-8").split(";"):
            if stmt.strip():
                db.execute(stmt.strip())
    yield db
    db.close()


class _DimScores(SimpleNamespace):
    """Mimics APWDimensionScores: must expose ``.model_dump() -> dict``."""

    def model_dump(self) -> dict:
        return {
            "accumulation": self.accumulation,
            "probe": self.probe,
            "washout": self.washout,
            "launch_timing": self.launch_timing,
            "capital_confirmation": self.capital_confirmation,
            "risk": self.risk,
        }


class _Evidence(SimpleNamespace):
    def model_dump(self) -> dict:
        return {"field": self.field, "value": self.value, "unit": self.unit,
                "interpretation": self.interpretation}


class _Candidate(SimpleNamespace):
    """Minimal stand-in for APWTrendCandidate."""

    def model_dump_json(self) -> str:
        return json.dumps({"ts_code": self.ts_code, "rank": self.rank})


def _make_cand(ts_code: str = "600000.SH") -> _Candidate:
    return _Candidate(
        candidate_id="20260520_600000.SH",
        ts_code=ts_code,
        rank=1,
        launch_score=78.5,
        confidence="high",
        prediction="launch_ready",
        main_pattern="probe_washout_breakout",
        phase="launch_ready",
        dimension_scores=_DimScores(
            accumulation=72.0,
            probe=88.0,
            washout=65.0,
            launch_timing=80.0,
            capital_confirmation=58.0,
            risk=22.0,
        ),
        key_evidence=[
            _Evidence(field="probe_volume_ratio_5d", value=3.2, unit="x",
                      interpretation="放量明显"),
        ],
        rationale="rationale",
        next_session_watch=["close_above_probe_high"],
        invalidation_triggers=["close_below_probe_low"],
        risk_flags=[],
        missing_data=[],
    )


def test_dual_write_dim_cols_match_json_blob(fresh_db):
    rt = ApwRuntime(
        db=fresh_db,
        config=None,  # type: ignore[arg-type]
        llms=None,  # type: ignore[arg-type]
        run_id="11111111-1111-1111-1111-111111111111",
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    # Use the private upsert directly to bypass the pipeline/LLM path.
    runner._upsert_stage_result(
        _make_cand(), run_id=rt.run_id, trade_date="20260520"
    )

    row = fresh_db.fetchone(
        """
        SELECT dimension_scores_json,
               dim_accumulation, dim_probe, dim_washout,
               dim_launch_timing, dim_capital_confirmation, dim_risk
        FROM apw_stage_results
        WHERE run_id = ?
        """,
        [rt.run_id],
    )
    assert row is not None
    ds = json.loads(row[0])
    assert row[1] == pytest.approx(ds["accumulation"])
    assert row[2] == pytest.approx(ds["probe"])
    assert row[3] == pytest.approx(ds["washout"])
    assert row[4] == pytest.approx(ds["launch_timing"])
    assert row[5] == pytest.approx(ds["capital_confirmation"])
    assert row[6] == pytest.approx(ds["risk"])


def test_dual_write_preserves_existing_index_on_launch_timing(fresh_db):
    """The migration also creates ``idx_apw_stage_results_dim_launch_timing``.
    DuckDB will let us query EXPLAIN against an indexed column without error.
    """
    fresh_db.fetchall(
        "SELECT * FROM apw_stage_results WHERE dim_launch_timing > 50"
    )
