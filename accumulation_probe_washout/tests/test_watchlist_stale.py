"""Round-2 P1 regression: stale watchlist rows must not feed today's analyze.

Seeds two ``apw_watchlist`` rows — one with ``last_seen_date = T`` (the day we
ask analyze to run for), one with ``last_seen_date = T-1`` — and asserts only
the T row reaches the LLM batcher and ``apw_stage_results``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from accumulation_probe_washout.runner import AnalyzeParams, ApwRunner
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


def _insert_row(db, *, ts_code: str, last_seen_date: str) -> None:
    cand = {
        "candidate_id": f"{last_seen_date}_{ts_code}",
        "ts_code": ts_code,
        "name": f"测试{ts_code[:6]}",
        "trade_date": last_seen_date,
        "phase": "launch_ready",
        "accumulation_score": 70.0,
        "probe_quality_score": 80.0,
        "washout_score": 75.0,
        "launch_setup_score": 78.0,
    }
    db.execute(
        """
        INSERT INTO apw_watchlist
        (ts_code, name, first_seen_date, last_seen_date, phase,
         accumulation_score, probe_quality_score, washout_score,
         launch_setup_score, raw_candidate_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ts_code, cand["name"], last_seen_date, last_seen_date, "launch_ready",
            70.0, 80.0, 75.0, 78.0,
            json.dumps(cand, ensure_ascii=False), datetime.now(),
        ],
    )


class _FakeConfig:
    def get(self, k, default=None):
        return "test-token" if k == "tushare.token" else default
    def get_app_config(self):
        raise NotImplementedError


class _FakeLLMClient:
    """Echoes back exactly the candidate_ids the prompt contained."""

    def __init__(self):
        self.last_ids: list[str] = []

    def complete_json(self, *, system, user, schema, profile, envelope_defaults=None):
        import re
        ids = re.findall(r'"candidate_id":\s*"([^"]+)"', user)
        self.last_ids = ids
        cands = []
        for i, cid in enumerate(ids, start=1):
            ts_code = cid.split("_", 1)[-1]
            cands.append({
                "candidate_id": cid,
                "ts_code": ts_code,
                "name": f"name_{ts_code}",
                "rank": i,
                "launch_score": 80 - i * 5,
                "confidence": "medium",
                "prediction": "launch_ready",
                "main_pattern": "probe_washout_breakout",
                "phase": "launch_ready",
                "dimension_scores": {
                    "accumulation": 70, "probe": 80, "washout": 75,
                    "launch_timing": 78, "capital_confirmation": 60, "risk": 30,
                },
                "rationale": "结构完整",
                "key_evidence": [{
                    "field": "probe_quality_score", "value": 80, "unit": "score",
                    "interpretation": "ok",
                }],
                "next_session_watch": ["watch"],
                "invalidation_triggers": ["trigger"],
            })
        envelope = envelope_defaults or {}
        payload = {
            "stage": envelope.get("stage", "accumulation_probe_washout_analysis"),
            "trade_date": envelope.get("trade_date", "20260515"),
            "next_trade_date": envelope.get("next_trade_date", "20260516"),
            "batch_no": envelope.get("batch_no", 1),
            "batch_total": envelope.get("batch_total", 1),
            "market_context_summary": "测试",
            "risk_disclaimer": "辅助判断",
            "candidates": cands,
        }
        return schema.model_validate(payload), {
            "input_tokens": 10, "output_tokens": 10, "latency_ms": 1,
        }


class _FakeLLMManager:
    def __init__(self):
        self.client = _FakeLLMClient()
    def get_client(self, name, *, plugin_id, run_id):
        return self.client


class _FakeTushare:
    def call(self, api, **kwargs):
        if api == "trade_cal":
            base = pd.date_range("2025-01-01", "2026-12-31", freq="D")
            return pd.DataFrame({
                "cal_date": base.strftime("%Y%m%d"),
                "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                "pretrade_date": [None] * len(base),
            })
        return pd.DataFrame()


def test_stale_watchlist_row_excluded_from_analyze(fresh_db):
    # T = 20260515 (a Friday — is_open=1)
    _insert_row(fresh_db, ts_code="600000.SH", last_seen_date="20260515")  # fresh
    _insert_row(fresh_db, ts_code="600999.SH", last_seen_date="20260514")  # stale

    llms = _FakeLLMManager()
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=llms,  # type: ignore[arg-type]
        tushare=_FakeTushare(),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_analyze(AnalyzeParams(trade_date="20260515"))

    # Only the T-day row reaches the LLM.
    assert llms.client.last_ids == ["20260515_600000.SH"]

    # And only the T-day row ends up in apw_stage_results.
    rows = fresh_db.fetchall(
        "SELECT ts_code, candidate_id, trade_date FROM apw_stage_results "
        "WHERE run_id = ?",
        [outcome.run_id],
    )
    assert len(rows) == 1
    assert rows[0][0] == "600000.SH"
    assert rows[0][2] == "20260515"

    # The stale row still exists in watchlist (we don't delete it; we just
    # refuse to analyze it today).
    wl = fresh_db.fetchall(
        "SELECT ts_code, last_seen_date FROM apw_watchlist ORDER BY ts_code"
    )
    assert wl == [("600000.SH", "20260515"), ("600999.SH", "20260514")]


def test_analyze_logs_warning_when_only_stale_rows(fresh_db):
    # Only a T-1 stale row — no T-day candidates.
    _insert_row(fresh_db, ts_code="600999.SH", last_seen_date="20260514")

    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMManager(),  # type: ignore[arg-type]
        tushare=_FakeTushare(),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_analyze(AnalyzeParams(trade_date="20260515"))

    # No predictions, no stage rows.
    assert outcome.summary["n_candidates"] == 0
    assert outcome.summary["n_predictions"] == 0
    rows = fresh_db.fetchall("SELECT COUNT(*) FROM apw_stage_results")
    assert int(rows[0][0]) == 0

    # The warning event was emitted into apw_events.
    ev = fresh_db.fetchall(
        "SELECT message, level FROM apw_events WHERE run_id = ? AND level = 'warn'",
        [outcome.run_id],
    )
    assert any("非当日行" in r[0] for r in ev), ev
