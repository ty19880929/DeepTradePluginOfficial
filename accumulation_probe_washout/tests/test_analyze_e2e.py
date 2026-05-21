"""End-to-end analyze test with mock LLM — T3.9.

Seeds apw_watchlist with 3 rows, hands a FakeLLM to the runner via a
FakeLLMManager.get_client, asserts apw_stage_results gets 3 rows with dense
rank 1..N.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.runner import AnalyzeParams, ApwRunner, RunParams
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


def _seed_watchlist(db, codes):
    now = datetime.now()
    for i, code in enumerate(codes, start=1):
        cand = {
            "candidate_id": f"20260515_{code}",
            "ts_code": code,
            "name": f"测试股{code[:6]}",
            "trade_date": "20260515",
            "phase": "launch_ready",
            "close": 10.0 + i,
            "accumulation_score": 70.0,
            "probe_quality_score": 80.0,
            "washout_score": 75.0,
            "launch_setup_score": 78.0,
            "above_ma5": True,
            "above_ma10": True,
            "current_volume_ratio_5d": 1.5,
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
                code, cand["name"], "20260510", "20260515", "launch_ready",
                70.0, 80.0, 75.0, 78.0,
                json.dumps(cand, ensure_ascii=False), now,
            ],
        )


class _FakeConfig:
    def get(self, k, default=None):
        return "test-token" if k == "tushare.token" else default
    def get_app_config(self):
        raise NotImplementedError


class _FakeLLMClient:
    """Returns a valid APWTrendResponse for any input batch."""

    def complete_json(self, *, system, user, schema, profile, envelope_defaults=None):
        import re
        # Extract candidate_ids from the user prompt (they're inlined as JSON).
        ids = re.findall(r'"candidate_id":\s*"([^"]+)"', user)
        candidates = []
        for i, cid in enumerate(ids, start=1):
            ts_code = cid.split("_", 1)[-1]
            candidates.append({
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
                    "interpretation": "试盘质量好",
                }],
                "next_session_watch": ["突破试盘高点"],
                "invalidation_triggers": ["跌破试盘日 low"],
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
            "candidates": candidates,
        }
        return schema.model_validate(payload), {
            "input_tokens": 100, "output_tokens": 200, "latency_ms": 50,
        }


class _FakeLLMManager:
    def get_client(self, name, *, plugin_id, run_id):
        return _FakeLLMClient()


class _FakeTushare:
    """Minimal trade_cal — analyze only needs the calendar."""

    def call(self, api, **kwargs):
        if api == "trade_cal":
            base = pd.date_range("2025-01-01", "2026-12-31", freq="D")
            return pd.DataFrame({
                "cal_date": base.strftime("%Y%m%d"),
                "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                "pretrade_date": [None] * len(base),
            })
        return pd.DataFrame()


def test_analyze_persists_stage_results(fresh_db):
    codes = ["600000.SH", "600001.SH", "600002.SH"]
    _seed_watchlist(fresh_db, codes)

    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMManager(),  # type: ignore[arg-type]
        tushare=_FakeTushare(),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_analyze(AnalyzeParams(trade_date="20260515"))

    assert outcome.status.value in {"success", "partial_failed"}, outcome.error
    rows = fresh_db.fetchall(
        "SELECT ts_code, rank, prediction FROM apw_stage_results "
        "WHERE run_id = ? ORDER BY rank",
        [outcome.run_id],
    )
    assert len(rows) == 3
    # rank is dense 1..3
    assert [r[1] for r in rows] == [1, 2, 3]
    # All landed as launch_ready
    assert all(r[2] == "launch_ready" for r in rows)

    # watchlist rows now carry latest_prediction
    wl = fresh_db.fetchall(
        "SELECT ts_code, latest_prediction FROM apw_watchlist ORDER BY ts_code"
    )
    assert {r[1] for r in wl} == {"launch_ready"}

    step5_events = fresh_db.fetchall(
        """
        SELECT event_type, message, payload_json
        FROM apw_events
        WHERE run_id = ?
          AND event_type IN ('step.started', 'step.finished')
          AND payload_json LIKE '%"step": 5%'
        ORDER BY seq
        """,
        [outcome.run_id],
    )
    assert [r[0] for r in step5_events] == ["step.started", "step.finished"]
    assert "写入 3 条" in step5_events[-1][1]
    payload = json.loads(step5_events[-1][2])
    assert payload["result_summary"][0] == {
        "rank": 1,
        "ts_code": "600000.SH",
        "name": "name_600000.SH",
        "current_price": 11.0,
        "launch_score": 75.0,
        "prediction": "launch_ready",
        "confidence": "medium",
        "llm_opinion": "结构完整",
    }


def test_run_substage_analyze_emits_step5_events(fresh_db):
    codes = ["600000.SH", "600001.SH"]
    _seed_watchlist(fresh_db, codes)

    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMManager(),  # type: ignore[arg-type]
        tushare=_FakeTushare(),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    run_id = runner._start_run("run", "20260515", RunParams(trade_date="20260515"))
    outcome = runner.execute_analyze(
        AnalyzeParams(trade_date="20260515"),
        _owns_run=False,
    )

    assert outcome.run_id == run_id
    assert outcome.status.value in {"success", "partial_failed"}, outcome.error

    step5_events = fresh_db.fetchall(
        """
        SELECT event_type, message, payload_json
        FROM apw_events
        WHERE run_id = ?
          AND event_type IN ('step.started', 'step.finished')
          AND payload_json LIKE '%"step": 5%'
        ORDER BY seq
        """,
        [run_id],
    )
    assert [r[0] for r in step5_events] == ["step.started", "step.finished"]
    assert "写入 2 条" in step5_events[-1][1]
    payload = json.loads(step5_events[-1][2])
    assert [row["current_price"] for row in payload["result_summary"]] == [11.0, 12.0]
