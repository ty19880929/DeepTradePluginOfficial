"""End-to-end test for execute_run — Fix 5 / P2-3.

Regression guard for detailed-design T4.7: ``run = screen → analyze`` must
share **one** run_id, write **one** ``apw_runs`` row with ``mode='run'``, and
emit a **single monotonic** event stream into ``apw_events``.

Before the fix, ``execute_run`` opened two child runs (one ``mode='screen'``,
one ``mode='analyze'``) and reset the renderer/seq state between them — so
no ``mode='run'`` audit row was ever persisted.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.runner import ApwRunner, RunParams
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import NullRenderer
from tests.conftest import make_quotes


MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@pytest.fixture
def fresh_db(tmp_path):
    from deeptrade.core.db import Database

    db = Database(tmp_path / "apw_run_test.duckdb")
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        for stmt in path.read_text(encoding="utf-8").split(";"):
            if stmt.strip():
                db.execute(stmt.strip())
    yield db
    db.close()


class _FakeConfig:
    def get(self, k: str, default: Any = None) -> Any:
        return "test-token" if k == "tushare.token" else default

    def get_app_config(self):  # pragma: no cover - unused
        raise NotImplementedError


class _FakeLLMClient:
    def complete_json(self, *, system, user, schema, profile, envelope_defaults=None):
        import re

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
    """Carbon copy of the screen-e2e fake, kept local so the two tests evolve
    independently."""

    def __init__(self) -> None:
        codes = ["600000.SH", "600001.SH", "600002.SH", "600003.SH", "600004.SH"]
        self._codes = codes
        self._daily: dict[str, pd.DataFrame] = {}
        for i, c in enumerate(codes):
            if i == 0:
                df = make_quotes(ts_code=c, pattern="flat", n=130,
                                 probe_index=110, probe_multiplier=5.0)
                probe_close = df.at[110, "close"]
                for j in range(111, 130):
                    df.at[j, "close"] = probe_close * 0.985
                    df.at[j, "high"] = df.at[j, "close"] * 1.005
                    df.at[j, "low"] = df.at[j, "close"] * 0.995
                    df.at[j, "vol"] = df.at[110, "vol"] * 0.4
                df.at[129, "close"] = probe_close * 1.02
                df.at[129, "high"] = df.at[129, "close"] * 1.02
                df.at[129, "low"] = df.at[128, "close"] * 1.00
                df.at[129, "vol"] = df.at[110, "vol"] * 0.8
            else:
                df = make_quotes(ts_code=c, pattern=("flat" if i % 2 else "uptrend"), n=130)
            self._daily[c] = df

    def call(self, api: str, **kwargs: Any):
        if api == "trade_cal":
            base = pd.date_range("2024-01-01", "2026-12-31", freq="D")
            return pd.DataFrame({
                "cal_date": base.strftime("%Y%m%d"),
                "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                "pretrade_date": [None] * len(base),
            })
        if api == "stock_basic":
            return pd.DataFrame([
                {"ts_code": c, "name": f"试盘股{i}", "market": "主板",
                 "exchange": "SSE", "list_status": "L", "list_date": "20100101",
                 "industry": "电子"}
                for i, c in enumerate(self._codes)
            ])
        if api == "stock_st":
            return pd.DataFrame([{"ts_code": "600001.SH"}])
        if api == "suspend_d":
            return pd.DataFrame()
        if api == "daily":
            codes = kwargs.get("ts_code", "").split(",")
            frames = [self._daily[c] for c in codes if c in self._daily]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if api == "daily_basic":
            codes = kwargs.get("ts_code", "").split(",")
            frames = [self._daily[c][["ts_code", "trade_date", "turnover_rate", "circ_mv"]]
                      for c in codes if c in self._daily]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if api == "moneyflow":
            codes = kwargs.get("ts_code", "").split(",")
            rows: list[dict] = []
            for c in codes:
                if c not in self._daily:
                    continue
                for d in self._daily[c]["trade_date"].tail(30):
                    rows.append({"ts_code": c, "trade_date": d, "net_mf_amount": 4000.0})
            return pd.DataFrame(rows)
        return pd.DataFrame()


def test_execute_run_uses_single_run_id(fresh_db):
    """run = screen + analyze must persist exactly one apw_runs row with
    mode='run', and every apw_events row must reference that same run_id."""
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMManager(),  # type: ignore[arg-type]
        tushare=_FakeTushare(),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_run(RunParams(trade_date="20240509"))

    assert outcome.status.value in {"success", "partial_failed"}, outcome.error

    runs = fresh_db.fetchall(
        "SELECT CAST(run_id AS VARCHAR), mode FROM apw_runs ORDER BY started_at"
    )
    # Before fix: 2 rows (mode='screen' and mode='analyze'); after fix: 1 row.
    assert len(runs) == 1, f"expected single audit row, got {len(runs)}: {runs}"
    assert runs[0][0] == outcome.run_id
    assert runs[0][1] == "run", f"audit row mode must be 'run', got {runs[0][1]!r}"

    # Every event must reference the parent run_id.
    event_rids = fresh_db.fetchall(
        "SELECT DISTINCT CAST(run_id AS VARCHAR) FROM apw_events"
    )
    assert {r[0] for r in event_rids} == {outcome.run_id}, (
        f"events leaked across run_ids: {event_rids}"
    )

    # Summary must carry both sub-stage payloads so downstream tooling can
    # still introspect screen + analyze counts.
    assert set(outcome.summary.keys()) >= {"screen", "analyze"}

    # Sequence numbers must be strictly monotonic within the single run.
    seqs = fresh_db.fetchall(
        "SELECT seq FROM apw_events WHERE run_id = ? ORDER BY seq",
        [outcome.run_id],
    )
    seq_list = [r[0] for r in seqs]
    assert seq_list == sorted(seq_list), "seq not monotonic"
    assert len(set(seq_list)) == len(seq_list), "seq duplicates within one run"


def test_execute_run_propagates_screen_failure(fresh_db, monkeypatch):
    """If screen fails the parent run row is finalised with the screen status
    and analyze is skipped."""
    from accumulation_probe_washout import runner as runner_mod

    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMManager(),  # type: ignore[arg-type]
        tushare=_FakeTushare(),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())

    # Force screen to fail by sabotaging stock_basic to an empty frame, then
    # patching filter_main_board so it raises.
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated screen failure")

    monkeypatch.setattr(runner_mod, "filter_main_board", _boom)
    outcome = runner.execute_run(RunParams(trade_date="20240509"))

    assert outcome.status.value == "failed"
    assert outcome.error is not None

    runs = fresh_db.fetchall(
        "SELECT mode, status, error FROM apw_runs ORDER BY started_at"
    )
    assert len(runs) == 1
    assert runs[0][0] == "run"
    assert runs[0][1] == "failed"
    # No stage_results — analyze never started.
    n_predictions = fresh_db.fetchone(
        "SELECT COUNT(*) FROM apw_stage_results"
    )[0]
    assert n_predictions == 0
