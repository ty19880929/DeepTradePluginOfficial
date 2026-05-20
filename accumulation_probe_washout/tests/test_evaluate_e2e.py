"""End-to-end evaluate test with mock Tushare — T5.6."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.runner import ApwRunner, EvaluateParams
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


def _seed_signal_history(db, *, codes_phases: list[tuple[str, str]], signal_date: str):
    now = datetime.now()
    for code, phase in codes_phases:
        cand = {
            "candidate_id": f"{signal_date}_{code}",
            "ts_code": code, "name": "test", "trade_date": signal_date,
            "phase": phase, "launch_setup_score": 70.0,
        }
        db.execute(
            """
            INSERT INTO apw_signal_history
            (trade_date, ts_code, name, phase, accumulation_score,
             probe_quality_score, washout_score, launch_setup_score,
             raw_candidate_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                signal_date, code, "test", phase, 70.0, 80.0, 75.0, 70.0,
                json.dumps(cand, ensure_ascii=False), now,
            ],
        )


class _FakeConfig:
    def get(self, k, default=None):
        return "test-token" if k == "tushare.token" else default
    def get_app_config(self):
        raise NotImplementedError


class _FakeLLMs:
    pass


class _FakeTushare:
    """Returns synthetic daily quotes — strong uptrend after T."""

    def __init__(self, codes):
        self._codes = codes

    def call(
        self,
        api,
        *,
        trade_date=None,
        params=None,
        fields=None,
        force_sync=False,
    ):
        params = params or {}
        if api == "daily":
            ts_codes = params["ts_code"].split(",")
            start = params["start_date"]
            end = params["end_date"]
            dates = pd.date_range(start, end, freq="D")
            rows = []
            for code in ts_codes:
                if code not in self._codes:
                    continue
                base = 10.0
                for i, d in enumerate(dates):
                    if d.weekday() >= 5:
                        continue
                    close = base * (1 + 0.02 * i)
                    rows.append({
                        "ts_code": code,
                        "trade_date": d.strftime("%Y%m%d"),
                        "close": round(close, 3),
                        "high": round(close * 1.02, 3),
                        "low": round(close * 0.99, 3),
                    })
            return pd.DataFrame(rows)
        return pd.DataFrame()


def test_evaluate_writes_realized_returns_d3_filter(fresh_db):
    codes_phases = [
        ("600000.SH", "launch_ready"),
        ("600001.SH", "washing_after_probe"),
        ("600002.SH", "accumulating"),  # early phase — excluded by default
        ("600003.SH", "probe_seen"),     # early phase — excluded by default
    ]
    _seed_signal_history(fresh_db, codes_phases=codes_phases, signal_date="20260101")

    tushare = _FakeTushare(codes={c for c, _ in codes_phases})
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=tushare,
    )
    runner = ApwRunner(rt, renderer=NullRenderer())

    # ---- default: D3 phase filter excludes early phases
    outcome = runner.execute_evaluate(
        EvaluateParams(from_date="20260101", to_date="20260101",
                       horizons="1,3,5,10", include_early_phases=False)
    )
    assert outcome.status.value == "success"
    rows = fresh_db.fetchall(
        "SELECT ts_code, phase, data_status FROM apw_realized_returns "
        "ORDER BY ts_code"
    )
    codes_evaluated = {r[0] for r in rows}
    assert codes_evaluated == {"600000.SH", "600001.SH"}
    # All have complete data (15+ days of synthetic quotes)
    assert all(r[2] == "complete" for r in rows)


def test_evaluate_include_early_phases_widens(fresh_db):
    codes_phases = [
        ("600000.SH", "launch_ready"),
        ("600002.SH", "accumulating"),
    ]
    _seed_signal_history(fresh_db, codes_phases=codes_phases, signal_date="20260101")

    tushare = _FakeTushare(codes={c for c, _ in codes_phases})
    rt = ApwRuntime(
        db=fresh_db, config=_FakeConfig(), llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=tushare,
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_evaluate(
        EvaluateParams(from_date="20260101", to_date="20260101",
                       horizons="5,10", include_early_phases=True)
    )
    assert outcome.status.value == "success"
    codes_evaluated = {r[0] for r in fresh_db.fetchall(
        "SELECT ts_code FROM apw_realized_returns"
    )}
    assert codes_evaluated == {"600000.SH", "600002.SH"}
