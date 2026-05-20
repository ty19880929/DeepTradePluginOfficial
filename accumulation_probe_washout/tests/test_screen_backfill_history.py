"""v0.3.0 — ``screen --backfill-history`` integration test.

Asserts:
  * iterates open trade dates in [start, end];
  * writes apw_signal_history rows;
  * NEVER touches apw_watchlist (regardless of phase);
  * resumes (skips dates with existing rows) when --overwrite is False;
  * --overwrite DELETEs and re-inserts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.runner import (
    ApwRunner,
    BackfillHistoryParams,
)
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import NullRenderer
from tests.conftest import make_quotes


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


class _FakeConfig:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {"tushare.token": "test-token"}

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


class _FakeTushare:
    """Just enough to satisfy execute_screen / _backfill loop for 2 trade days."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        codes = ["600000.SH"]
        self._codes = codes
        self._daily: dict[str, pd.DataFrame] = {}
        for c in codes:
            self._daily[c] = make_quotes(ts_code=c, pattern="flat", n=130,
                                          probe_index=110, probe_multiplier=5.0)

    def call(
        self,
        api: str,
        *,
        trade_date: str | None = None,
        params: dict[str, Any] | None = None,
        fields: str | None = None,
        force_sync: bool = False,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {
            "trade_date": trade_date,
            "params": params,
            "fields": fields,
            "force_sync": force_sync,
        }
        self.calls.append((api, kwargs))
        params = params or {}
        if api == "trade_cal":
            base = pd.date_range("2024-01-01", "2026-12-31", freq="D")
            return pd.DataFrame(
                {
                    "cal_date": base.strftime("%Y%m%d"),
                    "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                    "pretrade_date": [None] * len(base),
                }
            )
        if api == "stock_basic":
            return pd.DataFrame(
                [{"ts_code": c, "name": f"测试{i}", "market": "主板",
                  "exchange": "SSE", "list_status": "L", "list_date": "20100101",
                  "industry": "电子"} for i, c in enumerate(self._codes)]
            )
        if api == "stock_st":
            return pd.DataFrame()
        if api == "suspend_d":
            return pd.DataFrame()
        if api == "daily":
            codes = params.get("ts_code", "").split(",")
            frames = [self._daily[c] for c in codes if c in self._daily]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        if api == "daily_basic":
            # New per-day shape: runner calls daily_basic(trade_date=YYYYMMDD).
            if trade_date is not None:
                frames = []
                for c, df in self._daily.items():
                    sub = df[df["trade_date"].astype(str) == str(trade_date)][
                        ["ts_code", "trade_date", "turnover_rate", "circ_mv"]
                    ]
                    if not sub.empty:
                        frames.append(sub)
                return (
                    pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                )
            codes = params.get("ts_code", "").split(",")
            frames = []
            for c in codes:
                if c in self._daily:
                    frames.append(self._daily[c][["ts_code", "trade_date",
                                                   "turnover_rate", "circ_mv"]])
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if api == "moneyflow":
            codes = params.get("ts_code", "").split(",")
            rows: list[dict] = []
            for c in codes:
                if c not in self._daily:
                    continue
                for d in self._daily[c]["trade_date"].tail(30):
                    rows.append({"ts_code": c, "trade_date": d, "net_mf_amount": 4000.0})
            return pd.DataFrame(rows)
        return pd.DataFrame()


@pytest.fixture
def runtime(fresh_db) -> ApwRuntime:
    return ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),  # type: ignore[arg-type]
        llms=None,  # type: ignore[arg-type]
        tushare=_FakeTushare(),  # type: ignore[arg-type]
    )


def _existing_signal_dates(db) -> set[str]:
    rows = db.fetchall("SELECT DISTINCT trade_date FROM apw_signal_history")
    return {r[0] for r in (rows or [])}


def test_backfill_iterates_dates_writes_history_only(runtime):
    runner = ApwRunner(runtime, renderer=NullRenderer())
    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start="20240425", end="20240426", overwrite=False)
    )
    assert outcome.status.value == "success", outcome.error

    # Watchlist must remain empty: backfill skips it by contract.
    n_watch = runtime.db.fetchone("SELECT COUNT(*) FROM apw_watchlist")[0]
    assert n_watch == 0


def test_backfill_resume_skips_existing_dates(runtime):
    """A pre-seeded date is left untouched when --overwrite is False."""
    now = datetime.now()
    # Seed an "already processed" row for 20240425.
    runtime.db.execute(
        """
        INSERT INTO apw_signal_history
        (trade_date, ts_code, name, phase, accumulation_score,
         probe_quality_score, washout_score, launch_setup_score,
         raw_candidate_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "20240425", "999999.SH", "preexisting", "accumulating",
            50.0, 50.0, 50.0, 50.0, json.dumps({"ts_code": "999999.SH"}), now,
        ],
    )

    runner = ApwRunner(runtime, renderer=NullRenderer())
    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start="20240425", end="20240426", overwrite=False)
    )
    assert outcome.status.value == "success"
    assert outcome.summary["n_dates_skipped"] >= 1

    # 999999.SH preexisting row still there.
    preexisting = runtime.db.fetchone(
        "SELECT COUNT(*) FROM apw_signal_history "
        "WHERE trade_date = ? AND ts_code = ?",
        ["20240425", "999999.SH"],
    )[0]
    assert preexisting == 1


def test_backfill_overwrite_deletes_existing_then_refills(runtime):
    now = datetime.now()
    runtime.db.execute(
        """
        INSERT INTO apw_signal_history
        (trade_date, ts_code, name, phase, accumulation_score,
         probe_quality_score, washout_score, launch_setup_score,
         raw_candidate_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "20240425", "999999.SH", "preexisting", "accumulating",
            50.0, 50.0, 50.0, 50.0, json.dumps({"ts_code": "999999.SH"}), now,
        ],
    )

    runner = ApwRunner(runtime, renderer=NullRenderer())
    outcome = runner.execute_backfill_history(
        BackfillHistoryParams(start="20240425", end="20240426", overwrite=True)
    )
    assert outcome.status.value == "success"
    # The preexisting row should be gone because --overwrite did DELETE first.
    n_pre = runtime.db.fetchone(
        "SELECT COUNT(*) FROM apw_signal_history "
        "WHERE trade_date = ? AND ts_code = ?",
        ["20240425", "999999.SH"],
    )[0]
    assert n_pre == 0


def test_backfill_rejects_missing_dates(runtime):
    runner = ApwRunner(runtime, renderer=NullRenderer())
    with pytest.raises(ValueError, match="--start"):
        runner.execute_backfill_history(
            BackfillHistoryParams(start="", end="20240426")
        )


def test_backfill_rejects_inverted_range(runtime):
    runner = ApwRunner(runtime, renderer=NullRenderer())
    with pytest.raises(ValueError, match="<="):
        runner.execute_backfill_history(
            BackfillHistoryParams(start="20240601", end="20240501")
        )
