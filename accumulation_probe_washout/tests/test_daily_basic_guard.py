"""Coverage guards for the ``daily_basic`` snapshot.

When ``daily_basic`` returns 0 rows (or covers <50% of the liquid universe)
the runner can no longer apply the market-cap band reliably. Before this
guard the screen silently produced ``n_after_mv=0`` and finished as success,
which made data outages indistinguishable from "no candidates today" —
exactly the failure mode that motivated 《APW run 空结果问题修复方案》.

Now the runner fails fast with an actionable error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.runner import ApwRunner, ScreenParams
from accumulation_probe_washout.runtime import ApwRuntime
from accumulation_probe_washout.ui.protocol import NullRenderer
from tests.conftest import make_quotes


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


class _FakeConfig:
    def get(self, k, default=None):
        return "test-token" if k == "tushare.token" else default

    def get_app_config(self):
        raise NotImplementedError


class _FakeLLMs:
    pass


def _build_tushare(*, daily_basic_returns_empty: bool):
    """Build a fake Tushare client whose ``daily_basic`` is either healthy
    (per-day snapshot) or returns 0 rows on every call — mirroring the real
    Tushare misbehaviour described in the fix plan."""
    codes = ["600000.SH", "600002.SH", "600003.SH"]
    daily = {c: make_quotes(ts_code=c, pattern="flat", n=130) for c in codes}

    class _Fake:
        def __init__(self) -> None:
            self.daily_basic_calls: list[str] = []

        def call(
            self,
            api: str,
            *,
            trade_date: str | None = None,
            params: dict[str, Any] | None = None,
            fields: str | None = None,
            force_sync: bool = False,
        ):
            params = params or {}
            if api == "trade_cal":
                base = pd.date_range("2024-01-01", "2026-12-31", freq="D")
                return pd.DataFrame({
                    "cal_date": base.strftime("%Y%m%d"),
                    "is_open": [0 if d.weekday() >= 5 else 1 for d in base],
                    "pretrade_date": [None] * len(base),
                })
            if api == "stock_basic":
                return pd.DataFrame([
                    {"ts_code": c, "name": f"测试{c[:6]}", "market": "主板",
                     "exchange": "SSE", "list_status": "L",
                     "list_date": "20100101", "industry": "电子"}
                    for c in codes
                ])
            if api == "stock_st":
                return pd.DataFrame()
            if api == "suspend_d":
                return pd.DataFrame()
            if api == "daily":
                qcodes = params.get("ts_code", "").split(",")
                frames = [daily[c] for c in qcodes if c in daily]
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if api == "daily_basic":
                # Both code paths must degrade if daily_basic_returns_empty.
                if daily_basic_returns_empty:
                    self.daily_basic_calls.append(str(trade_date))
                    return pd.DataFrame()
                # Healthy per-day snapshot.
                if trade_date is not None:
                    self.daily_basic_calls.append(str(trade_date))
                    frames = []
                    for c, df in daily.items():
                        sub = df[df["trade_date"].astype(str) == str(trade_date)][
                            ["ts_code", "trade_date", "turnover_rate", "circ_mv"]
                        ]
                        if not sub.empty:
                            frames.append(sub)
                    return (
                        pd.concat(frames, ignore_index=True)
                        if frames else pd.DataFrame()
                    )
                return pd.DataFrame()
            if api == "moneyflow":
                return pd.DataFrame()
            return pd.DataFrame()

    return _Fake()


class TestDailyBasicCoverageGuard:
    def test_empty_daily_basic_fails_screen(self, fresh_db):
        """daily_basic returns 0 rows ⇒ run must FAIL with a clear error,
        not succeed with n_after_mv=0."""
        tushare = _build_tushare(daily_basic_returns_empty=True)
        rt = ApwRuntime(
            db=fresh_db,
            config=_FakeConfig(),
            llms=_FakeLLMs(),  # type: ignore[arg-type]
            tushare=tushare,
        )
        runner = ApwRunner(rt, renderer=NullRenderer())
        outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))

        assert outcome.status.value == "failed", (
            f"expected failed status, got {outcome.status.value} "
            f"(summary={outcome.summary})"
        )
        assert outcome.error and "circ_mv" in outcome.error.lower() or (
            outcome.error and "daily_basic" in outcome.error.lower()
        ), f"error message must mention daily_basic/circ_mv: {outcome.error}"

    def test_healthy_daily_basic_passes(self, fresh_db):
        """Sanity: with healthy per-day data the guard stays out of the way."""
        tushare = _build_tushare(daily_basic_returns_empty=False)
        rt = ApwRuntime(
            db=fresh_db,
            config=_FakeConfig(),
            llms=_FakeLLMs(),  # type: ignore[arg-type]
            tushare=tushare,
        )
        runner = ApwRunner(rt, renderer=NullRenderer())
        outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))
        assert outcome.status.value in {"success", "partial_failed"}, outcome.error
