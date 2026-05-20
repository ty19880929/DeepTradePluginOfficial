"""Round-2 P3 regression: ``min_circ_mv_yi`` / ``max_circ_mv_yi`` must filter
the universe during screen.

Before the fix the config keys were exposed via settings but never consulted
by the screen pipeline — users adjusting the band saw zero effect on the
candidate set. The fix wires daily_basic.circ_mv (万元) into a band check that
runs between the T-day amount filter and the per-stock funnel, surfacing the
post-filter count as ``DATA_SYNC_FINISHED.payload.n_after_mv``.

These tests exercise the helper + a thin end-to-end check that the screen
funnel honours the band.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from accumulation_probe_washout.config import ApwConfig, ApwConfigStore
from accumulation_probe_washout.runner import (
    ApwRunner,
    ScreenParams,
    _circ_mv_yi_on_date,
)
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


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestCircMvHelper:
    def test_converts_wan_to_yi(self) -> None:
        df = pd.DataFrame({
            "ts_code": ["600000.SH"],
            "trade_date": ["20240509"],
            "circ_mv": [500_000.0],  # 50 亿
        })
        out = _circ_mv_yi_on_date(df, "20240509")
        assert out == {"600000.SH": 50.0}

    def test_empty_frame_returns_empty(self) -> None:
        assert _circ_mv_yi_on_date(pd.DataFrame(), "20240509") == {}

    def test_missing_column_returns_empty(self) -> None:
        df = pd.DataFrame({"ts_code": ["x"], "trade_date": ["20240509"]})
        assert _circ_mv_yi_on_date(df, "20240509") == {}

    def test_filters_by_trade_date(self) -> None:
        df = pd.DataFrame({
            "ts_code": ["x", "x"],
            "trade_date": ["20240508", "20240509"],
            "circ_mv": [100_000.0, 200_000.0],
        })
        out = _circ_mv_yi_on_date(df, "20240509")
        assert out == {"x": 20.0}

    def test_skips_null_values(self) -> None:
        df = pd.DataFrame({
            "ts_code": ["a", "b"],
            "trade_date": ["20240509", "20240509"],
            "circ_mv": [None, 100_000.0],
        })
        out = _circ_mv_yi_on_date(df, "20240509")
        assert out == {"b": 10.0}


# ---------------------------------------------------------------------------
# End-to-end screen filter
# ---------------------------------------------------------------------------

class _FakeConfig:
    def get(self, k, default=None):
        return "test-token" if k == "tushare.token" else default
    def get_app_config(self):
        raise NotImplementedError


class _FakeLLMs:
    pass


def _build_tushare(circ_mv_by_code: dict[str, float]):
    """Build a fake Tushare client where each code's daily_basic.circ_mv is
    overridden per-row, otherwise inheriting make_quotes defaults."""
    daily: dict[str, pd.DataFrame] = {}
    for code in circ_mv_by_code:
        df = make_quotes(ts_code=code, pattern="flat", n=130)
        df["circ_mv"] = circ_mv_by_code[code]  # 万元
        daily[code] = df

    class _Fake:
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
                    for c in daily
                ])
            if api == "stock_st":
                return pd.DataFrame()
            if api == "suspend_d":
                return pd.DataFrame()
            if api == "daily":
                codes = params.get("ts_code", "").split(",")
                frames = [daily[c] for c in codes if c in daily]
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if api == "daily_basic":
                # New per-day shape: runner calls daily_basic(trade_date=...).
                if trade_date is not None:
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
                codes = params.get("ts_code", "").split(",")
                frames = [
                    daily[c][["ts_code", "trade_date", "turnover_rate", "circ_mv"]]
                    for c in codes if c in daily
                ]
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if api == "moneyflow":
                return pd.DataFrame()
            return pd.DataFrame()
    return _Fake()


def _last_data_sync_payload(db, run_id: str) -> dict:
    rows = db.fetchall(
        "SELECT payload_json FROM apw_events "
        "WHERE run_id = ? AND event_type = 'data.sync.finished' "
        "ORDER BY seq DESC",
        [run_id],
    )
    assert rows, "expected at least one data_sync_finished event"
    return json.loads(rows[0][0])


def test_band_excludes_tiny_and_huge_caps(fresh_db):
    # 600000 = 10 亿 (below default 20 亿 floor) → excluded
    # 600002 = 200 亿 (in band 20..1500) → kept
    # 600003 = 2000 亿 (above default 1500 亿 cap) → excluded
    circ_mv_wan = {
        "600000.SH": 100_000.0,
        "600002.SH": 2_000_000.0,
        "600003.SH": 20_000_000.0,
    }
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=_build_tushare(circ_mv_wan),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))
    assert outcome.status.value in {"success", "partial_failed"}, outcome.error

    payload = _last_data_sync_payload(fresh_db, outcome.run_id)
    # Liquidity passes all three (make_quotes amount is well above 1 亿).
    assert payload["n_after_liquidity"] == 3
    # Only the mid-cap survives the mv band.
    assert payload["n_after_mv"] == 1


def test_band_widens_when_user_adjusts_settings(fresh_db):
    """Persisting a looser ``max_circ_mv_yi`` via ApwConfigStore must propagate
    into screen — proving the setting is now actually consumed."""
    circ_mv_wan = {
        "600002.SH": 2_000_000.0,    # 200 亿
        "600003.SH": 20_000_000.0,   # 2000 亿
    }
    store = ApwConfigStore(fresh_db)
    store.set("max_circ_mv_yi", 5000.0)  # widen the cap

    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=_build_tushare(circ_mv_wan),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))
    payload = _last_data_sync_payload(fresh_db, outcome.run_id)
    # Both pass with the widened cap.
    assert payload["n_after_mv"] == 2


def test_band_boundary_inclusive(fresh_db):
    """Codes whose circ_mv equals the floor / cap exactly must pass."""
    circ_mv_wan = {
        "600002.SH": 200_000.0,      # 20 亿 — equal to default floor
        "600003.SH": 15_000_000.0,   # 1500 亿 — equal to default cap
    }
    rt = ApwRuntime(
        db=fresh_db,
        config=_FakeConfig(),
        llms=_FakeLLMs(),  # type: ignore[arg-type]
        tushare=_build_tushare(circ_mv_wan),
    )
    runner = ApwRunner(rt, renderer=NullRenderer())
    outcome = runner.execute_screen(ScreenParams(trade_date="20240509"))
    payload = _last_data_sync_payload(fresh_db, outcome.run_id)
    assert payload["n_after_mv"] == 2
