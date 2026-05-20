"""Regime classifier tests — one synthetic scenario per regime (PR-2.2).

Each scenario plants:
  * index_daily parquet for 中证全指 + 沪深300, shaped to control whether
    close > MA(120).
  * features_daily rows whose ``close_qfq > ma120`` ratio sets the breadth.

Then ``classify_regime`` is run and the resulting :class:`RegimeRow` is
checked against the expected (regime, exposure_cap) tuple.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from checkmate import paths
from checkmate.config import RegimeConfig
from checkmate.regime import (
    RegimeRow,
    classify_regime,
    compute_breadth_limit_down_5d,
    upsert_regime_daily,
)
from checkmate.runtime import CheckmateRuntime


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "migrations" / "20260520_001_init.sql"
)


# ---------------------------------------------------------------------------
# Stub Tushare — only index_daily is consulted (others use planted caches).
# ---------------------------------------------------------------------------


class _StubTushare:
    def call(self, api_name: str, **kwargs):
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_data_root", lambda: tmp_path / "checkmate")
    paths.ensure_layout()

    from deeptrade.core.db import Database  # noqa: PLC0415

    db = Database(tmp_path / "checkmate_test.duckdb")
    for stmt in MIGRATION_PATH.read_text(encoding="utf-8").split(";"):
        if stmt.strip():
            db.execute(stmt.strip())
    rt = CheckmateRuntime(db=db, config=None, tushare=_StubTushare())  # type: ignore[arg-type]
    yield rt
    db.close()


# ---------------------------------------------------------------------------
# Plant helpers
# ---------------------------------------------------------------------------


def _plant_index_above_ma(index_code: str, *, n: int = 150, base: float = 1000.0) -> None:
    """Plant a rising index series → close[-1] > MA(120)."""
    dates = pd.bdate_range(end="20240329", periods=n).strftime("%Y%m%d").tolist()
    closes = [base * (1.0 + 0.002 * i) for i in range(n)]  # +0.2%/day
    df = pd.DataFrame({
        "ts_code": [index_code] * n,
        "trade_date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
    })
    p = paths.index_daily_cache_dir() / f"{index_code}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _plant_index_below_ma(index_code: str, *, n: int = 150, base: float = 1000.0) -> None:
    """Plant a falling index series → close[-1] < MA(120)."""
    dates = pd.bdate_range(end="20240329", periods=n).strftime("%Y%m%d").tolist()
    closes = [base * (1.0 - 0.002 * i) for i in range(n)]
    df = pd.DataFrame({
        "ts_code": [index_code] * n,
        "trade_date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
    })
    p = paths.index_daily_cache_dir() / f"{index_code}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _seed_features_breadth(db, trade_date: str, n_total: int, n_above: int) -> None:
    """Seed ``n_total`` features rows where ``n_above`` have close_qfq > ma120."""
    for i in range(n_total):
        ts_code = f"TEST{i:04d}.SH"
        close = 10.0
        ma120 = 9.0 if i < n_above else 11.0  # close > ma120 for first n_above rows
        db.execute(
            """
            INSERT INTO checkmate_features_daily
                (trade_date, ts_code, close_qfq, ma120)
            VALUES (?, ?, ?, ?)
            """,
            [trade_date, ts_code, close, ma120],
        )


def _make_cfg(**overrides) -> RegimeConfig:
    base = RegimeConfig()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------------------------------------------------------------------------
# Four regimes, one scenario each
# ---------------------------------------------------------------------------


def test_regime_strong(rt) -> None:
    cfg = _make_cfg()
    _plant_index_above_ma(cfg.index_csi_code)
    _plant_index_above_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=70)  # 70%
    row = classify_regime(rt, "20240329", cfg)
    assert row.regime == "strong"
    assert row.exposure_cap == 1.0
    assert row.index_csi_above_ma120 is True
    assert row.index_hs300_above_ma120 is True
    assert row.breadth_ma120 == pytest.approx(0.7)


def test_regime_neutral_one_index_below(rt) -> None:
    cfg = _make_cfg()
    _plant_index_above_ma(cfg.index_csi_code)
    _plant_index_below_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=50)  # 50%
    row = classify_regime(rt, "20240329", cfg)
    assert row.regime == "neutral"
    assert row.exposure_cap == 0.6


def test_regime_neutral_strong_breadth_one_index(rt) -> None:
    """Even with 70% breadth, only 1 index above → neutral (not strong)."""
    cfg = _make_cfg()
    _plant_index_above_ma(cfg.index_csi_code)
    _plant_index_below_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=70)
    row = classify_regime(rt, "20240329", cfg)
    assert row.regime == "neutral"


def test_regime_weak(rt) -> None:
    cfg = _make_cfg()
    _plant_index_below_ma(cfg.index_csi_code)
    _plant_index_below_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=30)  # 30%
    row = classify_regime(rt, "20240329", cfg)
    assert row.regime == "weak"
    assert row.exposure_cap == 0.3


def test_regime_risk(rt) -> None:
    cfg = _make_cfg()
    _plant_index_below_ma(cfg.index_csi_code)
    _plant_index_below_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=10)  # 10% < 20%
    row = classify_regime(rt, "20240329", cfg)
    assert row.regime == "risk"
    assert row.exposure_cap == 0.0


# ---------------------------------------------------------------------------
# Degraded-data behaviour
# ---------------------------------------------------------------------------


def test_no_index_data_falls_through_to_weak(rt) -> None:
    """When index caches are absent, both flags are None and breadth alone
    drives the decision; mid breadth without index confirmation → weak."""
    cfg = _make_cfg()
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=50)
    row = classify_regime(rt, "20240329", cfg)
    assert row.regime == "weak"
    assert row.index_csi_above_ma120 is None
    assert row.index_hs300_above_ma120 is None


def test_no_features_breadth_defaults_neutral(rt) -> None:
    """With both indices above MA120 but the features table empty, breadth
    defaults to 0.5 → falls into the neutral branch (needs 1+ index above)."""
    cfg = _make_cfg()
    _plant_index_above_ma(cfg.index_csi_code)
    _plant_index_above_ma(cfg.index_hs300_code)
    row = classify_regime(rt, "20240329", cfg)
    assert row.breadth_ma120 is None
    assert row.regime == "neutral"  # 0.5 breadth, both indices above → neutral


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_upsert_regime_daily_writes_payload_json(rt) -> None:
    cfg = _make_cfg()
    _plant_index_above_ma(cfg.index_csi_code)
    _plant_index_above_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=70)
    row = classify_regime(rt, "20240329", cfg, breadth_limit_down_5d=0.04)
    upsert_regime_daily(rt.db, row)

    persisted = rt.db.execute(
        "SELECT regime, exposure_cap, breadth_ma120, breadth_limit_down_5d, payload_json "
        "FROM checkmate_regime_daily WHERE trade_date = ?",
        ["20240329"],
    ).fetchone()
    assert persisted is not None
    regime, exposure_cap, breadth, breadth_ld5, payload_json = persisted
    assert regime == "strong"
    assert exposure_cap == 1.0
    assert breadth == pytest.approx(0.7)
    assert breadth_ld5 == pytest.approx(0.04)
    payload = json.loads(payload_json)
    assert payload["index_csi"]["code"] == cfg.index_csi_code


def test_upsert_is_idempotent(rt) -> None:
    cfg = _make_cfg()
    _plant_index_above_ma(cfg.index_csi_code)
    _plant_index_above_ma(cfg.index_hs300_code)
    _seed_features_breadth(rt.db, "20240329", n_total=100, n_above=70)
    row = classify_regime(rt, "20240329", cfg)
    upsert_regime_daily(rt.db, row)
    upsert_regime_daily(rt.db, row)  # re-apply
    n = rt.db.execute("SELECT COUNT(*) FROM checkmate_regime_daily").fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# compute_breadth_limit_down_5d helper (used by PR-2.3 scan orchestration)
# ---------------------------------------------------------------------------


def test_compute_breadth_limit_down_5d_picks_up_planted_limit_days(rt) -> None:
    """One symbol with 2 down-limit days out of 5 → 2/5 = 0.4."""
    dates = pd.bdate_range(end="20240329", periods=10).strftime("%Y%m%d").tolist()
    # Build pct_chg series: 8 normal days + 2 down-limit days at the tail.
    closes = [10.0] * 10
    pre = [10.0] * 10
    # Force the last 2 entries to be < -9.7% drops.
    closes[-2] = 8.9   # pct = -10.1%
    pre[-2] = 9.9
    closes[-1] = 8.9
    pre[-1] = 9.9
    df = pd.DataFrame({
        "ts_code": ["X.SH"] * 10,
        "trade_date": dates,
        "close": closes,
        "pre_close": pre,
        "open": closes, "high": closes, "low": closes,
        "adj_factor": [1.0] * 10,
    })
    p = paths.daily_cache_dir() / "X.SH.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)

    out = compute_breadth_limit_down_5d(rt, "20240329", ["X.SH"])
    # Window=5 → tail of 5 rows includes both limit days → 2/5 = 0.4
    assert out is not None
    assert out == pytest.approx(0.4, abs=1e-6)


def test_compute_breadth_limit_down_5d_none_when_no_data(rt) -> None:
    """No cached daily → None (caller treats as 'unknown', not 0)."""
    assert compute_breadth_limit_down_5d(rt, "20240329", ["MISSING.SH"]) is None
    assert compute_breadth_limit_down_5d(rt, "20240329", []) is None
