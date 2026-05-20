"""Tests for checkmate.data — Tushare thin wrappers + parquet cache.

Goals (per iteration_tasks.md §2 PR-1.1):
  * cache hit (no Tushare re-call on second invocation)
  * parquet roundtrip (read what was written, identical rows)
  * qfq / raw双轨独立 — one cache feeds both views, but they return distinct
    price columns
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from checkmate import data


# ---------------------------------------------------------------------------
# Stub Tushare client
# ---------------------------------------------------------------------------


class _StubTushare:
    """Minimal duck-type: ``call(api_name, **kwargs) -> DataFrame``.

    ``frames`` is keyed by api_name; the value is either a DataFrame (returned
    verbatim) or a callable that receives kwargs and returns the frame.
    """

    def __init__(self, frames: dict[str, object]) -> None:
        self._frames = frames
        self.calls: list[tuple[str, dict]] = []

    def call(self, api_name: str, **kwargs: object) -> pd.DataFrame:
        self.calls.append((api_name, dict(kwargs)))
        producer = self._frames.get(api_name)
        if producer is None:
            return pd.DataFrame()
        if callable(producer):
            return producer(**kwargs).copy()
        return producer.copy()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code":    ["600519.SH"] * 5,
        "trade_date": ["20240102", "20240103", "20240104", "20240105", "20240108"],
        "open":       [100.0, 101.0, 102.0, 103.0, 104.0],
        "high":       [102.0, 103.0, 104.0, 105.0, 106.0],
        "low":        [ 99.0, 100.0, 101.0, 102.0, 103.0],
        "close":      [101.0, 102.0, 103.0, 104.0, 105.0],
        "pre_close":  [100.0, 101.0, 102.0, 103.0, 104.0],
    })


@pytest.fixture
def adj_factor_df() -> pd.DataFrame:
    # A 1:2 split between 0103 and 0104 doubles adj_factor (latest > earliest).
    return pd.DataFrame({
        "ts_code":    ["600519.SH"] * 5,
        "trade_date": ["20240102", "20240103", "20240104", "20240105", "20240108"],
        "adj_factor": [1.0, 1.0, 2.0, 2.0, 2.0],
    })


@pytest.fixture
def stub(daily_df: pd.DataFrame, adj_factor_df: pd.DataFrame) -> _StubTushare:
    return _StubTushare({"daily": daily_df, "adj_factor": adj_factor_df})


# ---------------------------------------------------------------------------
# Cache hit / parquet roundtrip
# ---------------------------------------------------------------------------


def test_fetch_daily_raw_writes_and_hits_cache(stub: _StubTushare, tmp_path: Path) -> None:
    out1 = data.fetch_daily_raw(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    assert len(out1) == 5
    cache_file = tmp_path / "600519.SH.parquet"
    assert cache_file.is_file()
    # Two API calls in the first invocation: daily + adj_factor.
    assert [c[0] for c in stub.calls] == ["daily", "adj_factor"]

    # Second invocation in the same window: no new Tushare calls.
    out2 = data.fetch_daily_raw(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    assert len(out2) == 5
    assert [c[0] for c in stub.calls] == ["daily", "adj_factor"]
    # Parquet roundtrip: rows are identical
    pd.testing.assert_frame_equal(
        out1.reset_index(drop=True), out2.reset_index(drop=True), check_like=True,
    )


def test_force_refresh_bypasses_cache(stub: _StubTushare, tmp_path: Path) -> None:
    data.fetch_daily_raw(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    data.fetch_daily_raw(
        stub, "600519.SH", "20240102", "20240108",
        force_refresh=True, cache_root=tmp_path,
    )
    # 4 calls = 2 fetches × (daily + adj_factor)
    assert len(stub.calls) == 4


def test_fetch_daily_qfq_reuses_same_cache_as_raw(stub: _StubTushare, tmp_path: Path) -> None:
    """A prior raw fetch populates the cache; qfq fetch must NOT re-pull."""
    data.fetch_daily_raw(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    calls_after_raw = list(stub.calls)
    qfq = data.fetch_daily_qfq(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    assert stub.calls == calls_after_raw  # zero extra API calls
    assert "close_qfq" in qfq.columns
    assert "close" in qfq.columns  # raw column preserved


def test_qfq_derivation_matches_adj_factor_ratio(
    stub: _StubTushare, tmp_path: Path, daily_df: pd.DataFrame, adj_factor_df: pd.DataFrame,
) -> None:
    out = data.fetch_daily_qfq(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    latest_adj = float(adj_factor_df.iloc[-1]["adj_factor"])  # 2.0
    expected_close_qfq = (
        daily_df["close"].astype(float).values * adj_factor_df["adj_factor"].astype(float).values
        / latest_adj
    )
    # close on the post-split day equals raw close (factor 2.0 / latest 2.0 = 1).
    # close on pre-split day is halved (factor 1.0 / latest 2.0 = 0.5).
    assert out["close_qfq"].iloc[0] == pytest.approx(expected_close_qfq[0])
    assert out["close_qfq"].iloc[-1] == pytest.approx(expected_close_qfq[-1])


def test_qfq_and_raw_return_independent_columns(stub: _StubTushare, tmp_path: Path) -> None:
    raw = data.fetch_daily_raw(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    qfq = data.fetch_daily_qfq(stub, "600519.SH", "20240102", "20240108", cache_root=tmp_path)
    assert "close_qfq" not in raw.columns  # raw view does not synthesise qfq cols
    assert "close_qfq" in qfq.columns
    # The qfq view's pre-split close differs from raw (split factor 2x).
    assert qfq["close_qfq"].iloc[0] != raw["close"].iloc[0]


def test_slice_respects_requested_window(stub: _StubTushare, tmp_path: Path) -> None:
    out = data.fetch_daily_raw(stub, "600519.SH", "20240103", "20240105", cache_root=tmp_path)
    assert list(out["trade_date"]) == ["20240103", "20240104", "20240105"]


# ---------------------------------------------------------------------------
# Other wrappers — minimal smoke
# ---------------------------------------------------------------------------


def test_fetch_daily_basic_cache(tmp_path: Path) -> None:
    db_df = pd.DataFrame({
        "ts_code":    ["600519.SH"] * 3,
        "trade_date": ["20240102", "20240103", "20240104"],
        "turnover_rate": [0.5, 0.6, 0.4],
        "amount":     [1.0e7, 1.1e7, 0.9e7],
        "total_mv":   [1.0e10, 1.0e10, 1.0e10],
    })
    stub = _StubTushare({"daily_basic": db_df})
    out = data.fetch_daily_basic(stub, "600519.SH", "20240102", "20240104", cache_root=tmp_path)
    assert len(out) == 3
    # Cache hit
    data.fetch_daily_basic(stub, "600519.SH", "20240102", "20240104", cache_root=tmp_path)
    assert len(stub.calls) == 1


def test_fetch_stk_limit_per_day_cache(tmp_path: Path) -> None:
    sl_df = pd.DataFrame({
        "ts_code":    ["600519.SH", "000001.SZ"],
        "trade_date": ["20240102", "20240102"],
        "up_limit":   [110.0, 11.0],
        "down_limit": [90.0, 9.0],
    })
    stub = _StubTushare({"stk_limit": sl_df})
    out = data.fetch_stk_limit(stub, "20240102", cache_root=tmp_path)
    assert len(out) == 2
    data.fetch_stk_limit(stub, "20240102", cache_root=tmp_path)
    assert len(stub.calls) == 1  # cache hit


def test_fetch_index_daily_cache(tmp_path: Path) -> None:
    idx_df = pd.DataFrame({
        "ts_code":    ["000001.SH"] * 3,
        "trade_date": ["20240102", "20240103", "20240104"],
        "close":      [3000.0, 3010.0, 3005.0],
    })
    stub = _StubTushare({"index_daily": idx_df})
    out = data.fetch_index_daily(stub, "000001.SH", "20240102", "20240104", cache_root=tmp_path)
    assert list(out["close"]) == [3000.0, 3010.0, 3005.0]
    data.fetch_index_daily(stub, "000001.SH", "20240102", "20240104", cache_root=tmp_path)
    assert len(stub.calls) == 1
