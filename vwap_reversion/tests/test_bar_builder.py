"""BarBuilder 差分语义（设计 §3.1，P1 验收项）。纯 stdlib。"""

from __future__ import annotations

import pytest

from vwap_reversion.feed.base import BarBuilder, CumulativeRegression
from vwap_reversion.schemas import Snapshot


def snap(ts: int, cv: float, ca: float, last: float = 1.0) -> Snapshot:
    return Snapshot(
        code="159518.SZ", trade_date="20260603", ts=ts,
        last=last, cum_vol=cv, cum_amount=ca,
    )


def test_first_snapshot_yields_since_open_bar() -> None:
    b = BarBuilder()
    bar = b.push(snap(1, 50_000, 52_000.0))
    assert bar is not None
    # 首条：区间 = 开盘至今（含集合竞价量）
    assert bar.interval_vol == 50_000
    assert bar.interval_amount == 52_000.0
    assert bar.cum_vol == 50_000


def test_diff_between_consecutive_snapshots() -> None:
    b = BarBuilder()
    b.push(snap(1, 50_000, 52_000.0))
    bar = b.push(snap(31, 53_000, 55_300.0, last=1.1))
    assert bar is not None
    assert bar.interval_vol == pytest.approx(3000)
    assert bar.interval_amount == pytest.approx(3300.0)
    assert bar.last == 1.1
    assert bar.ts == 31


def test_no_new_volume_returns_none_but_advances_baseline() -> None:
    b = BarBuilder()
    b.push(snap(1, 50_000, 52_000.0))
    assert b.push(snap(31, 50_000, 52_000.0)) is None  # 午休/无成交
    # 基线已推进到 ts=31；下一条只差出 31→61 的增量
    bar = b.push(snap(61, 50_100, 52_110.0))
    assert bar is not None
    assert bar.interval_vol == pytest.approx(100)


def test_cumulative_regression_raises_and_keeps_baseline() -> None:
    b = BarBuilder()
    b.push(snap(1, 50_000, 52_000.0))
    with pytest.raises(CumulativeRegression):
        b.push(snap(31, 40_000, 42_000.0))
    # 基线未被污染：下一条正常快照仍按旧基线差分
    bar = b.push(snap(61, 50_500, 52_550.0))
    assert bar is not None
    assert bar.interval_vol == pytest.approx(500)


def test_prime_resume_no_double_count() -> None:
    # 崩溃恢复：prime 最后一条已落库快照后，首条 bar 只含新增量，
    # 不会把「开盘至今」整段重复计入（那会让 Q_t 双计）。
    b = BarBuilder()
    b.prime(snap(100, 80_000, 83_000.0))
    bar = b.push(snap(130, 81_000, 84_100.0))
    assert bar is not None
    assert bar.interval_vol == pytest.approx(1000)
    assert bar.interval_amount == pytest.approx(1100.0)
