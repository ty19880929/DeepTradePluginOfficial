"""PR #2 — stats: 三档指标聚合 + 分组。"""

from __future__ import annotations

import pytest

from limit_up_board.winrate.persistence import PredictionRecord
from limit_up_board.winrate.resolver import ResolvedRecord
from limit_up_board.winrate.stats import (
    group_by_prediction,
    group_by_rank_bucket,
    summarize,
)


def _rec(ts: str, rank: int, prediction: str = "top_candidate") -> PredictionRecord:
    return PredictionRecord(
        trade_date="20260521",
        next_trade_date="20260522",
        ts_code=ts,
        name=ts,
        run_id="r",
        prediction=prediction,
        rank=rank,
        continuation_score=80.0,
        confidence="high",
        t_close_price=10.0,
        lgb_score=0.7,
        lgb_decile=9,
        raw_prediction_json=None,
    )


def _res(ts: str, rank: int, outcome: str, pct: float | None, prediction: str = "top_candidate") -> ResolvedRecord:
    return ResolvedRecord(
        record=_rec(ts, rank, prediction),
        t1_open_price=10.0 + (pct or 0) / 10.0,
        open_vs_limit_pct=pct,
        outcome=outcome,  # type: ignore[arg-type]
    )


def test_summary_basic() -> None:
    """5 胜 / 1 平 / 3 负 / 1 unresolved → 严格胜率 5/9, 非亏 6/9。"""
    resolved = (
        [_res(f"s{i}", 1, "win", 1.0) for i in range(5)]
        + [_res("f0", 1, "flat", 0.0)]
        + [_res(f"l{i}", 1, "loss", -1.0) for i in range(3)]
        + [_res("u0", 1, "unresolved", None)]
    )
    s = summarize(resolved)
    assert s.total == 10
    assert s.resolved == 9
    assert s.unresolved == 1
    assert s.win == 5
    assert s.flat == 1
    assert s.loss == 3
    assert s.strict_win_rate == pytest.approx(5 / 9)
    assert s.non_loss_rate == pytest.approx(6 / 9)


def test_summary_all_unresolved() -> None:
    """全部 unresolved → 胜率 / 平均都是 None。"""
    resolved = [_res("a", 1, "unresolved", None), _res("b", 2, "unresolved", None)]
    s = summarize(resolved)
    assert s.resolved == 0
    assert s.strict_win_rate is None
    assert s.non_loss_rate is None
    assert s.avg_open_vs_limit_pct is None


def test_summary_avg_only_resolved() -> None:
    """avg_open_vs_limit_pct 应只对已解析样本求均值。"""
    resolved = [
        _res("a", 1, "win", 2.0),
        _res("b", 1, "loss", -1.0),
        _res("c", 1, "unresolved", None),  # 不应进入分母
    ]
    s = summarize(resolved)
    assert s.avg_open_vs_limit_pct == pytest.approx((2.0 + -1.0) / 2)


def test_summary_empty() -> None:
    s = summarize([])
    assert s.total == 0
    assert s.resolved == 0
    assert s.strict_win_rate is None


def test_group_by_prediction_canonical_order() -> None:
    """分组按 top_candidate / watchlist / avoid 顺序。"""
    resolved = [
        _res("a", 1, "loss", -1.0, prediction="avoid"),
        _res("b", 1, "win", 1.0, prediction="top_candidate"),
        _res("c", 1, "win", 0.5, prediction="watchlist"),
        _res("d", 2, "win", 2.0, prediction="top_candidate"),
    ]
    groups = group_by_prediction(resolved)
    keys = [g.key for g in groups]
    assert keys == ["top_candidate", "watchlist", "avoid"]
    top = groups[0]
    assert top.total == 2
    assert top.win == 2
    assert top.strict_win_rate == pytest.approx(1.0)


def test_group_by_rank_bucket() -> None:
    resolved = [
        _res("a", 1, "win", 1.0),
        _res("b", 3, "win", 1.0),
        _res("c", 5, "loss", -1.0),
        _res("d", 10, "win", 0.5),
        _res("e", 11, "loss", -2.0),
        _res("f", 20, "unresolved", None),
    ]
    groups = group_by_rank_bucket(resolved)
    keys = [g.key for g in groups]
    assert keys == ["Top 1-3", "Top 4-10", "Top 11+"]
    assert groups[0].total == 2  # ranks 1, 3
    assert groups[1].total == 2  # ranks 5, 10
    assert groups[2].total == 2  # ranks 11, 20
    assert groups[2].resolved == 1
    assert groups[2].strict_win_rate == pytest.approx(0.0)
